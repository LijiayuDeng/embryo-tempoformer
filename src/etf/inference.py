"""Embryo-level sliding-window inference (paper Sections 3.2 and 3.9).

For every window start s (stride 8) the model predicts t_hat(s) in hpf, which is
converted to a start-time estimate t0_hat(s) = t_hat(s) - DT * s. The per-embryo
JSON stores all t0_hat(s); ``etf aggregate`` turns them into tempo readouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .config import TrainConfig
from .constants import CLIP_LEN, DT_H, IMG_SIZE, STRIDE, T0_HPF, TRIM
from .data import split_ids
from .model import EmbryoTempoFormer, build_model
from .train import load_weights, resolve_device
from .utils import read_json, write_json


def load_model(ckpt_path: str | Path, *, device: torch.device, use_ema: bool = True,
               mem_profile: str | None = "lowmem") -> tuple[EmbryoTempoFormer, dict[str, Any]]:
    """Rebuild the architecture stored in a checkpoint, load (EMA) weights, move to ``device``.

    ``mem_profile`` only sets how many frames are encoded per CNN call in eval
    mode; the released results use ``lowmem``. ``None`` keeps the checkpoint's profile.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    profile = mem_profile or str(ckpt_cfg.get("mem_profile", "balanced"))
    model = build_model(TrainConfig.from_dict(ckpt_cfg, mem_profile=profile))
    load_weights(model, ckpt, use_ema=use_ema)
    model.to(device)
    model.eval()
    return model, ckpt_cfg


def trimmed_mean(x: Sequence[float], trim: float = TRIM) -> float:
    """Mean after dropping the lowest and highest ``trim`` fraction (plain mean if nothing is dropped)."""
    xs = np.sort(np.asarray(list(x), dtype=np.float64))
    if xs.size == 0:
        return float("nan")
    k = int(trim * xs.size)
    if k <= 0 or 2 * k >= xs.size:
        return float(xs.mean())
    return float(xs[k:-k].mean())


@torch.no_grad()
def predict_embryo(
    model: torch.nn.Module,
    frames_u8: np.ndarray,
    *,
    device: torch.device,
    clip_len: int = CLIP_LEN,
    stride: int = STRIDE,
    trim: float = TRIM,
    amp: bool = True,
) -> dict[str, Any]:
    """Sliding-window prediction for one embryo stack [T, H, W] (uint8).

    Returns ``t0_final`` (trimmed mean of t0_hat, a descriptive start-time
    diagnostic), ``starts``, ``t0_hats`` and MAE/RMSE of the implied trajectory
    against the nominal axis (internal consistency only).
    """
    model.eval()
    n_frames = int(frames_u8.shape[0])
    starts = list(range(0, max(1, n_frames - clip_len + 1), int(stride)))
    use_amp = amp and device.type == "cuda"
    t0_hats = []
    for s in starts:
        clip = np.asarray(frames_u8[s:s + clip_len], dtype=np.uint8)
        if clip.shape[0] != clip_len:
            clip = np.concatenate([clip, np.repeat(clip[-1:], clip_len - clip.shape[0], axis=0)], axis=0)
        x = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0).unsqueeze(2).float().to(device) / 255.0
        if use_amp:
            x = x.half()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            offset = float(model(x).item())
        t0_hats.append(offset - DT_H * s)

    t0_final = trimmed_mean(t0_hats, trim=trim)
    t = np.arange(n_frames, dtype=np.float64)
    err = (t0_final + DT_H * t) - (T0_HPF + DT_H * t)
    metrics = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}
    return {"t0_final": float(t0_final), "starts": starts, "t0_hats": t0_hats, "metrics": metrics}


def infer_file(
    model: torch.nn.Module,
    input_path: str | Path,
    out_json: str | Path,
    *,
    device: torch.device,
    clip_len: int = CLIP_LEN,
    stride: int = STRIDE,
    trim: float = TRIM,
    amp: bool = True,
    img_size: int = IMG_SIZE,
) -> dict[str, Any]:
    """Predict one embryo from a processed ``.npy`` or a raw ``.tif/.tiff``.

    TIFF stacks are preprocessed on the fly (percentile clipping and resizing) and
    keep their recorded length.
    """
    input_path = Path(input_path)
    if input_path.suffix.lower() in (".tif", ".tiff"):
        from .preprocess import preprocess_stack, read_tiff_stack

        frames, meta = preprocess_stack(read_tiff_stack(input_path), img_size=img_size, pad_to_expect=False)
        info: dict[str, Any] = {"source": "tiff", "meta": meta}
    else:
        frames = np.load(input_path, mmap_mode=None)
        info = {"source": "npy"}
    out = predict_embryo(model, frames, device=device, clip_len=clip_len, stride=stride, trim=trim, amp=amp)
    out.update(info)
    write_json(out, out_json)
    return out


STACK_SUFFIXES = (".npy", ".tif", ".tiff")


def _check_settings(result_dir: Path, settings: dict[str, Any], force: bool) -> None:
    """Refuse to mix per-embryo results computed with different settings in one directory.

    ``infer_settings.json`` records how the JSON files in ``result_dir`` were made;
    existing files are only reused when the requested settings are the same.
    """
    path = result_dir / "infer_settings.json"
    if path.exists() and not force:
        saved = read_json(path)
        if saved != settings:
            diff = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v}
            raise SystemExit(f"[ERR] {result_dir} holds results computed with other settings {diff} (saved, requested); "
                             "use --force to recompute or choose another --outroot")
    write_json(settings, path)


def _embryo_inputs(proc_dir: Path, split_json: Path | None, max_eids: int) -> list[tuple[str, Path]]:
    """(eid, input file) pairs: the ``test`` list of a split file, or every stack in ``proc_dir``."""
    if split_json is not None:
        return [(eid, proc_dir / f"{eid}.npy") for eid in split_ids(split_json, "test", max_eids)]
    files = sorted(p for p in proc_dir.iterdir() if p.suffix.lower() in STACK_SUFFIXES)
    return [(p.stem, p) for p in (files[:max_eids] if max_eids > 0 else files)]


def infer_all(
    *,
    outroot: str | Path,
    datasets: Mapping[str, tuple[Path, Path | None]],
    checkpoints: Mapping[str, Path],
    clip_len: int = CLIP_LEN,
    stride: int = STRIDE,
    device: str = "auto",
    amp: bool = True,
    use_ema: bool = True,
    force: bool = False,
    max_eids: int = 0,
    log: Callable[[str], None] = print,
) -> None:
    """Infer every embryo of each dataset with each model.

    ``datasets`` maps a name to (directory, split file): the split's ``test`` list is
    used, or every .npy/.tif stack in the directory when the split file is None.
    ``checkpoints`` maps model tag -> checkpoint path. Writes
    ``<outroot>/<dataset>/<model>/json/<eid>.json``; existing files are kept unless
    ``force``. Each model is loaded once.
    """
    outroot = Path(outroot)
    dev = resolve_device(device)
    jobs = {tag: _embryo_inputs(Path(proc), None if split is None else Path(split), max_eids) for tag, (proc, split) in datasets.items()}
    for model_tag, ckpt in checkpoints.items():
        model = None
        for tag, inputs in jobs.items():
            json_dir = outroot / tag / model_tag / "json"
            _check_settings(json_dir.parent, {"checkpoint": Path(ckpt).name, "clip_len": clip_len, "stride": stride,
                                              "use_ema": bool(use_ema), "amp": bool(amp), "mem_profile": "lowmem"}, force)
            json_dir.mkdir(parents=True, exist_ok=True)
            todo = [(eid, path) for eid, path in inputs if force or not (json_dir / f"{eid}.json").exists()]
            log(f"[infer] {tag} / {model_tag}: {len(todo)} of {len(inputs)} embryos to run")
            for eid, path in todo:
                if not path.exists():
                    log(f"[WARN] missing {path}")
                    continue
                if model is None:
                    model, _ = load_model(ckpt, device=dev, use_ema=use_ema)
                infer_file(model, path, json_dir / f"{eid}.json", device=dev, clip_len=clip_len, stride=stride, amp=amp)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
