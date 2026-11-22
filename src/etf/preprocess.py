"""Raw stacks -> processed uint8 arrays, and split files.

Processing (paper Section 3.1): percentile clipping (default 1-99) and linear
mapping to [0, 255], bilinear resize to 384 x 384, and padding (repeat last
frame) or trimming of the time axis to 192 frames. One ``<eid>.npy`` per embryo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXPECT_T, IMG_SIZE
from .data import is_excluded
from .utils import write_json

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def read_tiff_stack(path: str | Path, max_pages: int | None = None) -> np.ndarray:
    """Read a (multi-page) TIFF / OME-TIFF as [T, H, W]."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        pages = tf.pages[:max_pages] if max_pages is not None else tf.pages
        arrs = [p.asarray() for p in pages]
    if not arrs:
        raise ValueError(f"Empty TIFF: {path}")
    for i, a in enumerate(arrs):
        if a.shape != arrs[0].shape:
            raise ValueError(f"Inconsistent TIFF pages in {path}: {arrs[0].shape} vs {a.shape} at page {i}")
    return np.stack(arrs, axis=0)


def percentile_clip_to_u8(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[np.ndarray, dict[str, float]]:
    """Clip intensities to the [p_lo, p_hi] percentiles of the whole stack and map to uint8."""
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    x = np.clip(arr.astype(np.float32), lo, hi)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8), {"p_lo": lo, "p_hi": hi}


def resize_stack_u8(frames_u8: np.ndarray, img_size: int) -> np.ndarray:
    """Bilinear resize of a uint8 [T, H, W] stack to [T, img_size, img_size]."""
    if frames_u8.ndim != 3 or frames_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 [T,H,W]")
    _, h, w = frames_u8.shape
    if h == img_size and w == img_size:
        return frames_u8
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for resizing")
    return np.stack(
        [np.asarray(Image.fromarray(f).resize((img_size, img_size), resample=Image.BILINEAR), dtype=np.uint8) for f in frames_u8],
        axis=0,
    )


def pad_or_trim_T(frames_u8: np.ndarray, expect_t: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Trim, or pad by repeating the last frame, so that T == expect_t."""
    t = int(frames_u8.shape[0])
    meta: dict[str, Any] = {"orig_T": t, "expect_t": int(expect_t), "action": "none"}
    if t == expect_t:
        return frames_u8, meta
    if t > expect_t:
        meta["action"] = "trim"
        return frames_u8[:expect_t], meta
    meta["action"] = "pad_last"
    return np.concatenate([frames_u8, np.repeat(frames_u8[-1:], expect_t - t, axis=0)], axis=0), meta


def preprocess_stack(
    raw: np.ndarray,
    expect_t: int = EXPECT_T,
    img_size: int = IMG_SIZE,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    pad_to_expect: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full preprocessing of one grayscale stack; returns (uint8 [T, S, S], metadata)."""
    u8, clip_meta = percentile_clip_to_u8(raw, p_lo=p_lo, p_hi=p_hi)
    u8 = resize_stack_u8(u8, img_size=img_size)
    if pad_to_expect:
        u8, t_meta = pad_or_trim_T(u8, expect_t=expect_t)
    else:
        t_meta = {"orig_T": int(u8.shape[0]), "expect_t": int(expect_t), "action": "keep_native"}
    return u8, {"clip": clip_meta, "t": t_meta, "img_size": int(img_size)}


def save_proc_npy(proc_dir: Path, eid: str, frames_u8: np.ndarray) -> Path:
    proc_dir = Path(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    out = proc_dir / f"{eid}.npy"
    np.save(out, frames_u8, allow_pickle=False)
    return out


def preprocess_tiff_dir(
    in_dir: str | Path,
    proc_dir: str | Path,
    expect_t: int = EXPECT_T,
    img_size: int = IMG_SIZE,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    max_pages: int = 0,
    limit: int = 0,
    pad_to_expect: bool = True,
) -> int:
    """Preprocess every ``*.tif*`` in ``in_dir`` (non-recursive) into ``proc_dir``.

    Excluded S-BIAD531 wells are skipped. Writes ``preprocess_meta.json``. Stacks are padded
    (last frame repeated) or trimmed to ``expect_t`` frames as for the released arrays; use
    ``pad_to_expect=False`` to keep the recorded length (recommended for new recordings).
    Returns the number of embryos written.
    """
    in_dir, proc_dir = Path(in_dir), Path(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    tiffs = sorted(in_dir.glob("*.tif*"))
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files in {in_dir}")

    meta_all: dict[str, Any] = {}
    for path in tiffs:
        eid = path.stem
        if is_excluded(eid):
            continue
        raw = read_tiff_stack(path, max_pages=max_pages if max_pages > 0 else None)
        proc, meta = preprocess_stack(raw, expect_t=expect_t, img_size=img_size, p_lo=p_lo, p_hi=p_hi, pad_to_expect=pad_to_expect)
        save_proc_npy(proc_dir, eid, proc)
        meta_all[eid] = meta
        if limit > 0 and len(meta_all) >= limit:
            break

    args = dict(in_dir=str(in_dir), proc_dir=str(proc_dir), expect_t=expect_t, img_size=img_size,
                p_lo=p_lo, p_hi=p_hi, max_pages=max_pages, limit=limit, pad_to_expect=pad_to_expect)
    write_json({"meta": meta_all, "args": args}, proc_dir / "preprocess_meta.json")
    print(f"[done] saved {len(meta_all)} embryos to {proc_dir}")
    return len(meta_all)


def make_split(proc_dir: str | Path, out_json: str | Path, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> dict[str, list[str]]:
    """Random embryo-level train/val/test split of ``<proc_dir>/*.npy`` (excluded wells removed)."""
    ids = [p.stem for p in sorted(Path(proc_dir).glob("*.npy")) if not is_excluded(p.stem)]
    if not ids:
        raise ValueError(f"No embryos found in {proc_dir}")
    np.random.default_rng(seed).shuffle(ids)

    n = len(ids)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_train = max(1, n - n_val - n_test)
    split = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}
    write_json(split, out_json)
    print(f"[split] n={n} train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} -> {out_json}")
    return split
