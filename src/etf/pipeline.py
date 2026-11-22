"""``etf reproduce``: the complete analysis of the paper from the released checkpoints."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import Settings, check
from .constants import ID28C5, MAIN_DATASETS, MODELS
from .utils import write_json

STEPS = ("infer", "aggregate", "stats", "power-continuous", "cliplen", "stage-error", "anchor", "stage-tempo", "figures")
CLIP_LENS = (4, 12, 24)  # clip-length sensitivity: 1, 3 and 6 h of context


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def write_run_info(outroot: Path, settings: Settings, datasets: dict, checkpoints: dict, extra: dict | None = None) -> None:
    import torch

    from .train import resolve_device

    device = resolve_device(settings.device)

    write_json({
        "etf_version": __version__,
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "settings": {"dt_h": settings.dt_h, "t0_hpf": settings.t0_hpf, "clip_len": settings.clip_len, "stride": settings.stride,
                     "device": settings.device, "amp": settings.amp, "use_ema": settings.use_ema},
        "datasets": {k: [str(p) for p in v] for k, v in datasets.items()},
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        **(extra or {}),
    }, outroot / "run_info.json")


def reproduce(settings: Settings, outroot: str | Path | None = None, *, skip: Sequence[str] = (), saliency: bool = False,
              saliency_model: str = "full", force: bool = False) -> Path:
    """Inference -> tempo readouts -> statistics -> sensitivity analyses -> figures and tables."""
    from .analyses import anchor_sensitivity, cliplen_sensitivity, context_fit, stage_error_bins, stage_tempo
    from .figures import make_paper_figures, paper_tables
    from .inference import infer_all
    from .stats import power_continuous, run_stats
    from .tempo import aggregate_outroot

    unknown = set(skip) - set(STEPS)
    if unknown:
        raise SystemExit(f"[ERR] unknown step(s) to skip: {', '.join(sorted(unknown))}; steps are {', '.join(STEPS)}")
    outroot = Path(outroot) if outroot else settings.runs_dir / f"paper_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    check(settings)
    datasets = settings.datasets(list(MAIN_DATASETS))
    checkpoints = settings.checkpoints(list(MODELS))
    outroot.mkdir(parents=True, exist_ok=True)
    write_run_info(outroot, settings, datasets, checkpoints, {"command": "reproduce", "skipped_steps": sorted(skip)})
    runtime = dict(device=settings.device, amp=settings.amp, use_ema=settings.use_ema)
    t_start = time.time()

    def step(name: str) -> bool:
        if name in skip:
            return False
        print(f"\n===== {name} =====")
        return True

    if step("infer"):
        infer_all(outroot=outroot, datasets=datasets, checkpoints=checkpoints, clip_len=settings.clip_len, stride=settings.stride,
                  force=force, **runtime)
    if step("aggregate"):
        aggregate_outroot(outroot, MAIN_DATASETS, MODELS, dt=settings.dt_h, t0=settings.t0_hpf, force=force)
    if step("stats"):
        run_stats(outroot)
    if step("power-continuous"):
        power_continuous(outroot)
    if step("cliplen"):
        cl_root = outroot / "cliplen_sensitivity"
        csv_path = cliplen_sensitivity(cl_root, datasets=datasets, checkpoints={"full": checkpoints["full"]}, clip_lens=CLIP_LENS,
                                       stride=settings.stride, dt=settings.dt_h, t0=settings.t0_hpf, force=force, **runtime)
        context_fit(outroot, csv_path, cl_root / "context_fit", clip_lens=CLIP_LENS, dt=settings.dt_h)
    if step("stage-error"):
        stage_error_bins(outroot)
    if step("anchor"):
        anchor_sensitivity(outroot, dt=settings.dt_h)
    if step("stage-tempo"):
        stage_tempo(outroot)
    if step("figures"):
        if all((outroot / f"CI_{m}_m_anchor.json").exists() for m in MODELS):
            make_paper_figures(outroot)
            paper_tables(outroot)
        else:
            print("[figures] skipped: Figs. 2-4 and the tables need the outputs of the stats step")
    if saliency:
        print("\n===== saliency =====")
        from .saliency import best_embryo_saliency

        best_embryo_saliency(outroot, saliency_model, ckpt=checkpoints[saliency_model], proc_dir=datasets[ID28C5][0],
                             clip_len=settings.clip_len, device=settings.device, use_ema=settings.use_ema)
    print(f"\n[done] {time.time() - t_start:.0f} s; all outputs under {outroot}")
    return outroot
