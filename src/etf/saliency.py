"""SmoothGrad saliency for clip predictions (paper Section 3.10, Fig. 5; qualitative only).

Saliency is |d t_hat / d input| averaged over N noisy copies of the clip
(sigma on the [0, 1] scale). Per-frame importance is the spatial mean of the
saliency, min-max normalised within each clip for display. Heat-map stretching,
blur and alpha masking are cosmetic and do not affect the prediction.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CLIP_LEN
from .inference import load_model
from .train import resolve_device


def select_best_embryo(embryo_csv: str | Path) -> str:
    """Embryo with the smallest anchored-fit residual RMSE (ties: smallest max |residual|)."""
    best: tuple[tuple[float, float], str] | None = None
    with open(embryo_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (float(row["rmse_resid"]), float(row["max_abs_resid"]))
            if all(math.isfinite(v) for v in key) and (best is None or key < best[0]):
                best = (key, str(row["eid"]))
    if best is None:
        raise SystemExit(f"No valid rows in {embryo_csv}")
    return best[1]


def five_starts(n_frames: int, clip_len: int) -> list[int]:
    """Early, early-mid, mid, mid-late and late clip starts."""
    m = max(0, n_frames - clip_len)
    return [0, m // 4, m // 2, (3 * m) // 4, m]


def _smooth(a01: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a01.astype(np.float32)
    return F.avg_pool2d(torch.from_numpy(a01[None, None]).float(), kernel_size=k, stride=1, padding=k // 2)[0, 0].numpy().astype(np.float32)


def _stretch(a: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """Percentile stretch + gamma for heavy-tailed saliency."""
    a = a.astype(np.float32)
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    if v1 <= v0 + 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (np.clip((a - v0) / (v1 - v0 + 1e-12), 0.0, 1.0) ** gamma).astype(np.float32)


def _alpha(hm01: np.ndarray, thr: float, alpha_max: float, alpha_gamma: float = 1.2) -> np.ndarray:
    z = np.clip((hm01.astype(np.float32) - thr) / (1.0 - thr + 1e-12), 0.0, 1.0) ** alpha_gamma
    return np.clip(z * alpha_max, 0.0, alpha_max).astype(np.float32)


def compute_saliency(model, x0: torch.Tensor, n_samples: int, sigma: float, amp: bool) -> tuple[torch.Tensor, float, float]:
    """Mean |gradient| over ``n_samples`` noisy inputs (plain gradient if n_samples <= 1); also mean/SD of t_hat."""
    grads, preds = None, []
    noisy = n_samples > 1
    for _ in range(max(1, n_samples)):
        x = ((x0 + torch.randn_like(x0) * sigma).clamp(0.0, 1.0) if noisy else x0.clone()).detach().requires_grad_(True)
        with torch.inference_mode(False), torch.enable_grad():
            with torch.autocast(device_type=x.device.type, enabled=bool(amp and x.device.type == "cuda")):
                t = model(x)[0]
        model.zero_grad(set_to_none=True)
        t.backward()
        preds.append(float(t.detach().cpu()))
        g = x.grad.detach().abs().float()
        grads = g if grads is None else grads + g
    return grads / float(len(preds)), float(np.mean(preds)), float(np.std(preds, ddof=0))


def run_saliency(ckpt: str | Path, npy: str | Path, out_dir: str | Path, *, starts: Sequence[int] | None = None,
                 clip_len: int = CLIP_LEN, n_samples: int = 20, sigma: float = 0.01, hm_lo: float = 90.0, hm_hi: float = 99.5,
                 hm_gamma: float = 0.55, blur_k: int = 9, alpha_thr: float = 0.25, alpha_max: float = 0.98, use_ema: bool = True,
                 amp: bool = False, device: str = "auto", seed: int = 0, dpi: int = 220) -> Path:
    """Saliency grids and temporal-importance curves for the given clip starts (default: five starts)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = resolve_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.load(npy)
    if frames.ndim != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [T,H,W], got {frames.dtype} {frames.shape}")
    starts = list(starts) if starts else five_starts(frames.shape[0], clip_len)
    model, _ = load_model(ckpt, device=dev, use_ema=use_ema, mem_profile=None)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    method = "smoothgrad" if n_samples > 1 else "grad"
    curves = []
    for s_req in starts:
        s = int(np.clip(int(s_req), 0, max(0, frames.shape[0] - clip_len)))
        clip = frames[s:s + clip_len]
        if clip.shape[0] < clip_len:
            clip = np.concatenate([clip, np.repeat(clip[-1:], clip_len - clip.shape[0], axis=0)], axis=0)
        x0 = (torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0).unsqueeze(0).unsqueeze(2).to(dev)
        g, t_mean, t_std = compute_saliency(model, x0, n_samples, sigma, amp)
        g = g[0, :, 0].cpu().numpy().astype(np.float32)  # [L, H, W]
        imp = g.reshape(clip_len, -1).mean(axis=1).astype(np.float32)
        hms = np.stack([_smooth(_stretch(g[i], hm_lo, hm_hi, hm_gamma), blur_k) for i in range(clip_len)], axis=0)
        span = float(imp.max() - imp.min())
        imp01 = (imp - imp.min()) / (span + 1e-12) if float(np.max(imp)) > 1e-12 else np.zeros_like(imp)
        curves.append((s, imp01, t_mean, t_std))
        np.save(out_dir / f"saliency_imp_start{s}.npy", imp)

        ncol = 6
        nrow = int(np.ceil(clip_len / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.25, nrow * 2.25))
        for i, ax in enumerate(np.array(axes).reshape(-1)):
            ax.axis("off")
            if i < clip_len:
                ax.imshow(clip[i].astype(np.float32) / 255.0, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.imshow(hms[i], cmap="turbo", alpha=_alpha(hms[i], alpha_thr, alpha_max), vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"t={s + i}", fontsize=8)
        extra = f" ±{t_std:.3f}" if method == "smoothgrad" else ""
        fig.suptitle(f"{method} overlay | start={s}, t_hat={t_mean:.3f}{extra}h", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / f"saliency_grid_start{s}.png", dpi=dpi)
        plt.close(fig)

        fig = plt.figure(figsize=(7.4, 2.7))
        plt.plot(np.arange(clip_len), imp01, marker="o")
        plt.ylim(-0.02, 1.02)
        plt.xlabel(f"frame index within clip (0..{clip_len - 1})")
        plt.ylabel("normalized saliency mass (within clip)")
        plt.title(f"Temporal importance ({method}, start={s})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"saliency_time_start{s}.png", dpi=dpi)
        plt.close(fig)
        print(f"[saliency] start={s}: t_hat={t_mean:.3f} ± {t_std:.3f} h")

    if len(curves) >= 2:
        fig = plt.figure(figsize=(8.2, 3.2))
        for s, imp01, t_mean, t_std in curves:
            label = f"start={s}, t_hat={t_mean:.2f}" + (f"±{t_std:.2f}" if method == "smoothgrad" else "")
            plt.plot(np.arange(clip_len), imp01, marker="o", label=label)
        plt.ylim(-0.02, 1.02)
        plt.xlabel(f"frame index within clip (0..{clip_len - 1})")
        plt.ylabel("normalized saliency mass (within clip)")
        plt.title(f"Temporal importance overlay ({method})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        fig.savefig(out_dir / "saliency_time_overlay.png", dpi=dpi)
        plt.close(fig)
    print(f"[saliency] outputs in {out_dir}")
    return out_dir


def best_embryo_saliency(outroot: str | Path, model: str, *, ckpt: str | Path, proc_dir: str | Path, **kwargs) -> Path:
    """SmoothGrad on five clips of the 28.5 C test embryo with the most consistent trajectory (paper Fig. 5)."""
    from .constants import ID28C5

    outroot = Path(outroot)
    embryo_csv = outroot / ID28C5 / model / "embryo.csv"
    if not embryo_csv.exists():
        raise SystemExit(f"[ERR] missing {embryo_csv}; run `etf aggregate` first")
    eid = select_best_embryo(embryo_csv)
    print(f"[saliency] best {ID28C5} embryo for {model}: {eid}")
    return run_saliency(ckpt, Path(proc_dir) / f"{eid}.npy", outroot / f"vis_best_{eid}_{model}_smoothgrad_five", **kwargs)
