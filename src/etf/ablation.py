"""Remove-one-family augmentation ablation for ETF-full (paper Section 4.3, Table 4).

Five matched single-seed retrainings that differ only in the augmentation
families switched off. ``no_temporal_sampling`` removes frame-drop augmentation
and the start-index jitter (it does not disable the model's temporal token dropout).
Each setting is trained in its own process with its own log.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .constants import MAIN_DATASETS, STRIDE
from .utils import fmt, markdown_table, read_json, write_csv, write_json

# setting -> (augmentation families disabled, start-index jitter)
SETTINGS: dict[str, tuple[str, int]] = {
    "baseline_full": ("", 2),
    "no_spatial": ("spatial", 2),
    "no_photometric": ("photometric", 2),
    "no_acquisition": ("acquisition", 2),
    "no_temporal_sampling": ("temporal", 0),
}

# Training settings of the Table 4 runs (single NVIDIA L40S, 46 GB); everything else is the
# released ETF-full configuration. Smaller batches fit smaller GPUs but change the results.
TABLE4_OVERRIDES: dict[str, Any] = dict(batch_size=88, val_batch_size=176, num_workers=16, val_cache_items=64, save_every=25, patience=20)


def train_settings(outroot: str | Path, settings: Sequence[str], *, proc_dir: str | Path, split_json: str | Path,
                   overrides: dict[str, Any] | None = None, dry_run: bool = False) -> None:
    """Train each setting with ``python -m etf train --config <setting>/train_config.json`` (output tee'd to logs/)."""
    from .config import TrainConfig

    unknown = [s for s in settings if s not in SETTINGS]
    if unknown:
        raise SystemExit(f"[ERR] unknown setting(s) {unknown}; choose from {', '.join(SETTINGS)}")
    outroot = Path(outroot)
    (outroot / "logs").mkdir(parents=True, exist_ok=True)
    manifest = outroot / "manifest.tsv"
    manifest.write_text("setting\tout_dir\tlog_file\taug_disable_groups\tjitter\tseed\n", encoding="utf-8")
    for setting in settings:
        groups, jitter = SETTINGS[setting]
        out_dir, log_file = outroot / setting, outroot / "logs" / f"{setting}.log"
        cfg = TrainConfig(proc_dir=str(proc_dir), split_json=str(split_json), out_dir=str(out_dir),
                          aug_disable_groups=groups, jitter=jitter, **{**TABLE4_OVERRIDES, **(overrides or {})})
        with open(manifest, "a", encoding="utf-8") as f:
            f.write("\t".join((setting, str(out_dir), str(log_file), groups or "none", str(jitter), str(cfg.seed))) + "\n")
        cfg_path = out_dir / "train_config.json"
        write_json(dataclasses.asdict(cfg), cfg_path)
        cmd = [sys.executable, "-m", "etf", "train", "--config", str(cfg_path)]
        print(f"\n[ablation] {setting}: aug_disable_groups={groups or '<none>'} jitter={jitter} -> {out_dir}")
        if dry_run:
            print("[dry-run] " + shlex.join(cmd))
            continue
        with open(log_file, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            if proc.wait() != 0:
                raise SystemExit(f"[ERR] training failed for {setting}; see {log_file}")


def evaluate_settings(outroot: str | Path, settings: Sequence[str], *, datasets: dict[str, tuple[Path, Path]], model: str = "full",
                      dt: float, t0: float, clip_len: int, stride: int = STRIDE, device: str = "auto", amp: bool = True,
                      use_ema: bool = True, force: bool = True) -> None:
    """Infer and aggregate ``<setting>/best.pt`` on the main test sets into ``<setting>/eval_main``."""
    from .inference import infer_all
    from .tempo import aggregate_outroot

    for setting in settings:
        ckpt = Path(outroot) / setting / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"[ERR] missing checkpoint for {setting}: {ckpt}")
        eval_root = Path(outroot) / setting / "eval_main"
        print(f"[ablation] evaluate {setting}")
        infer_all(outroot=eval_root, datasets=datasets, checkpoints={model: ckpt}, clip_len=clip_len, stride=stride,
                  device=device, amp=amp, use_ema=use_ema, force=force)
        aggregate_outroot(eval_root, list(datasets), [model], dt=dt, t0=t0, force=True)


def _best_val(history_csv: Path) -> dict[str, str]:
    import csv

    best: dict[str, str] | None = None
    if history_csv.exists():
        best_mae = float("inf")
        with open(history_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    mae = float(row["val_mae"])
                except (TypeError, ValueError):
                    continue
                if mae < best_mae:
                    best_mae, best = mae, row
    best = best or {}
    return {"best_epoch": best.get("epoch", ""), "best_val_mae": best.get("val_mae", ""),
            "best_val_rmse": best.get("val_rmse", ""), "best_val_r2": best.get("val_r2", "")}


def summarize_ablation(outroot: str | Path, settings: Sequence[str], *, datasets: Sequence[str] = MAIN_DATASETS, model: str = "full",
                       eval_dirname: str = "eval_main") -> list[dict[str, Any]]:
    """min_aug_ablation_summary.{csv,md}: best validation epoch and test readouts per setting x dataset."""
    import csv

    outroot = Path(outroot)
    manifest: dict[str, dict[str, str]] = {}
    if (outroot / "manifest.tsv").exists():
        with open(outroot / "manifest.tsv", newline="", encoding="utf-8") as f:
            manifest = {r["setting"].strip(): r for r in csv.DictReader(f, delimiter="\t") if (r.get("setting") or "").strip()}
    rows = []
    for setting in settings:
        best = _best_val(outroot / setting / "history.csv")
        mf = manifest.get(setting, {})
        for ds in datasets:
            p = outroot / setting / eval_dirname / ds / model / "summary.json"
            if not p.exists():
                continue
            s = read_json(p)
            gm, fa = s.get("global_metrics_points", {}), s.get("fit_anchor_T0", {})
            rows.append({
                "setting": setting, "dataset": ds, "model": model, "aug_disable_groups": mf.get("aug_disable_groups", ""),
                "jitter": mf.get("jitter", ""), **best, "n_embryos": gm.get("n_embryos", ""), "n_points": gm.get("n_points", ""),
                "mae_h": gm.get("mae", ""), "rmse_h": gm.get("rmse", ""), "r2_points": gm.get("r2", ""), "m_anchor": fa.get("m", ""),
                "rmse_resid_h": fa.get("rmse_resid", ""), "max_abs_resid_h": fa.get("max_abs_resid", ""),
            })
    if not rows:
        raise SystemExit(f"No aggregated evaluation summaries found under {outroot}")
    write_csv(outroot / "min_aug_ablation_summary.csv", rows, list(rows[0].keys()))
    headers = ["setting", "dataset", "aug_disable_groups", "jitter", "best val MAE", "test MAE (h)", "test RMSE (h)", "test R2",
               "m_anchor", "rmse_resid (h)", "max_abs_resid (h)"]
    pretty = [dict(zip(headers, [r["setting"], r["dataset"], r["aug_disable_groups"], r["jitter"], fmt(r["best_val_mae"]),
                                 fmt(r["mae_h"]), fmt(r["rmse_h"]), fmt(r["r2_points"]), fmt(r["m_anchor"]), fmt(r["rmse_resid_h"]),
                                 fmt(r["max_abs_resid_h"])])) for r in rows]
    (outroot / "min_aug_ablation_summary.md").write_text(markdown_table(pretty, headers), encoding="utf-8")
    print(f"[ablation] {outroot / 'min_aug_ablation_summary.csv'}")
    return rows
