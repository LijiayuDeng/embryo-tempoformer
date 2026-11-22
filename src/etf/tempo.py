"""Embryo-level tempo readouts from per-embryo inference JSON (paper Section 3.9).

For each window start s:  x = T0 + DT * s (nominal time),  y = t0_hat(s) + DT * s
(predicted developmental time). Within each embryo the anchored model
    y - T0 = m * (x - T0)
is fitted by least squares; m_anchor < 1 means slower tempo than the nominal
axis. Residual RMSE (rmse_resid) and the maximum absolute residual summarise how
well the windows of one embryo follow a single anchored trajectory.

Outputs per <dataset>/<model> directory:
  points.csv   one row per (embryo, window)
  embryo.csv   eid, n_points, m_anchor, rmse_resid, max_abs_resid
  summary.json pooled window-level metrics (descriptive), pooled anchored fit,
               and the distribution of per-embryo m_anchor
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np

from .constants import DT_H, T0_HPF
from .stats import describe


def fit_anchor(x: np.ndarray, y: np.ndarray, t0: float) -> tuple[float, np.ndarray]:
    """Least-squares slope of (y - t0) = m (x - t0) and its residuals (nan if undefined)."""
    xp, yp = x - t0, y - t0
    msk = np.abs(xp) > 1e-12
    xp2, yp2 = xp[msk], yp[msk]
    if xp2.size < 2:
        return float("nan"), np.full_like(yp, np.nan, dtype=np.float64)
    denom = float(np.sum(xp2 * xp2))
    if denom <= 1e-12:
        return float("nan"), np.full_like(yp, np.nan, dtype=np.float64)
    m = float(np.sum(xp2 * yp2) / denom)
    return m, yp - m * xp


def rmse(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def _max_abs(a: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else float("nan")


def aggregate_json_dir(json_dir: str | Path, out_dir: str | Path, dt: float = DT_H, t0: float = T0_HPF, verbose: bool = True) -> dict:
    """Aggregate ``<json_dir>/*.json`` into points.csv, embryo.csv and summary.json in ``out_dir``."""
    json_dir, out_dir = Path(json_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No json files under: {json_dir}")

    point_rows: list[tuple[str, int, float, float, float]] = []  # eid, start, x_true, y_pred, err
    embryo_rows: list[tuple[str, int, float, float, float]] = []  # eid, n_points, m_anchor, rmse_resid, max_abs_resid
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        starts, t0_hats = d.get("starts", []), d.get("t0_hats", [])
        if not isinstance(starts, list) or not isinstance(t0_hats, list) or len(starts) != len(t0_hats):
            continue  # malformed
        s_list, t0_list = [], []
        for s, t0h in zip(starts, t0_hats):
            try:
                s_int, t0f = int(s), float(t0h)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(t0f):
                s_list.append(s_int)
                t0_list.append(t0f)
        if len(s_list) < 2:
            continue

        s_arr = np.array(s_list, dtype=np.float64)
        x = t0 + dt * s_arr
        y = np.array(t0_list, dtype=np.float64) + dt * s_arr
        for s_int, xx, yy, ee in zip(s_list, x.tolist(), y.tolist(), (y - x).tolist()):
            point_rows.append((fp.stem, int(s_int), float(xx), float(yy), float(ee)))
        m, resid = fit_anchor(x, y, t0)
        embryo_rows.append((fp.stem, len(s_list), float(m), float(rmse(resid)), _max_abs(resid)))

    if not point_rows:
        raise RuntimeError(f"No valid points aggregated from {json_dir}. Check the json format.")

    with open(out_dir / "points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eid", "start", "x_true", "y_pred", "err", "abs_err", "sq_err"])
        for eid, s, x_true, y_pred, err in point_rows:
            w.writerow([eid, s, x_true, y_pred, err, abs(err), err * err])
    with open(out_dir / "embryo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eid", "n_points", "m_anchor", "rmse_resid", "max_abs_resid"])
        w.writerows(list(r) for r in embryo_rows)

    x_true = np.array([r[2] for r in point_rows], dtype=np.float64)
    y_pred = np.array([r[3] for r in point_rows], dtype=np.float64)
    err = y_pred - x_true
    sse = float(np.sum((y_pred - x_true) ** 2))
    sst = float(np.sum((x_true - float(np.mean(x_true))) ** 2))
    m_g, resid_g = fit_anchor(x_true, y_pred, t0)
    m_summary = describe(np.array([r[2] for r in embryo_rows], dtype=np.float64))

    summary = {
        "meta": {"dt_h": dt, "t0_hpf": t0, "json_dir": str(json_dir), "n_json": len(files)},
        "global_metrics_points": {
            "n_points": len(point_rows),
            "n_embryos": len({r[0] for r in point_rows}),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "r2": float(1.0 - sse / sst) if sst > 1e-12 else float("nan"),
        },
        "fit_anchor_T0": {
            "definition": "fit (y - T0) = m * (x - T0)",
            "m": float(m_g),
            "rmse_resid": float(rmse(resid_g)),
            "max_abs_resid": float(_max_abs(resid_g)),
        },
        "per_embryo_summary": {
            "m_mean": m_summary["mean"],
            "m_median": m_summary["median"],
            "m_p05": m_summary["p05"],
            "m_p95": m_summary["p95"],
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[aggregate] {out_dir}: {summary['global_metrics_points']['n_embryos']} embryos")
    return summary


def aggregate_outroot(
    outroot: str | Path,
    datasets: Sequence[str],
    models: Sequence[str],
    dt: float = DT_H,
    t0: float = T0_HPF,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Aggregate ``<outroot>/<dataset>/<model>/json`` for every dataset x model (skip if done unless ``force``)."""
    outroot = Path(outroot)
    if not outroot.is_dir():
        raise SystemExit(f"[ERR] outroot not found: {outroot}")
    for ds in datasets:
        for model in models:
            json_dir = outroot / ds / model / "json"
            if not json_dir.is_dir() or not any(json_dir.glob("*.json")):
                log(f"[WARN] no inference json in {json_dir} (skip)")
                continue
            summary = outroot / ds / model / "summary.json"
            if not force and summary.exists():
                n_json = len(list(json_dir.glob("*.json")))
                if json.loads(summary.read_text(encoding="utf-8")).get("meta", {}).get("n_json") == n_json:
                    log(f"[SKIP] {ds}/{model} already aggregated")
                    continue
                log(f"[aggregate] {ds}/{model}: inference files changed, re-aggregating")
            aggregate_json_dir(json_dir, outroot / ds / model, dt=dt, t0=t0)
