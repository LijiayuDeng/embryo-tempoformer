"""Sensitivity analyses reported in the supplement (Tables S6-S8 and the clip-length ladder).

* ``stage_error_bins``     -- window-level error by Kimmel period (descriptive)
* ``anchor_sensitivity``   -- re-aggregation under alternative anchors T0
* ``stage_tempo``          -- piecewise tempo slopes per Kimmel period with embryo-bootstrap CIs
* ``cliplen_sensitivity``  -- fixed-checkpoint inference with shorter clips (L = 4/12/24)
* ``context_fit``          -- 0 h / 1 h / 3 h / 6 h context ladder and descriptive linear fits
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .constants import DT_H, EXT25C, ID28C5, MAIN_DATASETS, MODELS, STRIDE, T0_HPF
from .tempo import aggregate_json_dir
from .utils import fmt, markdown_table, parse_list, read_csv_rows, read_json, write_csv

# Kimmel et al. (1995) broad periods (hpf). The imaged window (4.5-52 hpf) ends
# before hatching, so the stage-tempo analysis uses the first four.
KIMMEL_PERIODS: list[tuple[str, float, float]] = [
    ("blastula", 2.25, 5.25),
    ("gastrula", 5.25, 10.33),
    ("segmentation", 10.33, 24.0),
    ("pharyngula", 24.0, 48.0),
    ("hatching", 48.0, 72.0),
]


def _read_points(path: Path) -> list[dict[str, Any]]:
    return [{"eid": r["eid"], "x_true": float(r["x_true"]), "y_pred": float(r["y_pred"]), "err": float(r["err"]),
             "abs_err": float(r["abs_err"])} for r in read_csv_rows(path)]


# ---------------------------------------------------------------------------
# Stage-stratified error
# ---------------------------------------------------------------------------
STAGE_ERROR_FIELDS = ["dataset", "model", "scheme", "stage_name", "stage_label", "canonical_lo_hpf", "canonical_hi_hpf",
                      "observed_lo_hpf", "observed_hi_hpf", "n_points", "mae", "rmse", "bias"]


def stage_error_bins(outroot: str | Path, out_csv: str | Path | None = None, *, datasets: Sequence[str] = MAIN_DATASETS,
                     models: Sequence[str] = ("cnn_single", "full"), scheme: str = "kimmel_start", bins: str = "") -> list[dict[str, Any]]:
    """MAE / RMSE / bias of windows binned by their nominal start time (descriptive only)."""
    outroot = Path(outroot)
    if scheme == "fixed":
        edges = [float(x) for x in parse_list(bins)]
        if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError("--bins needs at least two strictly increasing edges when scheme=fixed")
        stages = [(f"fixed_{i + 1}", f"{edges[i]:.1f}-{edges[i + 1]:.1f}", edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    elif scheme == "kimmel_start":
        stages = [(name, f"{name} ({lo:.2f}-{hi:.2f} hpf)", lo, hi) for name, lo, hi in KIMMEL_PERIODS]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    rows = []
    for ds in datasets:
        for model in models:
            groups: list[list[dict[str, Any]]] = [[] for _ in stages]
            for p in _read_points(outroot / ds / model / "points.csv"):
                for i, (_, _, lo, hi) in enumerate(stages):
                    if lo <= p["x_true"] < hi:
                        groups[i].append(p)
                        break
            for (name, label, lo, hi), pts in zip(stages, groups):
                if not pts:
                    continue
                n = len(pts)
                rows.append({
                    "dataset": ds, "model": model, "scheme": scheme, "stage_name": name, "stage_label": label,
                    "canonical_lo_hpf": lo, "canonical_hi_hpf": hi,
                    "observed_lo_hpf": min(p["x_true"] for p in pts), "observed_hi_hpf": max(p["x_true"] for p in pts),
                    "n_points": n,
                    "mae": sum(p["abs_err"] for p in pts) / n,
                    "rmse": (sum(p["err"] * p["err"] for p in pts) / n) ** 0.5,
                    "bias": sum(p["err"] for p in pts) / n,
                })
    out_csv = Path(out_csv) if out_csv else outroot / "stage_error" / "stage_error_by_bin.csv"
    write_csv(out_csv, rows, STAGE_ERROR_FIELDS)
    print(f"[stage-error] {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# Anchor (T0) sensitivity
# ---------------------------------------------------------------------------
def anchor_sensitivity(outroot: str | Path, out_dir: str | Path | None = None, *, datasets: Sequence[str] = MAIN_DATASETS,
                       models: Sequence[str] = MODELS, t0_list: str = "4.0,4.5,5.0", dt: float = DT_H) -> list[dict[str, Any]]:
    """Re-aggregate the inference JSON with alternative anchors T0 (no retraining)."""
    outroot = Path(outroot)
    out_dir = Path(out_dir) if out_dir else outroot / "anchor_sensitivity"
    rows = []
    for tok in parse_list(t0_list):
        t0 = float(tok)
        for ds in datasets:
            for model in models:
                json_dir = outroot / ds / model / "json"
                if not json_dir.is_dir():
                    raise SystemExit(f"[ERR] missing json dir: {json_dir}")
                s = aggregate_json_dir(json_dir, out_dir / f"t0_{tok.replace('.', 'p')}" / ds / model, dt=dt, t0=t0, verbose=False)
                gm, fa = s["global_metrics_points"], s["fit_anchor_T0"]
                rows.append({"t0_hpf": t0, "dataset": ds, "model": model, "mae": gm["mae"], "rmse": gm["rmse"], "r2": gm["r2"],
                             "m_anchor_global": fa["m"], "rmse_resid": fa["rmse_resid"], "max_abs_resid": fa["max_abs_resid"]})
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["model"]), float(r["t0_hpf"])))
    write_csv(out_dir / "anchor_sensitivity_summary.csv", rows, list(rows[0].keys()))
    print(f"[anchor] {out_dir / 'anchor_sensitivity_summary.csv'}")
    return rows


# ---------------------------------------------------------------------------
# Stage-dependent tempo
# ---------------------------------------------------------------------------
def _ols(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    """Slope, intercept and r^2 of y_pred on x_true (nan if fewer than two distinct x)."""
    x = np.asarray([float(r["x_true"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["y_pred"]) for r in rows], dtype=np.float64)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    corr = np.corrcoef(x, y)[0, 1]
    return float(slope), float(intercept), float(corr * corr)


def _bootstrap_slopes(rows: Sequence[Mapping[str, Any]], seed: int, n_boot: int) -> np.ndarray:
    """Embryo-bootstrap distribution of the OLS slope (resample embryos with replacement)."""
    eids = sorted({str(r["eid"]) for r in rows})
    if len(eids) < 2:
        return np.asarray([], dtype=np.float64)
    by_eid: dict[str, list[Mapping[str, Any]]] = {eid: [] for eid in eids}
    for r in rows:
        by_eid[str(r["eid"])].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [row for eid in rng.choice(eids, size=len(eids), replace=True) for row in by_eid[eid]]
        slope = _ols(sample)[0]
        if math.isfinite(slope):
            vals.append(slope)
    return np.asarray(vals, dtype=np.float64)


def _quantile_ci(v: np.ndarray) -> tuple[float, float]:
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v.size else (float("nan"), float("nan"))


PIECEWISE_FIELDS = ["model", "dataset", "stage_name", "canonical_lo_hpf", "canonical_hi_hpf", "n_points", "piecewise_slope",
                    "piecewise_slope_ci_lo", "piecewise_slope_ci_hi", "n_boot_eff", "piecewise_intercept", "piecewise_corr2"]
DELTA_TAG = "delta_25C_minus_28C5"


def stage_tempo(outroot: str | Path, out_dir: str | Path | None = None, *, datasets: Sequence[str] = MAIN_DATASETS,
                models: Sequence[str] = MODELS, n_boot: int = 3000, seed: int = 42) -> None:
    """Kimmel-period piecewise slopes with embryo-bootstrap CIs, their 25 C - 28.5 C differences,
    between-stage contrasts, and local interval slopes of the stage-averaged trajectories."""
    outroot = Path(outroot)
    out_dir = Path(out_dir) if out_dir else outroot / "stage_tempo"
    periods = KIMMEL_PERIODS[:4]
    piecewise, local, contrasts = [], [], []

    for model_idx, model in enumerate(models):
        slopes: dict[str, dict[str, float]] = {}
        boots: dict[str, dict[str, np.ndarray]] = {}
        for ds_idx, ds in enumerate(datasets):
            rows = _read_points(outroot / ds / model / "points.csv")
            slopes[ds], boots[ds] = {}, {}
            for stage_idx, (stage, lo, hi) in enumerate(periods):
                sub = [r for r in rows if lo <= r["x_true"] < hi]
                slope, intercept, corr2 = _ols(sub)
                vals = _bootstrap_slopes(sub, seed=seed + 10000 * model_idx + 1000 * ds_idx + 100 * stage_idx, n_boot=n_boot)
                ci_lo, ci_hi = _quantile_ci(vals)
                slopes[ds][stage], boots[ds][stage] = slope, vals
                piecewise.append({"model": model, "dataset": ds, "stage_name": stage, "canonical_lo_hpf": lo, "canonical_hi_hpf": hi,
                                  "n_points": len(sub), "piecewise_slope": slope, "piecewise_slope_ci_lo": ci_lo,
                                  "piecewise_slope_ci_hi": ci_hi, "n_boot_eff": int(vals.size), "piecewise_intercept": intercept,
                                  "piecewise_corr2": corr2})

            # Local finite-difference slopes of the mean trajectory over nominal start times.
            xs = sorted({r["x_true"] for r in rows})
            means = [(x, float(np.mean([r["y_pred"] for r in rows if r["x_true"] == x]))) for x in xs]
            for (x0, y0), (x1, y1) in zip(means[:-1], means[1:]):
                local.append({"model": model, "dataset": ds, "x_lo_hpf": x0, "x_hi_hpf": x1, "x_mid_hpf": 0.5 * (x0 + x1),
                              "local_slope": (y1 - y0) / (x1 - x0)})

        if ID28C5 not in slopes or EXT25C not in slopes:
            continue
        # 25 C - 28.5 C slope differences (independent bootstrap draws paired by index).
        delta_boot: dict[str, np.ndarray] = {}
        for stage, lo, hi in periods:
            m28, m25 = slopes[ID28C5].get(stage, float("nan")), slopes[EXT25C].get(stage, float("nan"))
            v28, v25 = boots[ID28C5].get(stage, np.asarray([])), boots[EXT25C].get(stage, np.asarray([]))
            if v28.size and v25.size:
                n = min(v28.size, v25.size)
                delta_boot[stage] = v25[:n] - v28[:n]
                (ci_lo, ci_hi), n_eff = _quantile_ci(delta_boot[stage]), int(n)
            else:
                delta_boot[stage] = np.asarray([], dtype=np.float64)
                (ci_lo, ci_hi), n_eff = (float("nan"), float("nan")), ""
            piecewise.append({"model": model, "dataset": DELTA_TAG, "stage_name": stage, "canonical_lo_hpf": lo, "canonical_hi_hpf": hi,
                              "n_points": "", "piecewise_slope": m25 - m28 if not (np.isnan(m25) or np.isnan(m28)) else float("nan"),
                              "piecewise_slope_ci_lo": ci_lo, "piecewise_slope_ci_hi": ci_hi, "n_boot_eff": n_eff,
                              "piecewise_intercept": float("nan"), "piecewise_corr2": float("nan")})
        stages = ["gastrula", "segmentation", "pharyngula"]
        for i, a in enumerate(stages):
            for b in stages[i + 1:]:
                va, vb = delta_boot.get(a, np.asarray([])), delta_boot.get(b, np.asarray([]))
                if va.size and vb.size:
                    n = min(va.size, vb.size)
                    diff = va[:n] - vb[:n]
                    contrasts.append({"model": model, "contrast": f"{a}_minus_{b}", "estimate": float(np.mean(diff)),
                                      "ci_lo": float(np.quantile(diff, 0.025)), "ci_hi": float(np.quantile(diff, 0.975)),
                                      "n_boot_eff": int(n)})

    write_csv(out_dir / "piecewise_stage_slopes.csv", piecewise, PIECEWISE_FIELDS)
    write_csv(out_dir / "local_slope_by_interval.csv", local, ["model", "dataset", "x_lo_hpf", "x_hi_hpf", "x_mid_hpf", "local_slope"])
    write_csv(out_dir / "stage_delta_contrasts.csv", contrasts, ["model", "contrast", "estimate", "ci_lo", "ci_hi", "n_boot_eff"])
    print(f"[stage-tempo] {out_dir}")

    # Compact ETF-full table and figure (only when the 28.5 C vs 25 C comparison is available).
    lookup = {(r["dataset"], r["stage_name"]): r for r in piecewise if r["model"] == "full"}
    keys = [(ds, s) for ds in (ID28C5, EXT25C, DELTA_TAG) for s, _, _ in periods]
    if "full" not in models or not all(k in lookup for k in keys):
        print("[stage-tempo] full-model summary skipped (needs model 'full' and both test sets)")
        return

    def cell(r: Mapping[str, Any]) -> str:
        return f"{fmt(r['piecewise_slope'])} [{fmt(r['piecewise_slope_ci_lo'])}, {fmt(r['piecewise_slope_ci_hi'])}]"

    headers = ["Kimmel period", "Canonical window (hpf)", "28.5C piecewise slope [95% CI]", "25C piecewise slope [95% CI]",
               "delta slope (25C-28.5C) [95% CI]"]
    table = [dict(zip(headers, [s.capitalize(), f"{lo:.2f}-{hi:.2f}", cell(lookup[(ID28C5, s)]), cell(lookup[(EXT25C, s)]),
                                cell(lookup[(DELTA_TAG, s)])])) for s, lo, hi in periods]
    (out_dir / "stage_tempo_full_summary.md").write_text(markdown_table(table, headers), encoding="utf-8")

    from .figures import plot_local_slopes

    series = []
    for ds, color, label in [(ID28C5, "#1f4e79", "28.5C"), (EXT25C, "#b24a2a", "25C")]:
        sel = [r for r in local if r["model"] == "full" and r["dataset"] == ds]
        series.append((label, color, [float(r["x_mid_hpf"]) for r in sel], [float(r["local_slope"]) for r in sel]))
    plot_local_slopes(series, out_dir / "full_local_slope_by_interval.svg", "ETF-full local tempo slope by developmental window")


# ---------------------------------------------------------------------------
# Clip-length sensitivity and context ladder
# ---------------------------------------------------------------------------
CLIPLEN_FIELDS = ["clip_len", "dataset", "model", "n_embryos", "n_points", "mae", "rmse", "r2", "m_anchor_global",
                  "rmse_resid", "max_abs_resid"]


def cliplen_sensitivity(outroot: str | Path, *, datasets: Mapping[str, tuple[Path, Path]], checkpoints: Mapping[str, Path],
                        clip_lens: Sequence[int] = (4, 12, 24), stride: int = STRIDE, dt: float = DT_H, t0: float = T0_HPF,
                        device: str = "auto", amp: bool = True, use_ema: bool = True, max_eids: int = 0, force: bool = False,
                        log: Callable[[str], None] = print) -> Path:
    """Run the released checkpoints with shorter clips at the same 15-min sampling (no retraining).

    Writes ``<outroot>/L<NN>/<dataset>/<model>/...`` and ``<outroot>/cliplen_summary.csv``.
    """
    from .data import split_ids
    from .inference import infer_all

    outroot = Path(outroot)
    for L in clip_lens:
        sub = outroot / f"L{L:02d}"
        infer_all(outroot=sub, datasets=datasets, checkpoints=checkpoints, clip_len=L, stride=stride, device=device, amp=amp,
                  use_ema=use_ema, force=force, max_eids=max_eids, log=log)
        for ds, (proc_dir, split) in datasets.items():
            expected = sum((Path(proc_dir) / f"{eid}.npy").exists() for eid in split_ids(split, "test", max_eids))
            for model in checkpoints:
                json_dir = sub / ds / model / "json"
                n_json = len(list(json_dir.glob("*.json")))
                if n_json == 0:
                    continue
                if not force and (sub / ds / model / "summary.json").exists() and n_json == expected:
                    continue
                aggregate_json_dir(json_dir, sub / ds / model, dt=dt, t0=t0, verbose=False)

    rows = []
    for lp in sorted(outroot.glob("L*/")):
        try:
            clip_len = int(lp.name[1:])
        except ValueError:
            continue
        for ds_dir in sorted(p for p in lp.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                summ = model_dir / "summary.json"
                if not summ.exists():
                    continue
                s = read_json(summ)
                gm, fa = s.get("global_metrics_points", {}), s.get("fit_anchor_T0", {})
                rows.append({"clip_len": clip_len, "dataset": ds_dir.name, "model": model_dir.name,
                             "n_embryos": gm.get("n_embryos"), "n_points": gm.get("n_points"), "mae": gm.get("mae"),
                             "rmse": gm.get("rmse"), "r2": gm.get("r2"), "m_anchor_global": fa.get("m"),
                             "rmse_resid": fa.get("rmse_resid"), "max_abs_resid": fa.get("max_abs_resid")})
    write_csv(outroot / "cliplen_summary.csv", rows, CLIPLEN_FIELDS)
    log(f"[cliplen] {outroot / 'cliplen_summary.csv'}")
    return outroot / "cliplen_summary.csv"


CONTEXT_METRICS = ["mae", "rmse", "r2", "rmse_resid", "max_abs_resid", "m_anchor_global"]


def context_fit(main_outroot: str | Path, cliplen_csv: str | Path, out_dir: str | Path, *,
                clip_lens: Sequence[int] = (4, 12, 24), dt: float = DT_H) -> None:
    """Context ladder: cnn_single (0 h) plus ETF-full at each clip length, and per-metric linear fits
    across the ETF-full contexts only (cnn_single is a different model and is not part of the fit)."""
    main_outroot, out_dir = Path(main_outroot), Path(out_dir)
    clip_rows = read_csv_rows(cliplen_csv)
    ladder, fits = [], []
    for ds in sorted({r["dataset"] for r in clip_rows}):
        s = read_json(main_outroot / ds / "cnn_single" / "summary.json")
        gm, fa = s["global_metrics_points"], s["fit_anchor_T0"]
        ladder.append({"dataset": ds, "context_h": 0.0, "label": "cnn_single_0h", "model_family": "cnn_single", "clip_len": 1,
                       "mae": gm["mae"], "rmse": gm["rmse"], "r2": gm["r2"], "rmse_resid": fa["rmse_resid"],
                       "max_abs_resid": fa["max_abs_resid"], "m_anchor_global": fa["m"],
                       "fit_note": "separate image-only baseline; not included in full-context linear fit"})
        full = sorted((r for r in clip_rows if r["dataset"] == ds and r["model"] == "full" and int(r["clip_len"]) in clip_lens),
                      key=lambda r: int(r["clip_len"]))
        if len(full) != len(clip_lens):
            raise SystemExit(f"[ERR] {cliplen_csv} lacks ETF-full results for {ds} at clip lengths {list(clip_lens)}")
        x = np.asarray([float(int(r["clip_len"])) * dt for r in full], dtype=np.float64)
        for r, ctx in zip(full, x):
            L = int(r["clip_len"])
            ladder.append({"dataset": ds, "context_h": float(ctx), "label": f"full_L{L}", "model_family": "ETF-full", "clip_len": L,
                           "fit_note": "included in full-context linear fit", **{m: float(r[m]) for m in CONTEXT_METRICS}})
        for m in CONTEXT_METRICS:
            y = np.asarray([float(r[m]) for r in full], dtype=np.float64)
            coef = np.polyfit(x, y, deg=1)
            sse = float(np.sum((y - np.polyval(coef, x)) ** 2))
            sst = float(np.sum((y - float(np.mean(y))) ** 2))
            fits.append({"dataset": ds, "model_family": "ETF-full", "metric": m, "contexts_h": ",".join(f"{c:g}" for c in x),
                         "n_points": int(len(x)), "slope_per_hour": float(coef[0]), "intercept": float(coef[1]),
                         "fit_r2": float(1.0 - sse / sst) if sst > 1e-12 else float("nan")})
    write_csv(out_dir / "context_ladder.csv", ladder, ["dataset", "context_h", "label", "model_family", "clip_len", *CONTEXT_METRICS, "fit_note"])
    write_csv(out_dir / "full_context_fit.csv", fits, ["dataset", "model_family", "metric", "contexts_h", "n_points",
                                                       "slope_per_hour", "intercept", "fit_r2"])
    print(f"[context-fit] {out_dir}")
