"""Embryo-level statistics (paper Section 3.11).

Embryos, not windows, are the independent units: all confidence intervals
resample embryos with replacement within each condition.

* ``ci_delta_m``       -- Delta m = mean(m_A) - mean(m_B) with a percentile bootstrap CI.
* ``power_curve``      -- detection power (CI excludes 0) versus embryos per condition E,
                          for the observed effect.
* ``power_continuous`` -- embryos needed for 80/90/95 % power (E80/E90/E95) across a
                          continuous range of |Delta m| (planning curves).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .constants import EXT25C, ID28C5, MODELS
from .utils import parse_int_list, read_column, write_csv, write_json


def describe(x: np.ndarray) -> dict[str, float]:
    """n, mean, median, 5th/95th percentiles and SD (ddof=1) of the finite values."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"), p05=float("nan"), p95=float("nan"), std=float("nan"))
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p05=float(np.quantile(x, 0.05)),
        p95=float(np.quantile(x, 0.95)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
    )


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """95 % percentile bootstrap CI of mean(a) - mean(b), resampling each group with replacement."""
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0


# ---------------------------------------------------------------------------
# Delta m with bootstrap CI
# ---------------------------------------------------------------------------
def ci_delta_m(csv_a: str | Path, csv_b: str | Path, out_json: str | Path, *, m_col: str = "m_anchor",
               n_boot: int = 5000, seed: int = 0, label_a: str = "A", label_b: str = "B") -> dict[str, Any]:
    a, b = read_column(csv_a, m_col), read_column(csv_b, m_col)
    lo, hi = bootstrap_ci(a, b, n_boot, np.random.default_rng(seed))
    out = {
        "meta": {
            "csv_a": os.path.abspath(csv_a),
            "csv_b": os.path.abspath(csv_b),
            "label_a": label_a,
            "label_b": label_b,
            "m_col": m_col,
            "B": int(n_boot),
            "seed": int(seed),
            "definition": "Δ = mean(m_a) - mean(m_b)",
        },
        "group_a": describe(a),
        "group_b": describe(b),
        "delta": {"delta_obs": float(a.mean() - b.mean()), "ci95_low": lo, "ci95_high": hi},
    }
    write_json(out, out_json)
    d = out["delta"]
    print(f"[ci] {label_a} - {label_b}: Δm = {d['delta_obs']:.4f}  CI95 = [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]  "
          f"(n_a={out['group_a']['n']}, n_b={out['group_b']['n']})")
    return out


# ---------------------------------------------------------------------------
# Power curve for the observed effect
# ---------------------------------------------------------------------------
def default_E_list(n_min: int) -> list[int]:
    """Even embryo budgets 2..min(30, n_min), plus 3."""
    hi = min(30, n_min)
    xs = set(range(2, hi + 1, 2))
    if hi >= 3:
        xs.add(3)
    return sorted(x for x in xs if 2 <= x <= n_min)


def power_curve(csv_a: str | Path, csv_b: str | Path, out_csv: str | Path, out_png: str | Path | None = None, *,
                m_col: str = "m_anchor", E_list: Sequence[int] | None = None, R: int = 1000, n_boot: int = 1000,
                seed: int = 0, replacement: bool = False, label_a: str = "A", label_b: str = "B") -> list[dict[str, Any]]:
    """Simulated experiments with E embryos per condition; power = P(bootstrap CI excludes 0)."""
    A, B = read_column(csv_a, m_col), read_column(csv_b, m_col)
    nA, nB = A.size, B.size
    E_list = sorted(set(E_list)) if E_list else default_E_list(min(nA, nB))
    if not E_list:
        raise ValueError("E_list is empty.")
    if max(E_list) > min(nA, nB):
        raise ValueError(f"Max E={max(E_list)} exceeds available embryos nmin={min(nA, nB)}")

    rng = np.random.default_rng(seed)
    rows = []
    for E in E_list:
        succ, deltas, los, his = 0, [], [], []
        for _ in range(R):
            if replacement:
                idxA, idxB = rng.integers(0, nA, size=E), rng.integers(0, nB, size=E)
            else:
                idxA, idxB = rng.choice(nA, size=E, replace=False), rng.choice(nB, size=E, replace=False)
            mA, mB = A[idxA], B[idxB]
            lo, hi = bootstrap_ci(mA, mB, n_boot, rng)
            succ += int(_excludes_zero(lo, hi))
            deltas.append(float(mA.mean() - mB.mean()))
            los.append(lo)
            his.append(hi)
        power = succ / float(R)
        rows.append({
            "E": int(E), "power": float(power),
            "delta_mean": float(np.mean(deltas)), "ci_low_mean": float(np.mean(los)), "ci_high_mean": float(np.mean(his)),
            "nA": int(nA), "nB": int(nB), "R": int(R), "B": int(n_boot), "seed": int(seed),
            "replacement": bool(replacement), "label_a": label_a, "label_b": label_b,
        })
    print(f"[power] {label_a} vs {label_b}: " + "  ".join(f"E={r['E']}:{r['power']:.2f}" for r in rows))
    write_csv(out_csv, rows, list(rows[0].keys()))
    if out_png:
        from .figures import plot_power_curve

        plot_power_curve(rows, out_png, title=f"Power curve: {label_a} vs {label_b} ({m_col})")
    return rows


def run_stats(outroot: str | Path, *, models: Sequence[str] = MODELS, a_tag: str = EXT25C, b_tag: str = ID28C5,
              label_a: str = "25C", label_b: str = "28C5", m_col: str = "m_anchor", ci_boot: int = 5000,
              power_R: int = 500, power_boot: int = 500, seed: int = 0) -> None:
    """CI_<model>_<m_col>.json and power_<model>_<m_col>.{csv,png} for every model (paper Table 6)."""
    outroot = Path(outroot)
    for model in models:
        a, b = outroot / a_tag / model / "embryo.csv", outroot / b_tag / model / "embryo.csv"
        for p in (a, b):
            if not p.exists():
                raise SystemExit(f"[ERR] missing {p}; run `etf aggregate` first")
        print(f"== CI and power: {model} ==")
        ci_delta_m(a, b, outroot / f"CI_{model}_{m_col}.json", m_col=m_col, n_boot=ci_boot, seed=seed, label_a=label_a, label_b=label_b)
        power_curve(a, b, outroot / f"power_{model}_{m_col}.csv", outroot / f"power_{model}_{m_col}.png", m_col=m_col,
                    R=power_R, n_boot=power_boot, seed=seed, label_a=label_a, label_b=label_b)


# ---------------------------------------------------------------------------
# Continuous effect-size planning curves
# ---------------------------------------------------------------------------
def delta_grid(delta_max: float, delta_step: float) -> list[float]:
    """0, step, 2*step, ..., delta_max (rounded to 10 decimals)."""
    n = int(round(delta_max / delta_step))
    return [round(i * delta_step, 10) for i in range(n + 1)]


def simulate_power_surface(m_a_obs: np.ndarray, m_b_obs: np.ndarray, *, e_list: Sequence[int], deltas: Sequence[float],
                           r_outer: int, b_boot: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Power(E, |Delta m|) by resampling pooled centred residuals around synthetic group means.

    Condition A is shifted below B by delta (A slower), embryos are drawn from
    mean + residual, and detection means the bootstrap CI excludes 0.
    """
    mean_b = float(np.mean(m_b_obs))
    res = np.concatenate([m_a_obs - np.mean(m_a_obs), m_b_obs - np.mean(m_b_obs)]).astype(np.float64)
    rows = []
    for delta_abs in deltas:
        delta_signed = -float(delta_abs)
        mean_a = mean_b + delta_signed
        for e in e_list:
            success = 0
            for _ in range(r_outer):
                m_a = mean_a + res[rng.integers(0, res.size, size=e)]
                m_b = mean_b + res[rng.integers(0, res.size, size=e)]
                success += int(_excludes_zero(*bootstrap_ci(m_a, m_b, b_boot, rng)))
            rows.append({"delta_abs": float(delta_abs), "delta_signed": delta_signed, "E_per_group": int(e),
                         "power": success / float(r_outer)})
    return rows


def monotone_power_over_e(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Running maximum of power over increasing E (keeps the raw value in ``power_raw``)."""
    by_delta: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_delta.setdefault(float(r["delta_abs"]), []).append(dict(r))
    out = []
    for d in sorted(by_delta):
        best = -1.0
        for r in sorted(by_delta[d], key=lambda x: int(x["E_per_group"])):
            r["power_raw"] = float(r["power"])
            best = max(best, r["power_raw"])
            r["power"] = best
            out.append(r)
    return out


def thresholds(rows: Sequence[dict[str, Any]], e_list: Sequence[int], targets: Sequence[float]) -> list[dict[str, Any]]:
    """Smallest E reaching each target power, per |Delta m| (``">Emax"`` if never reached)."""
    by_delta: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_delta.setdefault(float(r["delta_abs"]), []).append(r)
    out = []
    for d in sorted(by_delta):
        rr = sorted(by_delta[d], key=lambda x: int(x["E_per_group"]))
        row: dict[str, Any] = {"delta_abs": d, "delta_signed": -d}
        for t in targets:
            thr = next((int(r["E_per_group"]) for r in rr if float(r["power"]) >= t), None)
            row[f"E{int(round(t * 100))}"] = thr if thr is not None else f">{max(e_list)}"
        out.append(row)
    return out


def monotone_thresholds_over_delta(rows: Sequence[dict[str, Any]], targets: Sequence[float], e_max: int) -> list[dict[str, Any]]:
    """Running minimum of the required E over increasing |Delta m|."""
    rr = sorted((dict(r) for r in rows), key=lambda x: float(x["delta_abs"]))
    for t in targets:
        key = f"E{int(round(t * 100))}"
        best = e_max + 1
        for r in rr:
            v = r[key]
            best = min(best, e_max + 1 if str(v).startswith(">") else int(v))
            r[key] = f">{e_max}" if best > e_max else best
    return rr


POWER_SURFACE_FIELDS = ["model", "delta_abs", "delta_signed", "E_per_group", "power", "power_raw", "R", "B_boot", "seed",
                        "monotone_power_e", "monotone_threshold_delta", "method"]
THRESHOLD_FIELDS = ["model", "delta_abs", "delta_signed", "E80", "E90", "E95", "monotone_power_e", "monotone_threshold_delta"]
POWER_TARGETS = (0.80, 0.90, 0.95)


def power_continuous(outroot: str | Path, out_dir: str | Path | None = None, *, models: Sequence[str] = MODELS,
                     a_tag: str = EXT25C, b_tag: str = ID28C5, m_col: str = "m_anchor",
                     E_list: str | Sequence[int] = "2,3,4,6,8,10,12,14,16,18,20,22", delta_max: float = 0.10,
                     delta_step: float = 0.002, R: int = 1200, B_boot: int = 800, seed: int = 20260310,
                     y_min: float = 0.0, y_max: float = 23.0, y_tick_step: float = 1.0,
                     monotone_power: bool = True, monotone_threshold: bool = True,
                     log: Callable[[str], None] = print) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Continuous planning curves (paper Fig. 4B); writes two CSVs and two SVG figures."""
    if y_tick_step <= 0 or y_max <= y_min:
        raise ValueError("need y_tick_step > 0 and y_max > y_min")
    outroot = Path(outroot)
    out_dir = Path(out_dir) if out_dir else outroot / "continuous_power"
    e_list = parse_int_list(E_list) if isinstance(E_list, str) else sorted(set(E_list))
    deltas = delta_grid(delta_max, delta_step)
    mp, mt = int(bool(monotone_power)), int(bool(monotone_threshold))
    rng = np.random.default_rng(seed)

    surface, thr_rows = [], []
    for model in models:
        csv_a, csv_b = outroot / a_tag / model / "embryo.csv", outroot / b_tag / model / "embryo.csv"
        if not csv_a.exists() or not csv_b.exists():
            raise FileNotFoundError(f"Missing embryo.csv for model={model}: {csv_a} | {csv_b}")
        a_obs, b_obs = read_column(csv_a, m_col), read_column(csv_b, m_col)
        log(f"[power-continuous] {model}: nA={a_obs.size} nB={b_obs.size}")
        rows = simulate_power_surface(a_obs, b_obs, e_list=e_list, deltas=deltas, r_outer=R, b_boot=B_boot, rng=rng)
        if mp:
            rows = monotone_power_over_e(rows)
        for r in rows:
            r.setdefault("power_raw", r["power"])
            r.update(model=model, R=R, B_boot=B_boot, seed=seed, monotone_power_e=mp, monotone_threshold_delta=mt,
                     method=f"pooled_residual_shift+bootstrap_CI_excludes_zero+monotone_powerE={mp}")
        surface.extend(rows)
        thr = thresholds(rows, e_list, POWER_TARGETS)
        if mt:
            thr = monotone_thresholds_over_delta(thr, POWER_TARGETS, max(e_list))
        for r in thr:
            r.update(model=model, monotone_power_e=mp, monotone_threshold_delta=mt)
        thr_rows.extend(thr)

    write_csv(out_dir / "continuous_power_by_model.csv", surface, POWER_SURFACE_FIELDS)
    write_csv(out_dir / "continuous_thresholds_by_model.csv", thr_rows, THRESHOLD_FIELDS)

    from .figures import plot_threshold_curves

    footnote = (f"Embryo-level simulation; R={R}, B_boot={B_boot}; monotone_powerE={mp}, "
                f"monotone_thresholdDelta={mt}; '>{max(e_list)}' shown as gaps")
    axes = dict(delta_max=delta_max, y_min=y_min, y_max=y_max, y_tick_step=y_tick_step, footnote=footnote)
    plot_threshold_curves(thr_rows, out_dir / "continuous_E80_by_model.svg", models=models, keys=("E80",),
                          title="Continuous Effect-Size Planning Curve (E80)", y_label="Required embryos per group (E80)", **axes)
    if "full" in models:
        plot_threshold_curves(thr_rows, out_dir / "continuous_full_E80_E90_E95.svg", models=("full",), keys=("E80", "E90", "E95"),
                              title="Continuous Effect-Size Planning Curve (full model)", y_label="Required embryos per group", **axes)
    log(f"[power-continuous] wrote {out_dir}")
    return surface, thr_rows
