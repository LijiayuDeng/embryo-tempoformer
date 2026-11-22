"""Paper figures and tables.

``make_paper_figures`` writes the data panels of Figs. 2-4 to ``<outroot>/figures``
and ``paper_tables`` writes Tables 2, 5 and 6 to ``<outroot>/paper_tables.md``.
SVG/PDF files are written without timestamps so reruns are byte-identical.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .constants import EXT25C, ID28C5, MODEL_LABELS, MODELS  # noqa: E402
from .utils import fmt, markdown_table, read_column, read_csv_rows, read_json  # noqa: E402

PAPER_STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.facecolor": "white",
    "axes.edgecolor": "#111111",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 2.0,
    "axes.grid": False,
    "grid.color": "#DADCE0",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.6,
}

# Emphasise the full model and the consistency ablation; de-emphasise baselines.
COLORS = {"cnn_single": "#9AA0A6", "meanpool": "#C0C6CF", "nocons": "#E69F00", "full": "#D33F49"}
# Colour-blind safe palette for the planning curves.
PLANNING_COLORS = {"cnn_single": "#009E73", "meanpool": "#0072B2", "nocons": "#E69F00", "full": "#D55E00"}
TARGET_COLORS = {"E80": "#D55E00", "E90": "#0072B2", "E95": "#009E73"}
LABELS = [MODEL_LABELS[m] for m in MODELS]


def _savefig(fig, path: Path, **kw: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"svg": {"Date": None}, "pdf": {"CreationDate": None}}.get(path.suffix.lstrip("."))
    with matplotlib.rc_context({"svg.hashsalt": "etf"}):  # stable element ids in SVG output
        fig.savefig(str(path), metadata=meta, **kw)


def _save_png_pdf(fig, out_base: Path, dpi: int) -> None:
    _savefig(fig, out_base.with_suffix(".png"), dpi=dpi)
    _savefig(fig, out_base.with_suffix(".pdf"))
    print(f"[figures] {out_base.with_suffix('.png')} (+ .pdf)")


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))


# ---------------------------------------------------------------------------
# Figures 2-4 (data panels)
# ---------------------------------------------------------------------------
def _violin_iqr(ax, data: list[np.ndarray], x: np.ndarray, colors: list[str], ylabel: str, title: str) -> None:
    """Violin + IQR bar + median dot + light jittered points."""
    parts = ax.violinplot(data, positions=x, widths=0.78, showmeans=False, showextrema=False, showmedians=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.18)
        body.set_linewidth(1.2)
    for xi, d, c in zip(x, data, colors):
        q1, med, q3 = (float(v) for v in np.quantile(d, np.array([0.25, 0.5, 0.75], dtype=np.float64)))
        ax.plot([xi, xi], [q1, q3], color=c, linewidth=3.0, solid_capstyle="round", zorder=3)
        ax.scatter([xi], [med], s=34, color=c, edgecolor="white", linewidth=0.9, zorder=4)
        jit = np.random.default_rng(42).normal(loc=0.0, scale=0.06, size=len(d)).astype(np.float64)
        ax.scatter(np.full(d.shape, float(xi), dtype=np.float64) + jit, d, s=8, color=c, alpha=0.10, linewidths=0, zorder=2)
    ax.yaxis.grid(True, alpha=0.22)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def fig2_id_ablation(outroot: Path, outdir: Path, dpi: int) -> None:
    """Fig. 2: window-level MAE/RMSE (A) and anchored-fit residual RMSE (B) on the 28.5 C test set."""
    mae, rmse_, resid = [], [], []
    for m in MODELS:
        d = read_json(outroot / ID28C5 / m / "summary.json")
        mae.append(float(d["global_metrics_points"]["mae"]))
        rmse_.append(float(d["global_metrics_points"]["rmse"]))
        resid.append(float(d["fit_anchor_T0"]["rmse_resid"]))
        n_emb = int(d["global_metrics_points"]["n_embryos"])
        n_pts = int(d["global_metrics_points"]["n_points"])

    x = np.arange(len(MODELS), dtype=np.float64)
    fig = plt.figure(figsize=(6.8, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    dx = 0.13
    for i, m in enumerate(MODELS):
        c = COLORS[m]
        ax.plot([x[i] - dx, x[i] + dx], [mae[i], rmse_[i]], color=c, alpha=0.25, zorder=1)
        ax.scatter(x[i] - dx, mae[i], s=70, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        ax.scatter(x[i] + dx, rmse_[i], s=70, facecolor="white", edgecolor=c, linewidth=2.0, zorder=3)
    ax.yaxis.grid(True, alpha=0.22)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Error (hours)")
    ax.set_title(f"ID 28.5°C test (n={n_emb} embryos, {n_pts} points; stride=8)")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=7, label="MAE (filled)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#111111", markersize=7, label="RMSE (hollow)"),
    ], frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    order = [int(i) for i in np.argsort(np.array(resid, dtype=np.float64))]
    y = np.arange(len(order), dtype=np.float64)
    for yy, idx in zip(y, order):
        c = COLORS[MODELS[idx]]
        ax2.plot([0, resid[idx]], [yy, yy], color=c, alpha=0.65, linewidth=3.0, solid_capstyle="round")
        ax2.scatter([resid[idx]], [yy], s=72, color=c, edgecolor="white", linewidth=0.9, zorder=3)
    ax2.xaxis.grid(True, alpha=0.22)
    ax2.set_yticks(y, [LABELS[i] for i in order])
    ax2.set_xlabel("Residual RMSE (hours)")
    ax2.set_title("Anchored-fit scatter (y−4.5 = m(x−4.5))")

    fig.suptitle("Figure 2. In-distribution ablation on 28.5°C test", y=1.02, fontsize=13)
    _save_png_pdf(fig, outdir / "fig2_id_ablation", dpi)
    plt.close(fig)


def _fig3_violins(outroot: Path, out_base: Path, dpi: int, col: str, ylabel: str, title: str, ref_line: float | None) -> None:
    data = [read_column(outroot / EXT25C / m / "embryo.csv", col) for m in MODELS]
    fig = plt.figure(figsize=(6.6, 6.2))
    ax = fig.add_subplot(1, 1, 1)
    _violin_iqr(ax, data, np.arange(1, len(MODELS) + 1, dtype=np.float64), [COLORS[m] for m in MODELS], ylabel,
                title.format(n=int(data[-1].size)))
    if ref_line is not None:
        ax.axhline(ref_line, color="#111111", linestyle="--", linewidth=1.1, alpha=0.55)
    _save_png_pdf(fig, out_base, dpi)
    plt.close(fig)


def fig4a_delta_m_forest(outroot: Path, outdir: Path, dpi: int) -> None:
    """Fig. 4A: Delta m with embryo-bootstrap 95 % CI per model."""
    cis = {m: read_json(outroot / f"CI_{m}_m_anchor.json")["delta"] for m in MODELS}
    order = list(reversed(MODELS))  # full, nocons, meanpool, cnn_single
    dm = np.array([cis[m]["delta_obs"] for m in order], dtype=np.float64)
    lo = np.array([cis[m]["ci95_low"] for m in order], dtype=np.float64)
    hi = np.array([cis[m]["ci95_high"] for m in order], dtype=np.float64)
    y = np.arange(len(order), dtype=np.float64)

    fig = plt.figure(figsize=(6.6, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    for yy, mval, ll, hh, m in zip(y, dm, lo, hi, order):
        c = COLORS[m]
        ax.plot([ll, hh], [yy, yy], color=c, linewidth=3.2, solid_capstyle="round", alpha=0.85)
        ax.scatter([mval], [yy], s=80, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(hh, yy, f"  {mval:.3f}", va="center", ha="left", fontsize=9, color="#111111")
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.xaxis.grid(True, alpha=0.22)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in order])
    ax.set_xlabel("Δm = mean(m_25C) − mean(m_28.5C)   (95% CI)")
    ax.set_title("Temperature effect size (embryo-bootstrap CI)")
    span = float(np.max(hi)) - float(np.min(lo)) or 1.0
    ax.set_xlim(float(np.min(lo)) - 0.18 * span, float(np.max(hi)) + 0.18 * span)
    _save_png_pdf(fig, outdir / "fig4a_delta_m_forest", dpi)
    plt.close(fig)


def make_paper_figures(outroot: str | Path, outdir: str | Path | None = None, dpi: int = 350) -> Path:
    """Data panels of Figs. 2, 3A/B and 4A/B under ``<outroot>/figures`` (PNG + PDF)."""
    outroot = Path(outroot)
    outdir = Path(outdir) if outdir else outroot / "figures"
    _require(*[outroot / ID28C5 / m / "summary.json" for m in MODELS],
             *[outroot / EXT25C / m / "embryo.csv" for m in MODELS],
             *[outroot / f"CI_{m}_m_anchor.json" for m in MODELS])
    with matplotlib.rc_context(PAPER_STYLE):
        fig2_id_ablation(outroot, outdir, dpi)
        _fig3_violins(outroot, outdir / "fig3a_ext_m_anchor", dpi, "m_anchor", "m_anchor (tempo slope)",
                      "External 25°C test: tempo slope distribution (n={n} embryos)", ref_line=1.0)
        _fig3_violins(outroot, outdir / "fig3b_ext_rmse_resid", dpi, "rmse_resid", "rmse_resid (hours)",
                      "External 25°C test: stability via fit residual scatter", ref_line=None)
        fig4a_delta_m_forest(outroot, outdir, dpi)
    thr_csv = outroot / "continuous_power" / "continuous_thresholds_by_model.csv"
    if not thr_csv.exists():
        print("[figures] Fig. 4B skipped: run the power-continuous step first")
    else:
        rows = read_csv_rows(thr_csv)
        delta_max = max(float(r["delta_abs"]) for r in rows)
        for suffix in (".png", ".pdf"):
            plot_threshold_curves(rows, (outdir / "fig4b_continuous_e80").with_suffix(suffix), models=MODELS, keys=("E80",),
                                  title="Continuous Effect-Size Planning Curve (E80)", y_label="Required embryos per group (E80)",
                                  delta_max=delta_max, dpi=dpi)
    return outdir


# ---------------------------------------------------------------------------
# Power and planning curves
# ---------------------------------------------------------------------------
def plot_power_curve(rows: Sequence[Mapping[str, Any]], out_png: str | Path, title: str) -> None:
    fig = plt.figure(figsize=(6.2, 4.0))
    plt.plot([r["E"] for r in rows], [r["power"] for r in rows], marker="o")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("E embryos per condition")
    plt.ylabel("Power (CI excludes 0)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, Path(out_png), dpi=200)
    plt.close(fig)


def monotone_smooth(points: Sequence[tuple[float, float]], samples_per_interval: int = 14) -> list[tuple[float, float]]:
    """Monotone cubic Hermite interpolation (Fritsch-Carlson): no overshoot between grid points."""
    if len(points) < 3:
        return list(points)
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    h = np.diff(x)
    if np.any(h <= 0):
        return list(points)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n, dtype=np.float64)
    m[0], m[-1] = d[0], d[-1]
    m[1:-1] = 0.5 * (d[:-1] + d[1:])
    for i in range(n - 1):
        if d[i] == 0.0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        s = a * a + b * b
        if s > 9.0:
            t = 3.0 / math.sqrt(s)
            m[i], m[i + 1] = t * a * d[i], t * b * d[i]
    out = [(float(x[0]), float(y[0]))]
    for i in range(n - 1):
        hi = x[i + 1] - x[i]
        for k in range(1, samples_per_interval + 1):
            t = k / samples_per_interval
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            out.append((float(x[i] + t * hi), float(h00 * y[i] + h10 * hi * m[i] + h01 * y[i + 1] + h11 * hi * m[i + 1])))
    return out


def _as_number(v: Any) -> float | None:
    """Threshold cell -> float, or None for '>Emax' / empty."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.startswith(">"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def plot_threshold_curves(rows: Iterable[Mapping[str, Any]], out_path: str | Path, *, models: Sequence[str], keys: Sequence[str],
                          title: str, y_label: str, delta_max: float, y_min: float = 0.0, y_max: float = 23.0,
                          y_tick_step: float = 1.0, footnote: str = "", dpi: int = 100) -> None:
    """Required embryos per group versus |Delta m|: smoothed curves, grid-point markers, gaps where not reached."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r["model"]), []).append(r)
    series = []
    for model in models:
        rr = sorted(by_model.get(model, []), key=lambda r: float(r["delta_abs"]))
        for key in keys:
            name = model if len(keys) == 1 else key
            color = PLANNING_COLORS.get(model, "#333333") if len(keys) == 1 else TARGET_COLORS.get(key, "#333333")
            series.append((name, color, [(float(r["delta_abs"]), _as_number(r[key])) for r in rr]))

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for name, color, pts in series:
        segments, seg = [], []
        for x, y in pts:
            if y is None:
                if seg:
                    segments.append(seg)
                seg = []
            else:
                seg.append((x, y))
        if seg:
            segments.append(seg)
        for seg in segments:
            sm = monotone_smooth(seg)
            ax.plot([p[0] for p in sm], [p[1] for p in sm], color=color, linewidth=2.5)
        valid = [(x, y) for x, y in pts if y is not None]
        ax.plot([p[0] for p in valid], [p[1] for p in valid], linestyle="none", marker="o", markersize=4, color=color)
        ax.plot([], [], color=color, linewidth=3, label=name)
    ax.set_xlim(0.0, delta_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0.0, delta_max + 1e-9, 0.05))
    ticks = np.arange(math.ceil(y_min / y_tick_step) * y_tick_step, y_max + 1e-9, y_tick_step)
    ax.set_yticks(ticks, [str(int(t)) if float(t).is_integer() else str(t) for t in ticks])
    ax.grid(True, color="#eeeeee", linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("|delta m|")
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(title="Legend", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _savefig(fig, Path(out_path), dpi=dpi)
    plt.close(fig)


def plot_local_slopes(series: Sequence[tuple[str, str, list[float], list[float]]], out_path: str | Path, title: str) -> None:
    """Local tempo slope versus nominal time (one line per condition)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for label, color, xs, ys in series:
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, color=color, label=label)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Nominal interval midpoint (hpf)")
    ax.set_ylabel("Local slope d(y_pred)/d(x_true)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    _savefig(fig, Path(out_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables 2, 5 and 6
# ---------------------------------------------------------------------------
def paper_tables(outroot: str | Path, out_md: str | Path | None = None) -> str:
    """Tables 2, 5 and 6 of the paper, recomputed from this run (three decimals)."""
    outroot = Path(outroot)
    order = list(reversed(MODELS))  # full, nocons, meanpool, cnn_single (as in the paper)

    def summary_rows(ds: str) -> list[dict[str, str]]:
        rows = []
        for m in order:
            s = read_json(outroot / ds / m / "summary.json")
            g, a, e = s["global_metrics_points"], s["fit_anchor_T0"], s["per_embryo_summary"]
            rows.append({"Model": "full (ETF)" if m == "full" else m, "MAE": fmt(g["mae"]), "RMSE": fmt(g["rmse"]), "R2": fmt(g["r2"]),
                         "m_anchor (mean)": fmt(e["m_mean"]), "rmse_resid": fmt(a["rmse_resid"]),
                         "max_abs_resid": fmt(a["max_abs_resid"]), "m_P05": fmt(e["m_p05"]), "m_P95": fmt(e["m_p95"])})
        return rows

    cols = ["Model", "MAE", "RMSE", "R2", "m_anchor (mean)", "rmse_resid", "max_abs_resid", "m_P05", "m_P95"]
    parts = [
        f"# Paper tables recomputed from `{outroot.name}`\n",
        "Window-level MAE/RMSE/R2 are descriptive (windows are correlated within embryos); errors are in hours.\n",
        "## Table 2. In-distribution test set, 28.5 °C (ID28C5_TEST)\n",
        markdown_table(summary_rows(ID28C5), cols),
        "## Table 5. Temperature-shift test set, 25 °C (EXT25C_TEST); MAE/RMSE/R2 are versus the nominal axis\n",
        markdown_table(summary_rows(EXT25C), cols),
    ]
    ci_rows = []
    for m in MODELS:
        p = outroot / f"CI_{m}_m_anchor.json"
        if p.exists():
            d = read_json(p)["delta"]
            ci_rows.append({"Model": "full (ETF)" if m == "full" else m, "Δm (25 °C − 28.5 °C)": fmt(d["delta_obs"]),
                            "95 % CI": f"[{fmt(d['ci95_low'])}, {fmt(d['ci95_high'])}]"})
    if ci_rows:
        parts += ["## Table 6. Temperature effect size with embryo-bootstrap 95 % CI\n",
                  markdown_table(ci_rows, ["Model", "Δm (25 °C − 28.5 °C)", "95 % CI"])]
    text = "\n".join(parts)
    out_md = Path(out_md) if out_md else outroot / "paper_tables.md"
    out_md.write_text(text, encoding="utf-8")
    print(f"[tables] {out_md}")
    return text
