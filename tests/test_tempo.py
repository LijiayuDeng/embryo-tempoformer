import csv
import json

import numpy as np

from etf.inference import trimmed_mean
from etf.tempo import aggregate_json_dir, fit_anchor


def test_fit_anchor_recovers_slope():
    x = 4.5 + 0.25 * np.arange(0, 170, 8, dtype=np.float64)
    y = 4.5 + 0.7 * (x - 4.5)
    m, resid = fit_anchor(x, y, 4.5)
    assert abs(m - 0.7) < 1e-12
    assert np.allclose(resid, 0.0)


def test_fit_anchor_degenerate():
    m, resid = fit_anchor(np.array([4.5]), np.array([5.0]), 4.5)
    assert np.isnan(m) and np.isnan(resid).all()


def _write_embryo(path, slope, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    starts = list(range(0, 169, 8))
    x = 4.5 + 0.25 * np.asarray(starts)
    y = 4.5 + slope * (x - 4.5) + rng.normal(0, noise, size=len(starts))
    t0_hats = (y - 0.25 * np.asarray(starts)).tolist()
    path.write_text(json.dumps({"t0_final": trimmed_mean(t0_hats), "starts": starts, "t0_hats": t0_hats}))


def test_aggregate_json_dir(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    _write_embryo(json_dir / "a.json", 1.0)
    _write_embryo(json_dir / "b.json", 0.7, noise=0.1, seed=1)
    (json_dir / "bad.json").write_text(json.dumps({"starts": [0, 8], "t0_hats": [4.5]}))  # malformed: skipped

    summary = aggregate_json_dir(json_dir, tmp_path, dt=0.25, t0=4.5, verbose=False)
    rows = {r["eid"]: r for r in csv.DictReader(open(tmp_path / "embryo.csv"))}
    assert set(rows) == {"a", "b"}
    assert abs(float(rows["a"]["m_anchor"]) - 1.0) < 1e-12
    assert abs(float(rows["b"]["m_anchor"]) - 0.7) < 0.01
    assert int(rows["a"]["n_points"]) == 22
    assert summary["global_metrics_points"]["n_embryos"] == 2
    assert summary["global_metrics_points"]["n_points"] == 44
    assert abs(summary["per_embryo_summary"]["m_mean"] - (1.0 + float(rows["b"]["m_anchor"])) / 2) < 1e-12


def test_trimmed_mean():
    assert trimmed_mean([]) != trimmed_mean([])  # nan
    assert trimmed_mean([1.0, 2.0, 3.0]) == 2.0  # k = 0: plain mean
    assert trimmed_mean([0, 1, 1, 1, 100], trim=0.2) == 1.0
