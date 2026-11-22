import csv

import numpy as np

from etf.stats import (
    bootstrap_ci,
    ci_delta_m,
    default_E_list,
    describe,
    monotone_power_over_e,
    monotone_thresholds_over_delta,
    thresholds,
)


def _embryo_csv(path, values):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eid", "m_anchor"])
        w.writerows([[f"e{i}", v] for i, v in enumerate(values)] + [["bad", "nan"]])


def test_bootstrap_ci_is_seeded_and_covers_the_effect():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.7, 0.03, 95), rng.normal(1.0, 0.03, 22)
    ci1 = bootstrap_ci(a, b, 2000, np.random.default_rng(1))
    ci2 = bootstrap_ci(a, b, 2000, np.random.default_rng(1))
    assert ci1 == ci2
    assert ci1[0] < a.mean() - b.mean() < ci1[1] < 0


def test_ci_delta_m(tmp_path):
    _embryo_csv(tmp_path / "a.csv", [0.70, 0.71, 0.69, 0.72])
    _embryo_csv(tmp_path / "b.csv", [1.00, 1.01, 0.99])
    out = ci_delta_m(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "ci.json", n_boot=500, seed=0)
    assert out["group_a"]["n"] == 4 and out["group_b"]["n"] == 3  # non-finite rows dropped
    assert abs(out["delta"]["delta_obs"] - (0.705 - 1.0)) < 1e-12
    assert out["delta"]["ci95_low"] <= out["delta"]["delta_obs"] <= out["delta"]["ci95_high"]


def test_describe_ignores_non_finite():
    d = describe(np.array([1.0, 2.0, np.nan, 3.0]))
    assert d["n"] == 3 and d["mean"] == 2.0 and d["median"] == 2.0


def test_default_E_list():
    assert default_E_list(22) == [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    assert default_E_list(2) == [2]


def test_monotone_postprocessing():
    rows = [{"delta_abs": 0.02, "E_per_group": e, "power": p} for e, p in [(2, 0.1), (4, 0.5), (6, 0.45), (8, 0.85)]]
    mono = monotone_power_over_e(rows)
    assert [r["power"] for r in mono] == [0.1, 0.5, 0.5, 0.85]
    assert [r["power_raw"] for r in mono] == [0.1, 0.5, 0.45, 0.85]

    thr = thresholds(mono, [2, 4, 6, 8], [0.8, 0.9])
    assert thr[0]["E80"] == 8 and thr[0]["E90"] == ">8"

    rows2 = [{"delta_abs": 0.01, "E80": 6}, {"delta_abs": 0.02, "E80": 8}, {"delta_abs": 0.03, "E80": ">8"}]
    out = monotone_thresholds_over_delta(rows2, [0.8], e_max=8)
    assert [r["E80"] for r in out] == [6, 6, 6]
