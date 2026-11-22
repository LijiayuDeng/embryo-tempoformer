# Reproducing the paper

`etf reproduce` recomputes every quantitative result of the main benchmark from
the released checkpoints; no training is involved.

## Run

```bash
cp .env.example .env            # set RELEASE_DIR to the extracted bundle
etf check                       # verifies all paths
etf reproduce                   # -> runs/paper_eval_<timestamp>/
```

`bash scripts/reproduce_all.sh` is equivalent. Options:

| Option | Effect |
|---|---|
| `--outroot DIR` | output directory (default `runs/paper_eval_<timestamp>`) |
| `--skip STEP,...` | skip steps, e.g. `--skip cliplen` (the figures need `stats`; Fig. 4B needs `power-continuous`) |
| `--saliency` | also compute the SmoothGrad maps of Fig. 5 (qualitative) |
| `--saliency_model MODEL` | model for the saliency maps (default `full`) |
| `--force` | recompute inference and aggregation even if their outputs exist |

Rerunning with the same `--outroot` reuses finished inference files, so an
interrupted run continues where it stopped; results computed with different
settings are never mixed (each result folder records its settings in
`infer_settings.json`). The whole run takes about 16 minutes on an RTX 4060
Laptop GPU, mostly inference (the 1.x scripts needed 54 minutes for the same work).

## Steps

| Step | What it does | Main outputs |
|---|---|---|
| `infer` | sliding-window inference (24-frame clips, stride 8) of the four models on the 28.5 °C (n = 22) and 25 °C (n = 95) test embryos | `<dataset>/<model>/json/<eid>.json` |
| `aggregate` | per-embryo anchored fits and pooled window metrics | `<dataset>/<model>/{points.csv, embryo.csv, summary.json}` |
| `stats` | Δm with embryo-bootstrap 95 % CI (B = 5000, seed 0) and power versus embryos per condition | `CI_<model>_m_anchor.json`, `power_<model>_m_anchor.{csv,png}` |
| `power-continuous` | embryos needed for 80/90/95 % power over \|Δm\| = 0–0.10 | `continuous_power/` |
| `cliplen` | released ETF-full applied to 4-, 12- and 24-frame clips (1, 3, 6 h) and the 0/1/3/6 h context ladder | `cliplen_sensitivity/` |
| `stage-error` | window-level error by Kimmel period | `stage_error/stage_error_by_bin.csv` |
| `anchor` | readouts re-aggregated with T0 = 4.0, 4.5 and 5.0 hpf | `anchor_sensitivity/` |
| `stage-tempo` | piecewise tempo slopes per Kimmel period with embryo-bootstrap CIs | `stage_tempo/` |
| `figures` | figures and tables of the paper | `figures/`, `paper_tables.md` |

Every step is also a command (`etf infer-all`, `etf aggregate`, `etf analyze stats`,
...; see `etf --help`), and `run_info.json` records the software versions and settings.

## Outputs and the paper

| Paper | Output |
|---|---|
| Table 2 (28.5 °C test) | `paper_tables.md`; `ID28C5_TEST/<model>/summary.json` |
| Table 3 (RMSE ladder) | RMSE column of Table 2 |
| Table 5 (25 °C test) | `paper_tables.md`; `EXT25C_TEST/<model>/summary.json` |
| Table 6 (Δm and 95 % CI) | `paper_tables.md`; `CI_<model>_m_anchor.json` |
| Fig. 2 | `figures/fig2_id_ablation.{png,pdf}` |
| Fig. 3A, 3B | `figures/fig3a_ext_m_anchor`, `figures/fig3b_ext_rmse_resid` |
| Fig. 4A, 4B | `figures/fig4a_delta_m_forest`, `figures/fig4b_continuous_e80` |
| Fig. 5 | `vis_best_<eid>_full_smoothgrad_five/` (with `--saliency`) |
| Supplementary planning curves and points | `continuous_power/`, `power_<model>_m_anchor.csv` |
| Supplementary anchor, stage-error and stage-tempo tables | `anchor_sensitivity/`, `stage_error/`, `stage_tempo/` |
| Clip-length ladder | `cliplen_sensitivity/context_fit/` |
| Table 4 (augmentation ablation) | `etf ablation` (retrains five models; see [training.md](training.md)) |
| Table 7 (Princeton cross-site) | `etf sbiad840 ...` (see [external-validation.md](external-validation.md)) |

## Readouts

For each embryo, window start `s` has nominal time `x = T0 + Δt·s` and predicted
time `y = t̂(s)` (T0 = 4.5 hpf, Δt = 0.25 h). The embryo-level readouts come from the
anchored least-squares fit `y − T0 = m (x − T0)`:

- `m_anchor`: developmental tempo relative to the nominal axis (1 = nominal,
  < 1 = slower, > 1 = faster);
- `rmse_resid`, `max_abs_resid`: how closely the windows of an embryo follow one
  anchored trajectory (lower = more consistent).

Window-level MAE, RMSE and R² pool correlated windows and are descriptive only.
At 25 °C the nominal axis is not the true developmental clock, so they measure
deviation from nominal time rather than accuracy; `t0_final` (trimmed mean of
`t̂(s) − Δt·s`) is a descriptive start-time diagnostic.

## Statistics

Embryos are the independent units throughout. Δm = mean(m_anchor, 25 °C) −
mean(m_anchor, 28.5 °C); its 95 % CI is the percentile bootstrap obtained by
resampling embryos with replacement within each condition. Power is the fraction
of simulated experiments with E embryos per condition whose bootstrap CI excludes
zero (500 experiments × 500 bootstrap replicates). The continuous planning curves
resample pooled centred residuals around shifted means (1200 × 800) and apply two
shape constraints: power never decreases with E, and the required E never
increases with |Δm|. Stage-wise tempo slopes are ordinary least-squares fits within
each Kimmel period with embryo-bootstrap CIs (3000 replicates).

## Numerical reproducibility

The published numbers were computed with PyTorch 2.5.1, CUDA 12.4 and cuDNN 9.1
(`requirements-lock.txt`). With these versions on the GPU used for the paper
(RTX 4060 Laptop), every per-embryo inference file, table, confidence interval and
power table produced by version 2.0 is byte-identical to the published run.
Other GPUs or library versions can change the last decimals. The tables agree with
the paper to three decimals, except four window-level entries of Table 2 that
differ by 0.001 h (full `max_abs_resid`; meanpool MAE, `rmse_resid`, `max_abs_resid`).
