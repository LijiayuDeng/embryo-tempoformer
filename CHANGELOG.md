# Changelog

## 2.0.0 (unreleased)

Restructured release. Numerical results are unchanged: with the same hardware and
library versions, every per-embryo inference file, tempo table, confidence
interval, power table, sensitivity table and S-BIAD840 table is identical to
version 1.1.0, and training, fine-tuning and ablation configurations are identical.

### Structure
- One installable package (`src/etf`) and one command, `etf`, replace the 24
  numbered shell scripts and 29 analysis scripts (mapping below).
  `scripts/reproduce_all.sh` remains and still honours the 1.x environment switches
  (`OUTROOT`, `RUN_*`, `SALIENCY_MODEL`, `PYTHON_BIN`).
- Inference runs in one process and loads each model once: the full reproduction
  takes about 16 minutes instead of 54 on an RTX 4060 Laptop GPU.
- `.env` only needs `RELEASE_DIR` (and `SBIAD840_DIR`); the individual path keys of
  1.x still work and take precedence. Still read: `RUNS_DIR`, `DEVICE`, `AMP`, `USE_EMA`,
  `DT_H`, `T0_HPF`, `CLIP_LEN`, `STRIDE`, `SBIAD840_SRC_ROOT`, `SBIAD840_OUT_ROOT`.
  No longer read: `BATCH_SIZE` (it had no effect on inference), `IMG_SIZE`, `EXPECT_T`
  (now command options) and the per-script variables of 1.x (`DATASETS`, `MODELS`,
  `FORCE`, `SEED`, `E_LIST`, `T0_LIST`, `SAL_*`, ...), which are now command options.

### Outputs
- `etf reproduce` also runs the stage-dependent tempo analysis (the 1.x pipeline
  skipped it although the documentation listed it) and writes `paper_tables.md`
  (Tables 2, 5 and 6) and `run_info.json` (versions, settings; replaces `OUTROOT.txt`).
- Figures are written to `figures/` (was `figures_jobs/`) and named after the paper
  panels (`fig2_id_ablation`, `fig3a_ext_m_anchor`, `fig3b_ext_rmse_resid`,
  `fig4a_delta_m_forest`, and the new `fig4b_continuous_e80`). The planning curves
  are drawn with matplotlib instead of hand-written SVG (same data and smoothing).
  SVG and PDF files no longer embed timestamps.
- Each inference folder records its settings in `infer_settings.json`; reusing a
  folder with different settings (e.g. another clip length) is refused unless
  `--force` is given. `etf aggregate` re-aggregates when the number of inference
  files changed. JSON files are written atomically.

### Defaults and behaviour
- `etf train` defaults are the released ETF-full configuration;
  `--variant cnn_single|meanpool|nocons|full` selects the others and `--config`
  starts from a run_config.json or checkpoint. `--device` is honoured for
  single-process training.
- `etf infer` and `etf eval` use EMA weights and the `lowmem` profile by default,
  as the 1.x pipeline did (the bare 1.x commands defaulted to raw weights and `balanced`).
- `etf infer` keeps the recorded length of TIFF inputs instead of padding to 192 frames.
- `etf analyze compare` (formerly `analysis/ci_delta_m.py` and `analysis/power_curve.py`)
  uses R = B = 500 for the power curve, as in the paper's analysis; the standalone
  1.x script defaulted to 1000.
- `etf ablation` defaults match the Table 4 runs (batch 88/176, patience 20,
  `save_every 25`, 16 workers); `--set KEY=VALUE` changes any other training
  option. The GPU presets (`GPU_PRESET=rtx5090|l40s`) and the tmux launcher are gone.
- SmoothGrad saliency takes a seed (`--seed`, default 0) and is reproducible.
- Option renames: `--enforce_monotone_power_e` / `--enforce_monotone_threshold_delta`
  are `--[no-]monotone_power` / `--[no-]monotone_threshold`; positional output roots
  are `--outroot`; analysis commands no longer pick the latest `runs/paper_eval_*`
  automatically.

### Other
- Requires PyTorch >= 2.3; `requirements-lock.txt` lists the versions used to verify
  the published numbers. The Docker image uses PyTorch 2.5.1 / CUDA 12.4.
- New: `etf infer-all --input_dir` for own recordings, `etf check`, unit tests and CI.
- Fixed: `--num_workers 1` crashed the validation loader.
- Removed: personal paths and dated default directories, the unused `pandas`
  dependency, `environment.yml` and the code-review templates.

| 1.x | 2.0 |
|---|---|
| `python src/EmbryoTempoFormer.py {preprocess,make_split,train,eval,infer}` | `etf {preprocess,make-split,train,eval,infer}` |
| `scripts/reproduce_all.sh`, `00_check_env.sh` | `etf reproduce` (the script is kept), `etf check` |
| `10_infer_all.sh`, `20_aggregate_all.sh` | `etf infer-all`, `etf aggregate` |
| `30_ci_power_all.sh` | `etf analyze stats` |
| `analysis/ci_delta_m.py`, `analysis/power_curve.py` | `etf analyze compare` |
| `31_power_curve_continuous.sh` | `etf analyze power-continuous` |
| `11_cliplen_sensitivity.sh`, `12_cliplen_context_fit.sh` | `etf analyze cliplen` |
| `32_stage_error_bins.sh`, `33_anchor_sensitivity.sh`, `44_stage_tempo_dependence.sh` | `etf analyze stage-error`, `anchor`, `stage-tempo` |
| `40_make_figures.sh`, `50_saliency_best_id28c5.sh` | `etf figures`, `etf saliency` |
| `34_preprocess_sbiad840.sh` | `etf sbiad840 preprocess` |
| `10_infer_all.sh` with `DATASETS=SBIAD840_*`, `35_aggregate_sbiad840.sh` | `etf infer-all --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST`, `etf aggregate --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST` |
| `36_summarize_sbiad840.sh` | `etf sbiad840 summarize` |
| `37_infer_sbiad840_cnn_dense.sh` | `etf infer-all --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST --models cnn_single --clip_len 1 --stride 1` |
| `38_compare_sbiad840_kimmelnet.sh` | `etf sbiad840 compare-kimmelnet` |
| `39_make_sbiad840_finetune_splits.sh` | `etf sbiad840 make-splits` |
| `41_finetune_sbiad840.sh` | `etf sbiad840 finetune` |
| `42_summarize_sbiad840_finetune.sh` (`summarize_sbiad840_transfer.py`) | `etf sbiad840 summarize-finetune` |
| `43_eval_sbiad840_finetuned.sh` | `etf sbiad840 eval-finetuned` |
| `45`–`47` (augmentation ablation) | `etf ablation` |

## 1.1.0 (2026-04-08)

Code at publication (Deng et al., *Artificial Intelligence in the Life Sciences*
9, 100170, 2026): adds the Princeton S-BIAD840 cross-site evaluation and low-shot
fine-tuning, the augmentation ablation, clip-length, anchor and stage-dependent
sensitivity analyses, continuous planning curves and a Docker image.

## 1.0.0 (2026-01-21)

First release: inference, aggregation, embryo-bootstrap CIs, power curves and
figures from the released checkpoints (Zenodo 10.5281/zenodo.18318139).
