# Training

Reproducing the paper does not require training: `etf reproduce` uses the released
checkpoints. This page describes how they were trained and how to retrain.

## Model and objective

Each frame of a 24-frame clip (6 h at 15 min/frame) is encoded by a lightweight
depthwise-separable CNN with GroupNorm and squeeze-excitation into one 128-d
token. The tokens are aggregated over time and a small MLP head regresses the
developmental time in hpf. The four variants of the paper differ only in the
temporal aggregation and in the consistency loss:

| Variant | Temporal aggregation | Consistency weight `lambda_diff` | Parameters |
|---|---|---:|---:|
| `cnn_single` | first frame only (`identity`) | 0 | 0.276 M |
| `meanpool` | mean of the frame tokens | 0 | 0.276 M |
| `nocons` | Transformer, 4 blocks, RoPE, CLS token | 0 | 0.804 M |
| `full` (ETF) | Transformer, 4 blocks, RoPE, CLS token | 1 | 0.804 M |

Training samples two clips from the same embryo (starts s1, s2). The loss is

```
L = lambda_abs · L_abs + lambda_diff · ramp · L_diff
L_abs  = ½ [ l(t̂(s1), T0 + Δt·s1) + l(t̂(s2), T0 + Δt·s2) ]        (l = L1)
L_diff = SmoothL1( (t̂(s2) − t̂(s1)) − Δt·(s2 − s1) )
```

where `ramp` rises linearly from 0 to 1 over the first 20 % of optimisation steps.

## Released configuration

`etf train` defaults are the configuration stored in the released ETF-full
checkpoint; `--variant` switches to the other three:

| Group | Settings |
|---|---|
| Data | `clip_len 24`, `img_size 384`, `samples_per_embryo 32`, start jitter ±2 frames, mild augmentation (below) |
| Optimisation | AdamW, `lr 6e-4`, `weight_decay 0.01`, 1 % linear warm-up then cosine decay to 5 %, gradient clipping 1.0, 300 epochs, batch 32 per GPU (validation 64), mixed precision |
| Model | `model_dim 128`, 4 heads, depth 4, MLP ratio 2, dropout 0.1, temporal token dropout 0.05, `cnn_base 32` |
| EMA | decay 0.99 from the first step; validation and released weights use the EMA |
| Memory | `mem_profile lowmem`; seed 42 |

The exact configuration of any checkpoint is stored in it:

```bash
python -c "import json, torch; print(json.dumps(torch.load('full_best.pt', map_location='cpu', weights_only=False)['cfg'], indent=2))"
```

The released checkpoints were trained with DDP on 4 × RTX 3090 (about 13–15 GB per GPU), so each
optimisation step saw 4 × 32 = 128 pairs of clips. One GPU with the same settings takes four times
as many smaller steps; use `torchrun --nproc_per_node 4` (below) to match the released training.

## Commands

```bash
# ETF-full with the released configuration (one GPU)
etf train --proc_dir $RELEASE_DIR/processed_28C5 --split_json $RELEASE_DIR/splits/28C5.json --out_dir runs/train_full

# another variant
etf train --variant nocons --proc_dir ... --split_json ... --out_dir runs/train_nocons

# exactly the configuration of a checkpoint or an earlier run (paths can be overridden)
etf train --config $RELEASE_DIR/checkpoints/full_best.pt --proc_dir ... --split_json ... --out_dir runs/retrain_full

# several GPUs
torchrun --nproc_per_node 4 -m etf train --proc_dir ... --split_json ... --out_dir runs/train_full

# continue an interrupted run from its latest periodic checkpoint
etf train --config runs/train_full/run_config.json --resume runs/train_full/ckpt/epoch120.pt
```

Each run writes `best.pt` (best validation MAE), `last.pt`, `history.csv`,
`run_config.json` and periodic checkpoints in `ckpt/` (`--save_every`). Clip-level
validation metrics of a checkpoint: `etf eval --ckpt best.pt --proc_dir ... --split_json ...`.
These clip-level numbers are descriptive only: clips from one embryo are correlated.

## Memory profiles

`--mem_profile` trades compute for activation memory; it does not change the model.

| Profile | Frames per CNN call | Checkpointing | Fine-tuning `full`, RTX 4060 Laptop (8 GB) |
|---|---:|---|---|
| `fast` | 16 | none | does not fit |
| `balanced` | 8 | inside the CNN blocks | 251.7 s/epoch, 8.80 GB |
| `lowmem` (released) | 4 | whole frame encoder per chunk | 145.5 s/epoch, 5.12 GB |
| `ultra` | 1 | whole frame encoder per frame | 142.9 s/epoch, 2.31 GB |

At inference time only the number of frames per CNN call matters; the published
numbers were computed with `lowmem`, the default of `etf infer` and `etf infer-all`.

## Hardware and runtime

Measured on one NVIDIA RTX 4060 Laptop GPU (8 GB), Intel i7-13650HX, 15 GB RAM:

| Task | Settings | Runtime | Peak GPU memory |
|---|---|---|---|
| Fine-tuning `cnn_single`, `frame_tail1`, 12 embryos | batch 32/64, AMP, `lowmem` | 73.7 s/epoch | 3.66 GB |
| Fine-tuning `full`, `temporal_last1`, 12 embryos | batch 16/32, AMP, `lowmem` | 135.7 s/epoch | 5.15 GB |

## Augmentation and the ablation (Table 4)

Training-time augmentation is mild and organised in four families, each of which
can be switched off with `--aug_disable_groups`:

| Family | Transforms |
|---|---|
| `spatial` | horizontal flip; one affine transform (±8°, ±3 % shift, 0.97–1.03 scale) shared by all frames |
| `photometric` | gamma, contrast, brightness, low-frequency shading |
| `acquisition` | additive Gaussian noise, box blur |
| `temporal` | frame drop (repeat the previous frame); start jitter is set separately with `--jitter` |

`etf ablation` retrains ETF-full five times, removing one family at a time
(`no_temporal_sampling` removes frame drop and start jitter, not the model's
temporal token dropout), then evaluates and summarises:

```bash
etf ablation --outroot runs/aug_ablation            # train, evaluate, summarise
etf ablation --outroot runs/aug_ablation --dry_run  # only write the configurations
```

Defaults match the Table 4 runs: one NVIDIA L40S (46 GB), batch 88 (validation 176),
up to 300 epochs with early stopping after 20 epochs without improvement, seed 42.
On smaller GPUs reduce `--batch_size`; results then differ from Table 4. Outputs:
`<setting>/best.pt`, `<setting>/eval_main/...`, `logs/<setting>.log`,
`min_aug_ablation_summary.{csv,md}`.

| Setting | ID 28.5 °C MAE (h) | EXT 25 °C `m_anchor` | Best epoch | Training time (L40S) |
|---|---:|---:|---:|---:|
| `baseline_full` | 0.818 | 0.710 | 122 | 10.8 h |
| `no_spatial` | 1.324 | 0.719 | 68 | 6.6 h |
| `no_photometric` | 0.745 | 0.790 | 157 | 13.2 h |
| `no_acquisition` | 0.900 | 0.709 | 54 | 5.5 h |
| `no_temporal_sampling` | 0.749 | 0.711 | 133 | 11.6 h |

This is a single-seed sensitivity analysis of the recipe, not a new hyperparameter
search: lower in-distribution error (for example without photometric
augmentation) does not imply a better external tempo estimate.
