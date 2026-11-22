# Cross-site validation on Princeton data (S-BIAD840)

The 25 °C test set probes a temperature shift within the same imaging setup. The
Princeton data set S-BIAD840 adds a different site: other microscope, image
scale, background and illumination (paper Section 4.6, Table 7). The released
checkpoints are first applied unchanged (zero-shot) and then adapted with a few
Princeton embryos (low-shot fine-tuning).

## Data

Download the processed bundle ([10.5281/zenodo.18979476](https://doi.org/10.5281/zenodo.18979476))
and extract it into `data/` (the default `SBIAD840_DIR`):

```bash
mkdir -p data && tar -xzf sbiad840_aligned_4p5.tar.gz -C data
etf check --with_sbiad840
```

Or rebuild it from the BioImage Archive PNG export:

```bash
etf sbiad840 preprocess --src_root /path/to/S-BIAD840/Files/Princeton_Data
```

S-BIAD840 stores one directory per time point (`Dataset_C`: 28.5 °C from 3.5 hpf;
`Dataset_D`: 25 °C from 2.5 hpf), each holding one PNG per embryo. The import
rebuilds one stack per embryo, drops the frames before 4.5 hpf so that the time
base matches the released checkpoints (4 frames for Dataset_C, 8 for Dataset_D,
leaving 168 frames up to 46.25 hpf) and applies the standard preprocessing
(1–99th percentile clipping, 384 × 384). Stacks keep their native length; use
`--pad_to_expect` for 192-frame arrays.

## Zero-shot evaluation

```bash
etf infer-all --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST --outroot runs/sbiad840_zero_shot
etf aggregate --outroot runs/sbiad840_zero_shot --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST
etf sbiad840 summarize --outroot runs/sbiad840_zero_shot
```

`sbiad840_external_summary.{csv,md}` reports ETF's anchored readouts and, for
comparison with KimmelNet, the through-origin fit `y = m·x` (`m_origin`, residual
mean ± SD and line-fit R²).

## KimmelNet comparison

KimmelNet predicts from single frames, so the matched ETF baseline is
`cnn_single` applied to every frame:

```bash
etf infer-all --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST --models cnn_single --clip_len 1 --stride 1 --outroot runs/sbiad840_dense
etf aggregate --outroot runs/sbiad840_dense --datasets SBIAD840_28C5_TEST,SBIAD840_25C_TEST --models cnn_single
etf sbiad840 summarize --outroot runs/sbiad840_dense --models cnn_single
etf sbiad840 compare-kimmelnet --base_outroot runs/sbiad840_zero_shot --dense_outroot runs/sbiad840_dense
```

This writes `runs/sbiad840_zero_shot/sbiad840_vs_kimmelnet.{csv,md}`. The KimmelNet
values are taken from Jones, Renshaw & Barry, *Life Science Alliance* 7,
e202302351 (2024).

## Low-shot fine-tuning

Fine-tuning uses Princeton 28.5 °C embryos only; the Princeton 25 °C embryos stay
an external test set.

1. Split the 96 Princeton 28.5 °C embryos (seed 42) into 12 train / 12 val / 72 test
   and 24 / 12 / 60:

   ```bash
   etf sbiad840 make-splits      # -> data/sbiad840_aligned_4p5/splits/finetune/
   ```

2. Fine-tune a released checkpoint. Training starts from its EMA weights with a
   fresh optimiser and schedule; architecture and optimiser settings come from the
   checkpoint. The settings of the two Table 7 runs:

   ```bash
   etf sbiad840 finetune --model cnn_single --stage frame_tail1 --num_workers 0 --out_dir runs/finetune_cnn_single_frame_tail1
   etf sbiad840 finetune --model full --stage temporal_last1 --epochs 10 --batch_size 16 --val_batch_size 32 --num_workers 0 \
     --out_dir runs/finetune_full_temporal_last1
   ```

   The published `cnn_single` run was stopped after 14 of its 30 scheduled epochs
   (best validation after epoch 11, `ckpt/epoch11.pt`). The fine-tuned checkpoints are not part of the data
   bundles, and GPU training is not bit-for-bit deterministic, so retrained models
   reproduce Table 7 closely rather than exactly.

   `--stage` chooses what is trained (`--dry_run` prints the resolved configuration):

   | Stage | Trainable parameters |
   |---|---|
   | `head_only` | regression head |
   | `proj_head` | head + frame projection |
   | `frame_tail1`, `frame_tail2` | head + frame projection + last 1 or 2 CNN blocks |
   | `temporal_last1`, `temporal_last2` | head + last 1 or 2 Transformer blocks + CLS token and final norm (Transformer models) |
   | `temporal_head` | everything except the frame encoder |
   | `full_trainable` | everything |

   Start with the lightest stage and widen only if calibration stays poor; the
   paper reports `frame_tail1` for `cnn_single` and `temporal_last1` for `full`.

3. Evaluate on the 72 held-out 28.5 °C embryos (the test list of the fine-tune
   split) and on all 96 Princeton 25 °C embryos, then tabulate (paths assume the
   default `SBIAD840_DIR`):

   ```bash
   SPLIT=data/sbiad840_aligned_4p5/splits/finetune/28C5_sbiad840_ft12_v12_seed42.json
   etf sbiad840 eval-finetuned --model cnn_single --ft_ckpt runs/finetune_cnn_single_frame_tail1/best.pt --split_28c5 $SPLIT --outroot runs/sbiad840_ft12_cnn_single
   etf sbiad840 eval-finetuned --model full --ft_ckpt runs/finetune_full_temporal_last1/best.pt --split_28c5 $SPLIT --outroot runs/sbiad840_ft12_full
   etf sbiad840 summarize-finetune \
     --experiment cnn_single_frame_tail1_ft12 runs/finetune_cnn_single_frame_tail1 runs/sbiad840_ft12_cnn_single runs/sbiad840_ft12_cnn_single \
     --experiment full_temporal_last1_ft12 runs/finetune_full_temporal_last1 runs/sbiad840_ft12_full runs/sbiad840_ft12_full
   ```

## Results (paper Table 7)

| Dataset | Model / setting | RMSE (h) | `m_origin` | Residual mean ± SD (h) | Line-fit R² |
|---|---|---:|---:|---:|---:|
| Princeton 28.5 °C | `cnn_single` zero-shot | 14.339 | 1.288 | 5.07 ± 11.31 | −3.611 |
| | `full` zero-shot | 9.862 | 1.203 | 2.78 ± 7.99 | 0.179 |
| | `cnn_single` + 12 embryos, `frame_tail1` | 5.270 | 1.003 | 0.98 ± 5.18 | 0.732 |
| | `full` + 12 embryos, `temporal_last1` | 3.556 | 0.948 | 0.47 ± 3.28 | 0.889 |
| Princeton 25 °C | `cnn_single` zero-shot | 14.116 | 1.260 | 5.09 ± 11.44 | −3.783 |
| | `full` zero-shot | 10.643 | 1.151 | 3.50 ± 9.31 | −0.531 |
| | `cnn_single` + 12 embryos, `frame_tail1` | 5.017 | 0.918 | 0.84 ± 4.50 | 0.759 |
| | `full` + 12 embryos, `temporal_last1` | 5.218 | 0.862 | 0.84 ± 3.81 | 0.785 |

Zero-shot transfer across sites is limited mainly by calibration; fine-tuning on
12 embryos recovers most of the lost accuracy. This is an initial cross-site test,
not evidence of site-invariant generalisation.
