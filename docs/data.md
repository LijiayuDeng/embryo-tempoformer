# Data

## Bundles

| Bundle | Contents | DOI |
|---|---|---|
| `embryo-tempoformer_release_v1_full.tar.gz` (main) | processed arrays (28.5 °C and 25 °C), split files, the four released checkpoints, `MANIFEST.json`, `SHA256SUMS.txt` | [10.5281/zenodo.18318139](https://doi.org/10.5281/zenodo.18318139) |
| `sbiad840_aligned_4p5.tar.gz` (optional) | Princeton S-BIAD840 arrays aligned to 4.5 hpf and test-only split files | [10.5281/zenodo.18979476](https://doi.org/10.5281/zenodo.18979476) |

```text
embryo-tempoformer_release_v1/          # RELEASE_DIR
├── processed_28C5/   <eid>.npy         # 28.5 °C, uint8 [192, 384, 384]
├── processed_25C/    <eid>.npy         # 25 °C
├── splits/           28C5.json  25C.json
├── checkpoints/      cnn_single_best.pt  meanpool_best.pt  nocons_best.pt  full_best.pt
├── MANIFEST.json
└── SHA256SUMS.txt                      # verify with: sha256sum -c SHA256SUMS.txt

data/sbiad840_aligned_4p5/              # SBIAD840_DIR (default location)
├── processed_28C5_sbiad840/  processed_25C_sbiad840/
└── splits/  28C5_sbiad840_test.json  25C_sbiad840_test.json
```

`etf` finds every file from `RELEASE_DIR` and `SBIAD840_DIR` in `.env`. Individual
keys (`PROC_28C5`, `SPLIT_25C`, `CKPT_FULL`, ...; see `.env.example`) override the
bundle layout, for example to evaluate a different checkpoint.

The raw recordings are on the BioImage Archive:
[S-BIAD531](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD531) (benchmark) and
[S-BIAD840](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD840) (Princeton cross-site data).

## Time base

S-BIAD531 recordings start at T0 = 4.5 hpf with one frame every Δt = 0.25 h (15 min).
Frame `s` therefore has nominal time `T0 + Δt·s`; 192 frames cover 4.5–52.25 hpf.
The released checkpoints assume this sampling.

## Processed arrays

`etf preprocess` turns raw OME-TIFF stacks into the processed format (paper Section 3.1):

1. clip intensities to the 1st–99th percentile of the stack and map linearly to 0–255;
2. resize every frame to 384 × 384 (bilinear);
3. pad (repeat the last frame) or trim to 192 frames;
4. save one `<eid>.npy` per embryo; the embryo id (eid) is the file stem.

```bash
etf preprocess --in_dir /path/to/ome_tiffs --proc_dir processed_28C5
```

## Splits and exclusions

Split files are JSON objects listing embryo ids:

```json
{"train": ["<eid>", "..."], "val": ["..."], "test": ["..."]}
```

Splits are made per embryo (one embryo = one stack), never per frame or clip.

| Condition | Raw embryos | Excluded | Included | Train / val / test |
|---|---:|---:|---:|---|
| 28.5 °C (`ID28C5`) | 192 | 51 | 141 | 98 / 21 / 22 |
| 25 °C (`EXT25C`) | 96 | 1 | 95 | test only |

28.5 °C embryos were excluded when non-viable or outside the field of view,
following the KimmelNet preprocessing and checked manually; the 25 °C file
`FishDev_WT_25C_1_MMStack_G11-Site_0.ome` was excluded by name (paper Table 1).
The released split files encode all exclusions and are the reference for
reproducing the paper. `etf.data.EXCLUDE_KEYS` lists 39 of the 51 excluded
28.5 °C wells, which `etf preprocess` and `etf make-split` skip; it does not cover
the other exclusions, so when rebuilding arrays from the raw stacks, take the
embryo lists from the released split files.

`etf make-split` creates a new random embryo-level split (default 70/15/15);
reproducing the paper requires the released split files.
