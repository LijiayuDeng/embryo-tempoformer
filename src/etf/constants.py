"""Protocol constants shared across the package (values used in the paper)."""

# Time base of the S-BIAD531 recordings: first frame at 4.5 hpf, one frame every 15 min.
T0_HPF = 4.5
DT_H = 0.25

# Clip / array geometry of the released checkpoints and processed arrays.
CLIP_LEN = 24
IMG_SIZE = 384
EXPECT_T = 192

# Sliding-window inference.
STRIDE = 8
TRIM = 0.2  # trimmed-mean proportion for the descriptive t0_final summary

# Model variants (ablation ladder) in the order used in the paper.
MODELS = ("cnn_single", "meanpool", "nocons", "full")
MODEL_LABELS = {
    "cnn_single": "CNN single",
    "meanpool": "Meanpool",
    "nocons": "No-cons",
    "full": "Full",
}

# Temporal aggregation of frame tokens: single first frame, mean pooling, Transformer.
TEMPORAL_MODES = ("transformer", "meanpool", "identity")

# Memory profiles trade compute for activation memory during training. In eval
# mode only ``frame_chunk`` matters (how many frames are encoded per CNN call).
MEM_PROFILES = {
    "fast": dict(frame_chunk=16, ckpt_frame=False, ckpt_cnn=False, ckpt_segments=1),  # fastest, most memory
    "balanced": dict(frame_chunk=8, ckpt_frame=False, ckpt_cnn=True, ckpt_segments=2),  # checkpoint inside CNN blocks
    "lowmem": dict(frame_chunk=4, ckpt_frame=True, ckpt_cnn=False, ckpt_segments=1),  # released checkpoints
    "ultra": dict(frame_chunk=1, ckpt_frame=True, ckpt_cnn=False, ckpt_segments=1),  # one frame at a time
}

# Dataset tags.
ID28C5 = "ID28C5_TEST"  # in-distribution 28.5 C test split (n = 22 embryos)
EXT25C = "EXT25C_TEST"  # held-out 25 C temperature-shift test set (n = 95)
SBIAD840_28C5 = "SBIAD840_28C5_TEST"  # Princeton cross-site data, 28.5 C
SBIAD840_25C = "SBIAD840_25C_TEST"  # Princeton cross-site data, 25 C
MAIN_DATASETS = (ID28C5, EXT25C)
SBIAD840_DATASETS = (SBIAD840_28C5, SBIAD840_25C)
