"""Configuration: the training configuration (``TrainConfig``) and the data /
checkpoint paths and runtime settings read from ``.env``.

Settings precedence: command-line option > environment variable > ``.env`` > default.
Relative paths are resolved against the current working directory (run ``etf``
from the repository root). See ``.env.example`` for the keys.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CLIP_LEN, DT_H, EXPECT_T, EXT25C, ID28C5, IMG_SIZE, MODELS, SBIAD840_25C, SBIAD840_28C5, STRIDE, T0_HPF

# Settings that distinguish the four model variants of the paper (Section 3.6).
VARIANTS: dict[str, dict[str, Any]] = {
    "cnn_single": dict(temporal_mode="identity", model_depth=0, temporal_drop_p=0.0, lambda_diff=0.0),
    "meanpool": dict(temporal_mode="meanpool", model_depth=0, temporal_drop_p=0.05, lambda_diff=0.0),
    "nocons": dict(temporal_mode="transformer", model_depth=4, temporal_drop_p=0.05, lambda_diff=0.0),
    "full": dict(temporal_mode="transformer", model_depth=4, temporal_drop_p=0.05, lambda_diff=1.0),
}


@dataclass
class TrainConfig:
    """Training configuration, stored in every checkpoint under ``cfg``.

    Defaults reproduce the released ETF-full checkpoint (other variants: see VARIANTS).
    """

    proc_dir: str = ""
    split_json: str = ""
    out_dir: str = ""

    clip_len: int = CLIP_LEN
    img_size: int = IMG_SIZE
    expect_t: int = EXPECT_T

    batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 8
    samples_per_embryo: int = 32
    jitter: int = 2
    aug_disable_groups: str = ""
    cache_items: int = 16
    val_cache_items: int = 16

    epochs: int = 300
    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    lr_min_ratio: float = 0.05
    max_grad_norm: float = 1.0
    grad_accum: int = 1

    model_dim: int = 128
    model_depth: int = 4
    model_heads: int = 4
    model_mlp_ratio: float = 2.0
    drop: float = 0.1
    attn_drop: float = 0.0
    temporal_drop_p: float = 0.05
    temporal_mode: str = "transformer"

    cnn_base: int = 32
    cnn_expand: int = 2
    cnn_se_reduction: int = 4

    mem_profile: str = "lowmem"

    lambda_abs: float = 1.0
    lambda_diff: float = 1.0
    cons_ramp_ratio: float = 0.2
    abs_loss_type: str = "l1"  # l1 | smoothl1

    amp: bool = True
    ema_decay: float = 0.99
    ema_start_ratio: float = 0.0
    ema_eval: bool = True

    seed: int = 42
    device: str = "auto"
    resume: str = ""
    init_ckpt: str = ""
    init_use_ema: bool = False
    freeze_frame_encoder: bool = False
    freeze_temporal: bool = False
    freeze_head: bool = False
    unfreeze_frame_proj: bool = False
    unfreeze_frame_tail_blocks: int = 0
    unfreeze_temporal_tail_blocks: int = 0
    save_every: int = 1
    patience: int = 0
    val_every: int = 1

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], **overrides: Any) -> TrainConfig:
        """Build from a (checkpoint) dict, ignoring unknown keys."""
        fields = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in d.items() if k in fields}
        values.update(overrides)
        return cls(**values)

# Dataset tag -> (.env key of the processed-array directory, .env key of the split file).
DATASET_KEYS = {
    ID28C5: ("PROC_28C5", "SPLIT_28C5"),
    EXT25C: ("PROC_25C", "SPLIT_25C"),
    SBIAD840_28C5: ("PROC_28C5_SBIAD840", "SPLIT_28C5_SBIAD840"),
    SBIAD840_25C: ("PROC_25C_SBIAD840", "SPLIT_25C_SBIAD840"),
}
# Model tag -> .env key of the released checkpoint.
CHECKPOINT_KEYS = {
    "cnn_single": "CKPT_CNN_SINGLE",
    "meanpool": "CKPT_MEANPOOL",
    "nocons": "CKPT_NOCONS",
    "full": "CKPT_FULL",
}
REQUIRED_KEYS = [*DATASET_KEYS[ID28C5], *DATASET_KEYS[EXT25C], *CHECKPOINT_KEYS.values()]
SBIAD840_KEYS = [*DATASET_KEYS[SBIAD840_28C5], *DATASET_KEYS[SBIAD840_25C]]

# Paths inside the extracted data bundles: setting RELEASE_DIR (main bundle) and
# SBIAD840_DIR (Princeton bundle) is enough; an explicit key always wins.
BUNDLE_LAYOUT = {
    "PROC_28C5": ("RELEASE_DIR", "processed_28C5"),
    "PROC_25C": ("RELEASE_DIR", "processed_25C"),
    "SPLIT_28C5": ("RELEASE_DIR", "splits/28C5.json"),
    "SPLIT_25C": ("RELEASE_DIR", "splits/25C.json"),
    **{key: ("RELEASE_DIR", f"checkpoints/{model}_best.pt") for model, key in CHECKPOINT_KEYS.items()},
    "PROC_28C5_SBIAD840": ("SBIAD840_DIR", "processed_28C5_sbiad840"),
    "PROC_25C_SBIAD840": ("SBIAD840_DIR", "processed_25C_sbiad840"),
    "SPLIT_28C5_SBIAD840": ("SBIAD840_DIR", "splits/28C5_sbiad840_test.json"),
    "SPLIT_25C_SBIAD840": ("SBIAD840_DIR", "splits/25C_sbiad840_test.json"),
}
BUNDLE_DEFAULTS = {"SBIAD840_DIR": "data/sbiad840_aligned_4p5"}


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (comments and blank lines ignored, surrounding quotes stripped)."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        value = os.environ.get(key) or self.values.get(key)
        if not value and key in BUNDLE_LAYOUT:
            root_key, rel = BUNDLE_LAYOUT[key]
            root = self.get(root_key, BUNDLE_DEFAULTS.get(root_key, ""))
            value = str(Path(root) / rel) if root else ""
        return value or default

    def path(self, key: str) -> Path | None:
        value = self.get(key)
        return Path(value).expanduser() if value else None

    # Runtime defaults (overridable per command).
    @property
    def runs_dir(self) -> Path:
        return Path(self.get("RUNS_DIR", "runs"))

    @property
    def dt_h(self) -> float:
        return float(self.get("DT_H", str(DT_H)))

    @property
    def t0_hpf(self) -> float:
        return float(self.get("T0_HPF", str(T0_HPF)))

    @property
    def clip_len(self) -> int:
        return int(self.get("CLIP_LEN", str(CLIP_LEN)))

    @property
    def stride(self) -> int:
        return int(self.get("STRIDE", str(STRIDE)))

    @property
    def device(self) -> str:
        return self.get("DEVICE", "auto")

    @property
    def amp(self) -> bool:
        return _truthy(self.get("AMP", "1"))

    @property
    def use_ema(self) -> bool:
        return _truthy(self.get("USE_EMA", "1"))

    def dataset(self, tag: str) -> tuple[Path, Path]:
        """(processed directory, split file) of a dataset tag."""
        if tag not in DATASET_KEYS:
            raise SystemExit(f"[ERR] unknown dataset {tag!r}; choose from {', '.join(DATASET_KEYS)}")
        proc_key, split_key = DATASET_KEYS[tag]
        proc, split = self.path(proc_key), self.path(split_key)
        if proc is None or split is None:
            raise SystemExit(f"[ERR] dataset {tag} needs {proc_key} and {split_key} in .env (see .env.example)")
        return proc, split

    def datasets(self, tags: list[str]) -> dict[str, tuple[Path, Path]]:
        return {tag: self.dataset(tag) for tag in tags}

    def checkpoint(self, model: str) -> Path:
        if model not in CHECKPOINT_KEYS:
            raise SystemExit(f"[ERR] unknown model {model!r}; choose from {', '.join(MODELS)}")
        ckpt = self.path(CHECKPOINT_KEYS[model])
        if ckpt is None:
            raise SystemExit(f"[ERR] model {model} needs {CHECKPOINT_KEYS[model]} in .env (see .env.example)")
        return ckpt

    def checkpoints(self, models: list[str]) -> dict[str, Path]:
        return {m: self.checkpoint(m) for m in models}


def load_settings(env_file: str | Path = ".env") -> Settings:
    return Settings(read_dotenv(env_file))


def check(settings: Settings, with_sbiad840: bool = False) -> None:
    """Verify that every configured path exists; exits with a message listing the problems."""
    keys = REQUIRED_KEYS + (SBIAD840_KEYS if with_sbiad840 else [])
    missing = [k for k in keys if not settings.get(k)]
    absent = [(k, settings.get(k)) for k in keys if settings.get(k) and not Path(settings.get(k)).expanduser().exists()]
    for k in keys:
        if k not in missing and all(k != a for a, _ in absent):
            print(f"[OK]  {k} = {settings.get(k)}")
    for k in missing:
        print(f"[ERR] {k} is not set")
    for k, v in absent:
        print(f"[ERR] {k} = {v} (path does not exist)")
    if missing or absent:
        raise SystemExit("Configuration incomplete: copy .env.example to .env and set the paths to the extracted data bundle.")
    print("[OK]  all paths found")
