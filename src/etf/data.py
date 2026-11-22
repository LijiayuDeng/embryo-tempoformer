"""Datasets, clip augmentation, split files and the S-BIAD531 exclusion rule.

Processed embryos are uint8 arrays [T, H, W] stored as ``<proc_dir>/<eid>.npy``;
the embryo id (eid) is the file stem. Split files are JSON objects with
``train`` / ``val`` / ``test`` lists of embryo ids.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DT_H, T0_HPF
from .utils import read_json

try:  # torchvision is only needed for the affine augmentation
    import torchvision.transforms.functional as TVF
except ImportError:  # pragma: no cover
    TVF = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# S-BIAD531 well exclusions
# ---------------------------------------------------------------------------
# Wells of the 28.5 C plates that were excluded before splitting (non-viable
# embryos or embryos outside the field of view; paper Section 3.3 and Table 1).
# The released split files already encode all exclusions, including the single
# 25 C file excluded by name (FishDev_WT_25C_1_MMStack_G11-Site_0.ome); this list
# is applied when preprocessing raw S-BIAD531 stacks and when building new splits.
EXCLUDE_KEYS = frozenset({
    "FishDev_WT_01_1-A3", "FishDev_WT_01_1-B3", "FishDev_WT_01_1-C3", "FishDev_WT_01_1-D3",
    "FishDev_WT_01_1-E3", "FishDev_WT_01_1-F3", "FishDev_WT_01_1-G3", "FishDev_WT_01_1-H3",
    "FishDev_WT_02_1-A1", "FishDev_WT_02_1-B1", "FishDev_WT_02_1-C1", "FishDev_WT_02_1-D1",
    "FishDev_WT_02_1-E1", "FishDev_WT_02_1-F1", "FishDev_WT_02_1-G1", "FishDev_WT_02_1-H1",
    "FishDev_WT_03_1-A2", "FishDev_WT_03_1-B2", "FishDev_WT_03_1-C2", "FishDev_WT_03_1-D2",
    "FishDev_WT_03_1-E2", "FishDev_WT_03_1-F2", "FishDev_WT_03_1-G2", "FishDev_WT_03_1-H2",
    "FishDev_WT_04_1-A2", "FishDev_WT_04_1-B2", "FishDev_WT_04_1-C2", "FishDev_WT_04_1-D2",
    "FishDev_WT_04_1-E2", "FishDev_WT_04_1-F2", "FishDev_WT_04_1-G2",
    "FishDev_WT_05_1-A4", "FishDev_WT_05_1-B4", "FishDev_WT_05_1-C4", "FishDev_WT_05_1-D4",
    "FishDev_WT_05_1-E4", "FishDev_WT_05_1-F4", "FishDev_WT_05_1-G4", "FishDev_WT_05_1-H4",
})


def well_key(eid: str) -> str:
    """Plate-well key of an embryo id, e.g. FishDev_WT_01_1_MMStack_A1-Site_0.ome -> FishDev_WT_01_1-A1."""
    m = re.search(r"^(?P<prefix>.+?)_MMStack_(?P<well>[A-H]\d{1,2})\b", eid)
    if m is None:
        m = re.search(r"^(?P<prefix>.+?)_(?P<well>[A-H]\d{1,2})\b", eid)
    return f"{m.group('prefix')}-{m.group('well')}" if m else eid


def is_excluded(eid: str) -> bool:
    return eid in EXCLUDE_KEYS or well_key(eid) in EXCLUDE_KEYS


def filter_excluded(ids: Sequence[str]) -> list[str]:
    return [x for x in ids if not is_excluded(x)]


def split_ids(split_json: str | Path, key: str = "test", limit: int = 0) -> list[str]:
    """Embryo ids listed under ``key`` in a split file (optionally only the first ``limit``)."""
    ids = read_json(split_json).get(key)
    if not isinstance(ids, list):
        raise KeyError(f"Split key {key!r} not found or not a list in {split_json}")
    ids = [str(x) for x in ids]
    return ids[:limit] if limit > 0 else ids


def load_frames(proc_dir: Path, eid: str) -> np.ndarray:
    """Memory-map ``<proc_dir>/<eid>.npy`` (read-only)."""
    path = Path(proc_dir) / f"{eid}.npy"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugParams:
    """Clip-level augmentation; geometric transforms are shared by all frames of a clip."""

    p_hflip: float = 0.5

    p_affine: float = 0.5
    max_rotate_deg: float = 8.0
    max_translate: float = 0.03
    scale_min: float = 0.97
    scale_max: float = 1.03

    p_gamma: float = 0.5
    gamma_min: float = 0.8
    gamma_max: float = 1.15

    p_contrast: float = 0.5
    contrast_min: float = 0.8
    contrast_max: float = 1.15

    p_brightness: float = 0.5
    brightness_min: float = -0.07
    brightness_max: float = 0.07

    p_shade: float = 0.25
    shade_amp: float = 0.20

    p_noise: float = 0.5
    noise_sigma: float = 0.02

    p_blur: float = 0.2
    blur_ks: tuple[int, ...] = (3, 5)

    p_frame_drop: float = 0.10
    frame_drop_max: int = 2


# Augmentation families that can be switched off (paper Section 3.5 / Table 4).
# Start-index jitter belongs to the temporal/sampling family but is controlled by
# the dataset's ``jitter`` argument, not by AugParams.
AUG_GROUPS = {
    "spatial": ("p_hflip", "p_affine"),
    "photometric": ("p_gamma", "p_contrast", "p_brightness", "p_shade"),
    "acquisition": ("p_noise", "p_blur"),
    "temporal": ("p_frame_drop", "frame_drop_max"),
}


def parse_aug_groups(text: str) -> set[str]:
    groups = {tok.strip().lower() for tok in (text or "").split(",") if tok.strip()}
    unknown = sorted(groups - set(AUG_GROUPS))
    if unknown:
        raise ValueError(f"Unknown augmentation group(s): {', '.join(unknown)}")
    return groups


def build_aug_params(disable_groups: str = "") -> AugParams:
    """AugParams with the named families (comma-separated) disabled."""
    aug = AugParams()
    for group in parse_aug_groups(disable_groups):
        for field in AUG_GROUPS[group]:
            setattr(aug, field, 0 if field == "frame_drop_max" else 0.0)
    return aug


def aug_params_dict(disable_groups: str = "") -> dict[str, Any]:
    return dataclasses.asdict(build_aug_params(disable_groups))


def _box_blur_u8(frame_u8: np.ndarray, k: int) -> np.ndarray:
    """Box blur via an integral image."""
    if k <= 1:
        return frame_u8
    pad = k // 2
    x = np.pad(frame_u8.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    H2, W2 = x.shape
    ii = np.zeros((H2 + 1, W2 + 1), dtype=np.float32)
    ii[1:, 1:] = np.cumsum(np.cumsum(x, axis=0), axis=1)
    y = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / float(k * k)
    return np.clip(y, 0, 255).astype(np.uint8)


def augment_clip(clip_u8: np.ndarray, rng: np.random.Generator, p: AugParams) -> np.ndarray:
    """Apply mild stochastic augmentation to a uint8 clip [L, H, W]."""
    out = clip_u8.copy()
    L, H, W = out.shape

    # Temporal: replace a few frames with the previous frame.
    if rng.random() < p.p_frame_drop and p.frame_drop_max > 0:
        for _ in range(int(rng.integers(1, p.frame_drop_max + 1))):
            t = int(rng.integers(1, L))
            out[t] = out[t - 1]

    # Spatial: horizontal flip and one affine transform shared by all frames.
    if rng.random() < p.p_hflip:
        out = out[:, :, ::-1].copy()
    if rng.random() < p.p_affine and TVF is not None:
        angle = float(rng.uniform(-p.max_rotate_deg, p.max_rotate_deg))
        translate_px = [
            int(rng.uniform(-p.max_translate, p.max_translate) * W),
            int(rng.uniform(-p.max_translate, p.max_translate) * H),
        ]
        scale = float(rng.uniform(p.scale_min, p.scale_max))
        frames = []
        for t in range(L):
            img = torch.from_numpy(out[t]).unsqueeze(0).float() / 255.0
            img = TVF.affine(
                img,
                angle=angle,
                translate=translate_px,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TVF.InterpolationMode.BILINEAR,
                fill=0.0,
            )
            frames.append((img.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).squeeze(0).numpy())
        out = np.stack(frames, axis=0)

    # Photometric.
    xf = out.astype(np.float32) / 255.0
    if rng.random() < p.p_gamma:
        xf = np.clip(xf, 0.0, 1.0) ** float(rng.uniform(p.gamma_min, p.gamma_max))
    if rng.random() < p.p_contrast:
        c = float(rng.uniform(p.contrast_min, p.contrast_max))
        m = float(np.mean(xf))
        xf = (xf - m) * c + m
    if rng.random() < p.p_brightness:
        xf = xf + float(rng.uniform(p.brightness_min, p.brightness_max))
    if rng.random() < p.p_shade:  # multiplicative low-frequency illumination field
        grid = rng.normal(loc=0.0, scale=1.0, size=(8, 8)).astype(np.float32)
        grid = (grid - grid.mean()) / (grid.std() + 1e-6)
        shade = 1.0 + p.shade_amp * grid
        if Image is not None:
            im = Image.fromarray(((shade - shade.min()) / (shade.max() - shade.min() + 1e-6) * 255).astype(np.uint8))
            im = im.resize((W, H), resample=Image.BILINEAR)
            shade_u = 0.75 + 0.5 * (np.asarray(im, dtype=np.float32) / 255.0)
        else:  # pragma: no cover
            tile = np.ones((int(np.ceil(H / 8)), int(np.ceil(W / 8))), dtype=np.float32)
            shade_u = np.clip(np.kron(shade, tile)[:H, :W], 0.75, 1.25)
        xf = xf * shade_u[None, :, :]

    # Acquisition: additive noise and blur.
    if rng.random() < p.p_noise:
        xf = xf + rng.normal(loc=0.0, scale=p.noise_sigma, size=xf.shape).astype(np.float32)
    out = (np.clip(xf, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if rng.random() < p.p_blur:
        k = int(rng.choice(np.array(p.blur_ks)))
        for t in range(L):
            out[t] = _box_blur_u8(out[t], k=k)
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _LRUCache:
    """Tiny LRU cache of open memmaps."""

    def __init__(self, max_items: int = 16):
        self.max_items = int(max_items)
        self._data: dict[str, Any] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Any | None:
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        if len(self._order) > self.max_items:
            self._data.pop(self._order.pop(0), None)


def _clip_at(frames: np.ndarray, start: int, clip_len: int) -> np.ndarray:
    """Copy ``frames[start:start+clip_len]``, repeating the last frame if the stack is too short."""
    clip = np.array(frames[start:start + clip_len], dtype=np.uint8, copy=True)
    if clip.shape[0] != clip_len:
        clip = np.concatenate([clip, np.repeat(clip[-1:], clip_len - clip.shape[0], axis=0)], axis=0)
    return clip


def nominal_time(start: int | float) -> float:
    """Nominal developmental time (hpf) of a clip starting at frame ``start``."""
    return float(T0_HPF + DT_H * start)


class PairQueueDataset(Dataset):
    """Training pairs of clips from the same embryo.

    Each item holds two clips (uint8 [L, 1, H, W]) starting at s1 and s2 with
    their nominal times; the pair supports the within-embryo temporal-difference
    consistency loss. Start indices are drawn without replacement from a
    per-embryo queue and jittered by up to ``jitter`` frames.
    """

    def __init__(
        self,
        proc_dir: Path,
        embryo_ids: Sequence[str],
        clip_len: int,
        augment: bool,
        aug: AugParams,
        seed: int,
        samples_per_embryo: int = 32,
        jitter: int = 2,
        cache_items: int = 16,
    ):
        self.proc_dir = Path(proc_dir)
        self.embryo_ids = filter_excluded(list(embryo_ids))
        if not self.embryo_ids:
            raise ValueError("No embryo ids after filtering.")
        self.clip_len = int(clip_len)
        self.augment = bool(augment)
        self.aug = aug
        self.samples_per_embryo = int(samples_per_embryo)
        self.jitter = int(jitter)
        self._cache = _LRUCache(max_items=cache_items)
        self.reseed(seed)
        self._queues: dict[str, tuple[int, list[int]]] = {}

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return max(1, len(self.embryo_ids) * self.samples_per_embryo)

    def _load(self, eid: str) -> np.ndarray:
        arr = self._cache.get(eid)
        if arr is None:
            arr = load_frames(self.proc_dir, eid)
            self._cache.put(eid, arr)
        return arr

    def _pop_start(self, eid: str, max_start: int) -> int:
        item = self._queues.get(eid)
        if item is None or item[0] != max_start or not item[1]:
            q = list(range(0, max_start + 1))
            self.rng.shuffle(q)
            self._queues[eid] = (max_start, q)
        return int(self._queues[eid][1].pop())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        eid = self.embryo_ids[idx % len(self.embryo_ids)]
        frames = self._load(eid)
        max_start = max(0, int(frames.shape[0]) - self.clip_len)

        s1 = self._pop_start(eid, max_start)
        s2 = self._pop_start(eid, max_start)
        if self.jitter > 0 and max_start > 0:
            s1 = int(np.clip(s1 + int(self.rng.integers(-self.jitter, self.jitter + 1)), 0, max_start))
            s2 = int(np.clip(s2 + int(self.rng.integers(-self.jitter, self.jitter + 1)), 0, max_start))

        clip1 = _clip_at(frames, int(np.clip(s1, 0, max_start)), self.clip_len)
        clip2 = _clip_at(frames, int(np.clip(s2, 0, max_start)), self.clip_len)
        if self.augment:
            clip1 = augment_clip(clip1, self.rng, self.aug)
            clip2 = augment_clip(clip2, self.rng, self.aug)

        # Clips stay uint8 inside DataLoader workers; normalisation happens on the device.
        return {
            "eid": eid,
            "start1": int(s1),
            "start2": int(s2),
            "x1": torch.from_numpy(np.ascontiguousarray(clip1)).unsqueeze(1),
            "x2": torch.from_numpy(np.ascontiguousarray(clip2)).unsqueeze(1),
            "offset1": torch.tensor(nominal_time(s1), dtype=torch.float32),
            "offset2": torch.tensor(nominal_time(s2), dtype=torch.float32),
        }


class DeterministicValDataset(Dataset):
    """Every clip start of every embryo (clip-level validation).

    Clips from one embryo are strongly correlated: clip-level metrics track
    training but are not a valid unit for inference.
    """

    def __init__(self, proc_dir: Path, embryo_ids: Sequence[str], clip_len: int, cache_items: int = 16):
        self.proc_dir = Path(proc_dir)
        self.clip_len = int(clip_len)
        self.embryo_ids = filter_excluded(list(embryo_ids))
        self._cache = _LRUCache(max_items=cache_items)
        self.samples: list[tuple[str, int]] = []
        for eid in self.embryo_ids:
            n_frames = int(load_frames(self.proc_dir, eid).shape[0])
            self.samples.extend((eid, s) for s in range(0, max(0, n_frames - self.clip_len) + 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        eid, s = self.samples[idx]
        frames = self._cache.get(eid)
        if frames is None:
            frames = load_frames(self.proc_dir, eid)
            self._cache.put(eid, frames)
        s = int(np.clip(s, 0, max(0, int(frames.shape[0]) - self.clip_len)))
        clip = _clip_at(frames, s, self.clip_len)
        return {
            "eid": eid,
            "start": int(s),
            "x": torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(1),
            "target": torch.tensor(nominal_time(s), dtype=torch.float32),
        }
