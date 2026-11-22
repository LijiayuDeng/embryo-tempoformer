import json

import numpy as np
import pytest
import torch

from etf.config import TrainConfig
from etf.data import EXCLUDE_KEYS, PairQueueDataset, build_aug_params, is_excluded, well_key
from etf.inference import predict_embryo
from etf.model import build_model


def test_exclusion_rule():
    assert len(EXCLUDE_KEYS) == 39
    assert well_key("FishDev_WT_01_1_MMStack_A3-Site_0.ome") == "FishDev_WT_01_1-A3"
    assert is_excluded("FishDev_WT_01_1_MMStack_A3-Site_0.ome")
    assert not is_excluded("FishDev_WT_01_1_MMStack_A1-Site_0.ome")


def test_aug_groups():
    aug = build_aug_params("spatial,temporal")
    assert aug.p_hflip == aug.p_affine == aug.p_frame_drop == 0.0 and aug.frame_drop_max == 0
    assert aug.p_gamma > 0 and aug.p_noise > 0
    with pytest.raises(ValueError):
        build_aug_params("colour")


@pytest.fixture
def proc_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        np.save(tmp_path / f"e{i}.npy", rng.integers(0, 255, size=(40, 32, 32), dtype=np.uint8))
    return tmp_path


def test_pair_dataset(proc_dir):
    ds = PairQueueDataset(proc_dir, ["e0", "e1", "e2"], clip_len=8, augment=True, aug=build_aug_params(), seed=0, samples_per_embryo=2)
    assert len(ds) == 6
    item = ds[0]
    assert item["x1"].shape == (8, 1, 32, 32) and item["x1"].dtype == torch.uint8
    assert abs(float(item["offset1"]) - (4.5 + 0.25 * item["start1"])) < 1e-6
    assert 0 <= item["start1"] <= 32


def test_predict_embryo_windows(proc_dir):
    torch.manual_seed(0)
    model = build_model(TrainConfig(model_dim=32, model_heads=2, model_depth=1, cnn_base=8)).eval()
    frames = np.load(proc_dir / "e0.npy")
    out = predict_embryo(model, frames, device=torch.device("cpu"), clip_len=8, stride=8, amp=True)
    assert out["starts"] == [0, 8, 16, 24, 32]
    assert len(out["t0_hats"]) == 5 and np.isfinite(out["t0_final"])
    json.dumps(out)  # JSON-serialisable
