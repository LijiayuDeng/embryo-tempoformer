import dataclasses
import json

import numpy as np
import pytest
import torch

from etf.ablation import train_settings
from etf.config import TrainConfig
from etf.inference import infer_all
from etf.model import build_model
from etf.tempo import aggregate_outroot


@pytest.fixture
def tiny_setup(tmp_path):
    torch.manual_seed(0)
    cfg = TrainConfig(model_dim=32, model_heads=2, model_depth=1, cnn_base=8)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": build_model(cfg).state_dict(), "ema": None, "cfg": dataclasses.asdict(cfg)}, ckpt)
    stacks = tmp_path / "condA"
    stacks.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        np.save(stacks / f"e{i}.npy", rng.integers(0, 255, size=(40, 32, 32), dtype=np.uint8))
    return ckpt, stacks, tmp_path / "out"


def _infer(ckpt, stacks, out, **kw):
    infer_all(outroot=out, datasets={"condA": (stacks, None)}, checkpoints={"m": ckpt}, device="cpu", log=lambda _: None, **kw)


def test_infer_settings_guard(tiny_setup):
    ckpt, stacks, out = tiny_setup
    _infer(ckpt, stacks, out, clip_len=8, stride=8)
    saved = json.loads((out / "condA" / "m" / "infer_settings.json").read_text())
    assert saved["clip_len"] == 8 and saved["checkpoint"] == "tiny.pt"
    with pytest.raises(SystemExit):  # same folder, different windowing
        _infer(ckpt, stacks, out, clip_len=4, stride=4)
    _infer(ckpt, stacks, out, clip_len=4, stride=4, force=True)
    assert len(json.loads((out / "condA" / "m" / "json" / "e0.json").read_text())["starts"]) == 10


def test_aggregate_redoes_stale_summaries(tiny_setup):
    ckpt, stacks, out = tiny_setup
    _infer(ckpt, stacks, out, clip_len=8, stride=8)
    aggregate_outroot(out, ["condA"], ["m"], log=lambda _: None)
    summary = out / "condA" / "m" / "summary.json"
    assert json.loads(summary.read_text())["global_metrics_points"]["n_embryos"] == 2
    np.save(stacks / "e2.npy", np.zeros((40, 32, 32), dtype=np.uint8))
    _infer(ckpt, stacks, out, clip_len=8, stride=8)
    aggregate_outroot(out, ["condA"], ["m"], log=lambda _: None)  # new embryo -> re-aggregated without --force
    assert json.loads(summary.read_text())["global_metrics_points"]["n_embryos"] == 3


def test_ablation_rejects_unknown_settings_before_training(tmp_path):
    with pytest.raises(SystemExit):
        train_settings(tmp_path / "abl", ["baseline_full", "no_colour"], proc_dir="p", split_json="s", dry_run=True)
    assert not (tmp_path / "abl").exists()
    train_settings(tmp_path / "abl", ["baseline_full", "no_temporal_sampling"], proc_dir="p", split_json="s", dry_run=True)
    rows = (tmp_path / "abl" / "manifest.tsv").read_text().splitlines()
    assert rows[0].startswith("setting") and [r.split("\t")[0] for r in rows[1:]] == ["baseline_full", "no_temporal_sampling"]
    cfg = json.loads((tmp_path / "abl" / "no_temporal_sampling" / "train_config.json").read_text())
    assert (cfg["aug_disable_groups"], cfg["jitter"], cfg["batch_size"]) == ("temporal", 0, 88)
