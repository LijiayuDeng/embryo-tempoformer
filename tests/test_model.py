import json
from pathlib import Path

import pytest
import torch

from etf.config import VARIANTS, TrainConfig
from etf.model import build_model

LAYOUT = json.loads((Path(__file__).parent / "data" / "released_checkpoint_layout.json").read_text())


@pytest.mark.parametrize("variant", ["cnn_single", "full"])
def test_state_dict_matches_released_checkpoints(variant):
    """Module names and shapes must stay loadable with the released checkpoints (strict=True)."""
    model = build_model(TrainConfig.from_dict(LAYOUT[variant]["cfg"]))
    got = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert got == LAYOUT[variant]["state_dict"]


@pytest.mark.parametrize("variant,n_params", [("cnn_single", 275_761), ("meanpool", 275_761), ("nocons", 803_633), ("full", 803_633)])
def test_parameter_counts_match_paper(variant, n_params):
    # Paper Section 3.6: 0.276 M (cnn_single, meanpool) and 0.804 M (nocons, full).
    model = build_model(TrainConfig(**VARIANTS[variant]))
    assert sum(p.numel() for p in model.parameters()) == n_params


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_forward_shape(variant):
    torch.manual_seed(0)
    cfg = TrainConfig(model_dim=32, model_heads=2, cnn_base=8, **{**VARIANTS[variant], "model_depth": min(2, VARIANTS[variant]["model_depth"])})
    model = build_model(cfg).eval()
    x = torch.rand(3, 6, 1, 48, 48)
    with torch.no_grad():
        assert model(x).shape == (3,)


def test_frame_chunking_does_not_change_predictions():
    torch.manual_seed(0)
    x = torch.rand(2, 8, 1, 48, 48)
    outs = []
    for profile in ("fast", "lowmem", "ultra"):
        torch.manual_seed(1)
        model = build_model(TrainConfig(model_dim=32, model_heads=2, model_depth=1, cnn_base=8, mem_profile=profile)).eval()
        with torch.no_grad():
            outs.append(model(x))
    for out in outs[1:]:
        torch.testing.assert_close(out, outs[0], rtol=1e-5, atol=1e-5)
