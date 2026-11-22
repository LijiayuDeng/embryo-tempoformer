import json
import subprocess
import sys

import pytest

from etf import cli
from etf.config import read_dotenv

COMMANDS = [
    ["check"], ["reproduce"], ["preprocess"], ["make-split"], ["train"], ["eval"], ["infer"], ["infer-all"], ["aggregate"],
    ["analyze", "stats"], ["analyze", "compare"], ["analyze", "power-continuous"], ["analyze", "cliplen"], ["analyze", "stage-error"],
    ["analyze", "anchor"], ["analyze", "stage-tempo"], ["figures"], ["saliency"],
    ["sbiad840", "preprocess"], ["sbiad840", "summarize"], ["sbiad840", "compare-kimmelnet"], ["sbiad840", "make-splits"],
    ["sbiad840", "finetune"], ["sbiad840", "eval-finetuned"], ["sbiad840", "summarize-finetune"], ["ablation"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_help(command):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([*command, "-h"])
    assert exc.value.code == 0


def test_help_does_not_import_torch():
    code = "import sys; from etf import cli; cli.build_parser(); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_train_config_resolution(tmp_path):
    parse = cli.build_parser().parse_args
    cfg = cli._train_config(parse(["train", "--proc_dir", "p", "--split_json", "s", "--out_dir", "o"]))
    assert (cfg.temporal_mode, cfg.lambda_diff, cfg.cnn_base, cfg.mem_profile) == ("transformer", 1.0, 32, "lowmem")

    cfg = cli._train_config(parse(["train", "--variant", "cnn_single", "--proc_dir", "p", "--split_json", "s", "--out_dir", "o", "--lr", "1e-3"]))
    assert (cfg.temporal_mode, cfg.model_depth, cfg.lambda_diff, cfg.lr) == ("identity", 0, 0.0, 1e-3)

    saved = tmp_path / "run_config.json"
    saved.write_text(json.dumps({"train_config": {"proc_dir": "p", "split_json": "s", "out_dir": "o", "epochs": 7, "unknown": 1}}))
    with pytest.raises(SystemExit):  # a new run from a saved configuration needs its own --out_dir
        cli._train_config(parse(["train", "--config", str(saved)]))
    cfg = cli._train_config(parse(["train", "--config", str(saved), "--out_dir", "o2", "--no-amp"]))
    assert (cfg.epochs, cfg.amp, cfg.proc_dir, cfg.out_dir) == (7, False, "p", "o2")
    cfg = cli._train_config(parse(["train", "--config", str(saved), "--resume", "o/ckpt/epoch3.pt"]))
    assert (cfg.out_dir, cfg.resume, cfg.init_ckpt) == ("o", "o/ckpt/epoch3.pt", "")

    with pytest.raises(SystemExit):
        cli._train_config(parse(["train"]))


def test_read_dotenv(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nPROC_28C5 = /data/a\nSPLIT_28C5=\"/data/s.json\"\nnot a pair\n")
    assert read_dotenv(env) == {"PROC_28C5": "/data/a", "SPLIT_28C5": "/data/s.json"}


def test_bundle_layout(monkeypatch):
    from etf.config import BUNDLE_LAYOUT, Settings

    for key in [*BUNDLE_LAYOUT, "RELEASE_DIR", "SBIAD840_DIR"]:
        monkeypatch.delenv(key, raising=False)
    s = Settings({"RELEASE_DIR": "/r"})
    assert s.get("PROC_28C5") == "/r/processed_28C5"
    assert s.get("CKPT_FULL") == "/r/checkpoints/full_best.pt"
    assert s.get("SPLIT_25C_SBIAD840") == "data/sbiad840_aligned_4p5/splits/25C_sbiad840_test.json"
    assert Settings({"RELEASE_DIR": "/r", "CKPT_FULL": "/x.pt"}).get("CKPT_FULL") == "/x.pt"
