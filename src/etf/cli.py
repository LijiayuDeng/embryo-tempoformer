"""Command-line interface: ``etf <command> [options]`` (also ``python -m etf``).

Heavy dependencies (torch, matplotlib) are imported inside the commands so that
``etf --help`` stays fast.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .constants import (
    CLIP_LEN,
    EXPECT_T,
    IMG_SIZE,
    MAIN_DATASETS,
    MEM_PROFILES,
    MODELS,
    SBIAD840_25C,
    SBIAD840_28C5,
    SBIAD840_DATASETS,
    STRIDE,
    TRIM,
)
from .utils import parse_int_list, parse_list

EPILOG = """\
paper reproduction:
  check            verify the data and checkpoint paths in .env
  reproduce        run the full analysis of the paper (about 15 min on one GPU)

model:
  preprocess       raw OME-TIFF stacks -> processed .npy arrays
  make-split       random embryo-level train/val/test split
  train            train or fine-tune a model (multi-GPU: torchrun ... -m etf train)
  eval             clip-level validation metrics of a checkpoint
  infer            embryo-level inference for one stack (.npy or .tif)

pipeline steps (run in order by `etf reproduce`):
  infer-all        inference for every test embryo of each dataset x model
  aggregate        per-embryo tempo readouts (points.csv, embryo.csv, summary.json)
  analyze          statistics and sensitivity analyses
  figures          paper figures and tables
  saliency         SmoothGrad saliency (qualitative)

extended analyses:
  sbiad840         Princeton S-BIAD840 cross-site validation and low-shot fine-tuning
  ablation         augmentation-family ablation (Table 4)

Run `etf <command> -h` for the options of a command.
"""


# ---------------------------------------------------------------------------
# Option helpers
# ---------------------------------------------------------------------------
def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _add_outroot(p: argparse.ArgumentParser, required: bool = True, help: str = "evaluation root directory") -> None:
    p.add_argument("--outroot", required=required, type=Path, help=help)


def _add_datasets(p: argparse.ArgumentParser, default: tuple[str, ...] = MAIN_DATASETS) -> None:
    p.add_argument("--datasets", type=parse_list, default=list(default), help=f"comma-separated dataset tags (default: {','.join(default)})")


def _add_models(p: argparse.ArgumentParser, default: tuple[str, ...] = MODELS) -> None:
    p.add_argument("--models", type=parse_list, default=list(default), help=f"comma-separated model tags (default: {','.join(default)})")


def _add_runtime(p: argparse.ArgumentParser, amp_default: bool | None = None) -> None:
    p.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu (default: DEVICE in .env, else auto)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=amp_default, help="mixed precision on CUDA (default: AMP in .env, else on)")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None, help="use EMA weights (default: USE_EMA in .env, else on)")


def _runtime(args: argparse.Namespace, settings) -> dict[str, Any]:
    return dict(device=args.device or settings.device, amp=settings.amp if args.amp is None else args.amp,
                use_ema=settings.use_ema if args.use_ema is None else args.use_ema)


def _or(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_check(args, settings) -> None:
    from .config import check

    check(settings, with_sbiad840=args.with_sbiad840)


def cmd_reproduce(args, settings) -> None:
    from .pipeline import reproduce

    reproduce(settings, args.outroot, skip=args.skip, saliency=args.saliency, saliency_model=args.saliency_model, force=args.force)


def cmd_preprocess(args, settings) -> None:
    from .preprocess import preprocess_tiff_dir

    preprocess_tiff_dir(args.in_dir, args.proc_dir, expect_t=args.expect_t, img_size=args.img_size, p_lo=args.p_lo, p_hi=args.p_hi,
                        max_pages=args.max_pages, limit=args.limit, pad_to_expect=args.pad_to_expect)


def cmd_make_split(args, settings) -> None:
    from .preprocess import make_split

    make_split(args.proc_dir, args.out_json, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)


TRAIN_HELP = {
    "proc_dir": "directory with processed <eid>.npy arrays",
    "split_json": "split file (the train and val lists are used)",
    "out_dir": "output directory (best.pt, last.pt, history.csv, run_config.json)",
    "jitter": "random shift of clip starts (frames)",
    "aug_disable_groups": "augmentation families to switch off: spatial,photometric,acquisition,temporal",
    "temporal_mode": "transformer | meanpool | identity",
    "mem_profile": "fast | balanced | lowmem | ultra (activation-memory/compute trade-off)",
    "lambda_diff": "weight of the temporal-difference consistency loss",
    "cons_ramp_ratio": "fraction of steps over which the consistency weight ramps up",
    "resume": "resume training state from a checkpoint",
    "init_ckpt": "initialise weights from a checkpoint (fresh optimiser; for fine-tuning)",
    "init_use_ema": "with --init_ckpt, load the EMA weights",
    "unfreeze_frame_tail_blocks": "after freezing the frame encoder, train its projection and last N CNN blocks",
    "unfreeze_temporal_tail_blocks": "after freezing the temporal module, train its last N blocks and the CLS/norm readout",
    "save_every": "save a training checkpoint every N epochs (0 = only best.pt and last.pt)",
    "patience": "early stopping after N validations without improvement (0 = off)",
    "val_every": "validate every N epochs (always on the last epoch)",
}


def _add_train_args(p: argparse.ArgumentParser) -> None:
    from .config import VARIANTS, TrainConfig
    from .constants import MEM_PROFILES, TEMPORAL_MODES

    p.add_argument("--config", type=Path, help="start from a saved configuration: run_config.json, a TrainConfig JSON or a checkpoint (.pt)")
    p.add_argument("--variant", choices=list(VARIANTS), help="paper model variant (default: full)")
    choices = {"temporal_mode": list(TEMPORAL_MODES), "mem_profile": list(MEM_PROFILES), "abs_loss_type": ["l1", "smoothl1"]}
    for f in dataclasses.fields(TrainConfig):
        help_ = TRAIN_HELP.get(f.name, f.name.replace("_", " "))
        if isinstance(f.default, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=None, help=f"{help_} (default: {f.default})")
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=None, choices=choices.get(f.name),
                           help=f"{help_} (default: {f.default!r})" if f.default != "" else help_)


def _train_config(args: argparse.Namespace):
    from .config import VARIANTS, TrainConfig

    values = dataclasses.asdict(TrainConfig())
    if args.config:
        if args.config.suffix == ".pt":
            import torch

            loaded = torch.load(args.config, map_location="cpu", weights_only=False)["cfg"]
        else:
            loaded = json.loads(args.config.read_text(encoding="utf-8"))
            loaded = loaded.get("train_config", loaded)
        values.update({k: v for k, v in loaded.items() if k in values})
        values["resume"] = ""  # a saved configuration never resumes by itself
        if not args.resume and args.out_dir is None:
            raise SystemExit("[ERR] give --out_dir for the new run (or --resume CHECKPOINT to continue the saved run in place)")
    if args.variant:
        values.update(VARIANTS[args.variant])
    if args.resume and args.init_ckpt is None:
        values["init_ckpt"] = ""  # resuming restores the weights; a saved init_ckpt would conflict
    values.update({k: getattr(args, k) for k in values if getattr(args, k, None) is not None})
    cfg = TrainConfig(**values)
    missing = [k for k in ("proc_dir", "split_json", "out_dir") if not getattr(cfg, k)]
    if missing:
        raise SystemExit(f"[ERR] missing {', '.join('--' + k for k in missing)}")
    return cfg


def cmd_train(args, settings) -> None:
    from .train import train

    train(_train_config(args))


def cmd_eval(args, settings) -> None:
    from .train import evaluate

    rt = _runtime(args, settings)
    metrics = evaluate(str(args.ckpt), args.proc_dir, args.split_json, clip_len=_or(args.clip_len, settings.clip_len),
                       batch_size=args.batch_size, num_workers=args.num_workers, mem_profile=args.mem_profile, **rt)
    print(json.dumps(metrics, indent=2))


def cmd_infer(args, settings) -> None:
    from .inference import infer_file, load_model
    from .train import resolve_device

    rt = _runtime(args, settings)
    ckpt = args.ckpt or settings.checkpoint(args.model)
    dev = resolve_device(rt["device"])
    model, _ = load_model(ckpt, device=dev, use_ema=rt["use_ema"], mem_profile=args.mem_profile)
    out = infer_file(model, args.input_path, args.out_json, device=dev, clip_len=_or(args.clip_len, settings.clip_len),
                     stride=_or(args.stride, settings.stride), trim=args.trim, amp=rt["amp"], img_size=args.img_size)
    print(f"[infer] {args.out_json}: t0_final = {out['t0_final']:.3f} hpf over {len(out['starts'])} windows")


def cmd_infer_all(args, settings) -> None:
    from .inference import infer_all

    if args.ckpt and len(args.models) != 1:
        raise SystemExit("[ERR] --ckpt needs exactly one --models entry")
    checkpoints = {args.models[0]: args.ckpt} if args.ckpt else settings.checkpoints(args.models)
    if args.input_dir:
        if args.datasets:
            raise SystemExit("[ERR] use either --input_dir or --datasets")
        names = [d.resolve().name for d in args.input_dir]
        if len(set(names)) != len(names):
            raise SystemExit(f"[ERR] --input_dir folders need distinct names (outputs are named after them): {names}")
        datasets = {name: (d, None) for name, d in zip(names, args.input_dir)}
    else:
        datasets = settings.datasets(args.datasets or list(MAIN_DATASETS))
    outroot = args.outroot or settings.runs_dir / f"eval_{_stamp()}"
    print(f"[infer-all] outroot = {outroot}")
    infer_all(outroot=outroot, datasets=datasets, checkpoints=checkpoints,
              clip_len=_or(args.clip_len, settings.clip_len), stride=_or(args.stride, settings.stride), force=args.force,
              max_eids=args.max_eids, **_runtime(args, settings))


def cmd_aggregate(args, settings) -> None:
    from .tempo import aggregate_json_dir, aggregate_outroot

    dt, t0 = _or(args.dt, settings.dt_h), _or(args.t0, settings.t0_hpf)
    if args.json_dir:
        aggregate_json_dir(args.json_dir, args.out_dir or args.json_dir.parent, dt=dt, t0=t0)
    elif args.outroot:
        aggregate_outroot(args.outroot, args.datasets, args.models, dt=dt, t0=t0, force=args.force)
    else:
        raise SystemExit("[ERR] give --outroot (all datasets x models) or --json_dir")


def cmd_analyze_stats(args, settings) -> None:
    from .stats import run_stats

    run_stats(args.outroot, models=args.models, a_tag=args.a_test, b_tag=args.b_test, label_a=args.label_a, label_b=args.label_b,
              m_col=args.m_col, ci_boot=args.ci_boot, power_R=args.power_R, power_boot=args.power_boot, seed=args.seed)


def cmd_analyze_compare(args, settings) -> None:
    from .stats import ci_delta_m, power_curve

    out = args.out_dir
    ci_delta_m(args.csv_a, args.csv_b, out / f"CI_{args.m_col}.json", m_col=args.m_col, n_boot=args.ci_boot, seed=args.seed,
               label_a=args.label_a, label_b=args.label_b)
    if args.power:
        E_list = parse_int_list(args.E_list) if args.E_list else None
        power_curve(args.csv_a, args.csv_b, out / f"power_{args.m_col}.csv", out / f"power_{args.m_col}.png", m_col=args.m_col,
                    E_list=E_list, R=args.power_R, n_boot=args.power_boot, seed=args.seed, replacement=args.replacement,
                    label_a=args.label_a, label_b=args.label_b)


def cmd_analyze_power_continuous(args, settings) -> None:
    from .stats import power_continuous

    power_continuous(args.outroot, args.out_dir, models=args.models, a_tag=args.a_test, b_tag=args.b_test, m_col=args.m_col,
                     E_list=args.E_list, delta_max=args.delta_max, delta_step=args.delta_step, R=args.R, B_boot=args.B_boot,
                     seed=args.seed, y_min=args.y_min_plot, y_max=args.y_max_plot, y_tick_step=args.y_tick_step,
                     monotone_power=args.monotone_power, monotone_threshold=args.monotone_threshold)


def cmd_analyze_cliplen(args, settings) -> None:
    from .analyses import cliplen_sensitivity, context_fit

    out_dir = args.out_dir or args.outroot / "cliplen_sensitivity"
    csv_path = cliplen_sensitivity(out_dir, datasets=settings.datasets(args.datasets), checkpoints=settings.checkpoints(args.models),
                                   clip_lens=[int(x) for x in args.clip_lens], stride=_or(args.stride, settings.stride),
                                   dt=settings.dt_h, t0=settings.t0_hpf, max_eids=args.max_eids, force=args.force, **_runtime(args, settings))
    if "full" in args.models and all((args.outroot / ds / "cnn_single" / "summary.json").exists() for ds in args.datasets):
        context_fit(args.outroot, csv_path, out_dir / "context_fit", clip_lens=[int(x) for x in args.clip_lens], dt=settings.dt_h)
    else:
        print("[cliplen] context ladder skipped (needs model 'full' and cnn_single summaries under --outroot)")


def cmd_analyze_stage_error(args, settings) -> None:
    from .analyses import stage_error_bins

    stage_error_bins(args.outroot, args.out_csv, datasets=args.datasets, models=args.models, scheme=args.scheme, bins=args.bins)


def cmd_analyze_anchor(args, settings) -> None:
    from .analyses import anchor_sensitivity

    anchor_sensitivity(args.outroot, args.out_dir, datasets=args.datasets, models=args.models, t0_list=args.t0_list, dt=settings.dt_h)


def cmd_analyze_stage_tempo(args, settings) -> None:
    from .analyses import stage_tempo

    stage_tempo(args.outroot, args.out_dir, datasets=args.datasets, models=args.models, n_boot=args.n_boot, seed=args.seed)


def cmd_figures(args, settings) -> None:
    from .figures import make_paper_figures, paper_tables

    make_paper_figures(args.outroot, args.out_dir, dpi=args.dpi)
    paper_tables(args.outroot)


def cmd_saliency(args, settings) -> None:
    from . import saliency

    rt = _runtime(args, settings)
    opts = dict(clip_len=_or(args.clip_len, settings.clip_len), n_samples=args.sg_N, sigma=args.sg_sigma, hm_lo=args.hm_lo,
                hm_hi=args.hm_hi, hm_gamma=args.hm_gamma, blur_k=args.blur_k, alpha_thr=args.alpha_thr, alpha_max=args.alpha_max,
                use_ema=rt["use_ema"], amp=bool(args.amp), device=rt["device"], seed=args.seed, dpi=args.dpi)
    opts["starts"] = [int(s) for s in parse_list(args.starts)] if args.starts else None
    if args.npy:
        if not (args.ckpt and args.out_dir):
            raise SystemExit("[ERR] --npy needs --ckpt and --out_dir")
        saliency.run_saliency(args.ckpt, args.npy, args.out_dir, **opts)
    elif args.outroot:
        from .constants import ID28C5

        saliency.best_embryo_saliency(args.outroot, args.model, ckpt=args.ckpt or settings.checkpoint(args.model),
                                      proc_dir=settings.dataset(ID28C5)[0], **opts)
    else:
        raise SystemExit("[ERR] give --outroot (best 28.5 C test embryo) or --npy/--ckpt/--out_dir")


def cmd_sbiad840_preprocess(args, settings) -> None:
    from .sbiad840 import preprocess_png

    src = args.src_root or settings.path("SBIAD840_SRC_ROOT")
    if not src:
        raise SystemExit("[ERR] give --src_root (the S-BIAD840 Princeton_Data directory) or set SBIAD840_SRC_ROOT")
    out = args.out_root or settings.path("SBIAD840_OUT_ROOT") or Path(settings.get("SBIAD840_DIR", "data/sbiad840_aligned_4p5"))
    preprocess_png(src, out, align_start_hpf=args.align_start_hpf, expect_t=args.expect_t, img_size=args.img_size, p_lo=args.p_lo,
                   p_hi=args.p_hi, pad_to_expect=args.pad_to_expect, limit=args.limit)


def cmd_sbiad840_summarize(args, settings) -> None:
    from .sbiad840 import summarize_external

    summarize_external(args.outroot, datasets=args.datasets, models=args.models)


def cmd_sbiad840_compare(args, settings) -> None:
    from .sbiad840 import compare_kimmelnet

    compare_kimmelnet(args.base_outroot / "sbiad840_external_summary.csv", args.dense_outroot / "sbiad840_external_summary.csv",
                      args.base_outroot / "sbiad840_vs_kimmelnet.csv", args.base_outroot / "sbiad840_vs_kimmelnet.md")


def _default_finetune_split(settings, n_train: int = 12) -> Path:
    _, split = settings.dataset(SBIAD840_28C5)
    return split.parent / "finetune" / f"28C5_sbiad840_ft{n_train}_v12_seed42.json"


def cmd_sbiad840_make_splits(args, settings) -> None:
    from .sbiad840 import make_finetune_splits

    source = args.source_split or settings.dataset(SBIAD840_28C5)[1]
    make_finetune_splits(source, args.out_dir or Path(source).parent / "finetune", train_counts=args.train_counts,
                         val_count=args.val_count, seed=args.seed, prefix=args.prefix)


def cmd_sbiad840_finetune(args, settings) -> None:
    from .sbiad840 import finetune_config
    from .train import train

    cfg = finetune_config(
        args.model, args.stage, init_ckpt=args.init_ckpt or settings.checkpoint(args.model),
        split_json=str(args.split_json or _default_finetune_split(settings)),
        proc_dir=str(args.proc_dir or settings.dataset(SBIAD840_28C5)[0]),
        out_dir=str(args.out_dir or settings.runs_dir / f"finetune_{args.model}_{args.stage}_{_stamp()}"),
        epochs=args.epochs, num_workers=args.num_workers, batch_size=args.batch_size, val_batch_size=args.val_batch_size,
        samples_per_embryo=args.samples_per_embryo, jitter=args.jitter, cache_items=args.cache_items, grad_accum=args.grad_accum,
        patience=args.patience, save_every=args.save_every, seed=args.seed, device=args.device or settings.device, lr=args.lr,
        mem_profile=args.mem_profile, ema_decay=args.ema_decay, ema_start_ratio=args.ema_start_ratio,
        amp=settings.amp if args.amp is None else args.amp,
        init_use_ema=args.init_use_ema, ema_eval=args.ema_eval,
    )
    print(json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True))
    if not args.dry_run:
        train(cfg)


def cmd_sbiad840_eval_finetuned(args, settings) -> None:
    from .sbiad840 import eval_finetuned

    datasets = settings.datasets(args.datasets)
    for tag, split in ((SBIAD840_28C5, args.split_28c5), (SBIAD840_25C, args.split_25c)):
        if split and tag in datasets:
            datasets[tag] = (datasets[tag][0], split)
    eval_finetuned(args.ft_ckpt, args.model, args.outroot, datasets=datasets, clip_len=_or(args.clip_len, settings.clip_len),
                   stride=_or(args.stride, settings.stride), dt=settings.dt_h, t0=settings.t0_hpf, force=args.force,
                   **_runtime(args, settings))


def cmd_sbiad840_summarize_finetune(args, settings) -> None:
    from .sbiad840 import summarize_finetune

    summarize_finetune(args.experiment, args.out_dir / "sbiad840_finetune_compare.csv", args.out_dir / "sbiad840_finetune_compare.md")


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """KEY=VALUE strings -> typed TrainConfig overrides."""
    from .config import TrainConfig

    defaults = dataclasses.asdict(TrainConfig())
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in defaults:
            raise SystemExit(f"[ERR] --set expects KEY=VALUE with a training option name, got {item!r}")
        kind = type(defaults[key])
        out[key] = value.strip().lower() in {"1", "true", "yes", "on"} if kind is bool else kind(value)
    return out


def cmd_ablation(args, settings) -> None:
    from .ablation import SETTINGS, evaluate_settings, summarize_ablation, train_settings
    from .constants import ID28C5

    steps = set(args.steps)
    unknown = steps - {"train", "evaluate", "summarize"}
    if unknown:
        raise SystemExit(f"[ERR] unknown step(s): {', '.join(sorted(unknown))}")
    settings_list = args.settings or list(SETTINGS)
    outroot = args.outroot or settings.runs_dir / f"min_aug_ablation_{_stamp()}"
    if "train" in steps:
        proc_dir, split_json = settings.dataset(ID28C5)
        overrides = {k: v for k, v in dict(batch_size=args.batch_size, val_batch_size=args.val_batch_size, num_workers=args.num_workers,
                                           epochs=args.epochs, patience=args.patience, save_every=args.save_every, seed=args.seed,
                                           mem_profile=args.mem_profile, device=args.device, amp=args.amp).items() if v is not None}
        overrides.update(_parse_overrides(args.set))
        train_settings(outroot, settings_list, proc_dir=proc_dir, split_json=split_json, overrides=overrides, dry_run=args.dry_run)
    if "evaluate" in steps and not args.dry_run:
        evaluate_settings(outroot, settings_list, datasets=settings.datasets(list(MAIN_DATASETS)), dt=settings.dt_h, t0=settings.t0_hpf,
                          clip_len=settings.clip_len, stride=settings.stride, **_runtime(args, settings))
    if "summarize" in steps and not args.dry_run:
        summarize_ablation(outroot, settings_list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etf", description="EmbryoTempoFormer: developmental tempo from zebrafish time-lapse microscopy.",
                                     epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"etf {__version__}")
    parser.add_argument("--env_file", default=".env", help="file with data and checkpoint paths (default: ./.env)")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True, help="one of the commands listed below")

    def command(name: str, func: Callable | None, help: str, parent=sub, aliases: tuple[str, ...] = ()) -> argparse.ArgumentParser:
        listed = {} if parent is sub else {"help": help}  # top-level commands are listed (grouped) in EPILOG
        p = parent.add_parser(name, description=help, aliases=list(aliases), **listed)
        if func is not None:
            p.set_defaults(func=func)
        return p

    p = command("check", cmd_check, "verify the data and checkpoint paths configured in .env")
    p.add_argument("--with_sbiad840", action="store_true", help="also check the S-BIAD840 paths")

    p = command("reproduce", cmd_reproduce, "run the complete analysis of the paper")
    _add_outroot(p, required=False, help="output root (default: <RUNS_DIR>/paper_eval_<timestamp>)")
    from .pipeline import STEPS

    p.add_argument("--skip", type=parse_list, default=[], help=f"steps to skip: {','.join(STEPS)}")
    p.add_argument("--saliency", action="store_true", help="also compute SmoothGrad saliency (qualitative; slower)")
    p.add_argument("--saliency_model", default="full", choices=MODELS)
    p.add_argument("--force", action="store_true", help="recompute inference and aggregation even if outputs exist")

    p = command("preprocess", cmd_preprocess, "convert raw OME-TIFF stacks to processed .npy arrays")
    p.add_argument("--in_dir", required=True, type=Path, help="directory containing *.tif* files (not recursive)")
    p.add_argument("--proc_dir", required=True, type=Path, help="output directory")
    p.add_argument("--expect_t", type=int, default=EXPECT_T, help="frames after padding/trimming")
    p.add_argument("--pad_to_expect", action=argparse.BooleanOptionalAction, default=True,
                   help="pad (repeat the last frame) or trim to --expect_t, as for the released arrays; "
                        "--no-pad_to_expect keeps the recorded length (recommended for new recordings)")
    p.add_argument("--img_size", type=int, default=IMG_SIZE, help="output frame size")
    p.add_argument("--p_lo", type=float, default=1.0, help="lower intensity percentile")
    p.add_argument("--p_hi", type=float, default=99.0, help="upper intensity percentile")
    p.add_argument("--max_pages", type=int, default=0, help="read at most N TIFF pages (0 = all)")
    p.add_argument("--limit", type=int, default=0, help="process at most N embryos (0 = all)")

    p = command("make-split", cmd_make_split, "random embryo-level train/val/test split of a processed directory", aliases=("make_split",))
    p.add_argument("--proc_dir", required=True, type=Path)
    p.add_argument("--out_json", required=True, type=Path)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p = command("train", cmd_train, "train a model (defaults = released ETF-full configuration) or fine-tune with --init_ckpt")
    p.usage = "etf train [--config FILE] [--variant NAME] --proc_dir DIR --split_json FILE --out_dir DIR [options]"
    _add_train_args(p)

    p = command("eval", cmd_eval, "clip-level validation metrics of a checkpoint (descriptive)")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--proc_dir", required=True, type=Path)
    p.add_argument("--split_json", required=True, type=Path)
    p.add_argument("--clip_len", type=int, default=None, help=f"default: CLIP_LEN in .env, else {CLIP_LEN}")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mem_profile", default="lowmem", choices=list(MEM_PROFILES))
    _add_runtime(p)

    p = command("infer", cmd_infer, "embryo-level sliding-window inference for one stack (.npy or .tif)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="checkpoint file")
    src.add_argument("--model", choices=MODELS, help="released checkpoint configured in .env")
    p.add_argument("--input_path", required=True, type=Path)
    p.add_argument("--out_json", required=True, type=Path)
    p.add_argument("--clip_len", type=int, default=None, help=f"default: CLIP_LEN in .env, else {CLIP_LEN}")
    p.add_argument("--stride", type=int, default=None, help=f"default: STRIDE in .env, else {STRIDE}")
    p.add_argument("--trim", type=float, default=TRIM, help="trimmed-mean proportion for t0_final")
    p.add_argument("--img_size", type=int, default=IMG_SIZE, help="frame size when preprocessing a TIFF (its length is kept)")
    p.add_argument("--mem_profile", default="lowmem", choices=list(MEM_PROFILES), help="frames per CNN call (released results: lowmem)")
    _add_runtime(p)

    p = command("infer-all", cmd_infer_all, "inference for every test embryo of each dataset x model")
    _add_outroot(p, required=False, help="output root (default: <RUNS_DIR>/eval_<timestamp>)")
    p.add_argument("--datasets", type=parse_list, default=None, help=f"comma-separated dataset tags (default: {','.join(MAIN_DATASETS)})")
    _add_models(p)
    p.add_argument("--input_dir", type=Path, action="append", default=[],
                   help="your own directory of .npy/.tif stacks (all files; repeatable; output is named after the directory)")
    p.add_argument("--ckpt", type=Path, help="use this checkpoint instead of the released one (single model)")
    p.add_argument("--clip_len", type=int, default=None, help=f"default: CLIP_LEN in .env, else {CLIP_LEN}")
    p.add_argument("--stride", type=int, default=None, help=f"default: STRIDE in .env, else {STRIDE}")
    p.add_argument("--max_eids", type=int, default=0, help="first N embryos per dataset only (smoke tests)")
    p.add_argument("--force", action="store_true", help="overwrite existing per-embryo JSON")
    _add_runtime(p)

    p = command("aggregate", cmd_aggregate, "per-embryo tempo readouts (points.csv, embryo.csv, summary.json)")
    _add_outroot(p, required=False)
    _add_datasets(p)
    _add_models(p)
    p.add_argument("--json_dir", type=Path, help="aggregate a single directory of inference JSON instead")
    p.add_argument("--out_dir", type=Path, help="output directory for --json_dir (default: its parent)")
    p.add_argument("--dt", type=float, default=None, help="sampling interval in hours (default: DT_H in .env, else 0.25)")
    p.add_argument("--t0", type=float, default=None, help="anchor / first-frame time in hpf (default: T0_HPF in .env, else 4.5)")
    p.add_argument("--force", action="store_true", help="re-aggregate even if summary.json exists")

    analyze = command("analyze", None, "statistics and sensitivity analyses")
    asub = analyze.add_subparsers(dest="analysis", metavar="<analysis>", required=True)

    p = command("stats", cmd_analyze_stats, "Delta m with embryo-bootstrap CI and power curves (Table 6)", parent=asub)
    _add_outroot(p)
    _add_models(p)
    p.add_argument("--a_test", default="EXT25C_TEST", help="condition A dataset (Delta m = A - B)")
    p.add_argument("--b_test", default="ID28C5_TEST", help="condition B dataset")
    p.add_argument("--label_a", default="25C")
    p.add_argument("--label_b", default="28C5")
    p.add_argument("--m_col", default="m_anchor")
    p.add_argument("--ci_boot", type=int, default=5000, help="bootstrap replicates for the CI")
    p.add_argument("--power_R", type=int, default=500, help="simulated experiments per embryo budget")
    p.add_argument("--power_boot", type=int, default=500, help="bootstrap replicates per simulated experiment")
    p.add_argument("--seed", type=int, default=0)

    p = command("compare", cmd_analyze_compare, "Delta m (A - B) with embryo-bootstrap CI and power curve for any two embryo.csv files",
                parent=asub)
    p.add_argument("--csv_a", required=True, type=Path, help="embryo.csv of condition A")
    p.add_argument("--csv_b", required=True, type=Path, help="embryo.csv of condition B")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--m_col", default="m_anchor")
    p.add_argument("--ci_boot", type=int, default=5000)
    p.add_argument("--power", action=argparse.BooleanOptionalAction, default=True, help="also compute the power curve")
    p.add_argument("--E_list", default="", help="embryo budgets, e.g. 2,4,6 (default: 2, 3 and even numbers up to min(30, smaller group))")
    p.add_argument("--power_R", type=int, default=500)
    p.add_argument("--power_boot", type=int, default=500)
    p.add_argument("--replacement", action="store_true", help="draw embryos with replacement in simulated experiments")
    p.add_argument("--seed", type=int, default=0)

    p = command("power-continuous", cmd_analyze_power_continuous, "E80/E90/E95 planning curves over |Delta m| (Fig. 4B)", parent=asub)
    _add_outroot(p)
    p.add_argument("--out_dir", type=Path, help="default: <outroot>/continuous_power")
    _add_models(p)
    p.add_argument("--a_test", default="EXT25C_TEST")
    p.add_argument("--b_test", default="ID28C5_TEST")
    p.add_argument("--m_col", default="m_anchor")
    p.add_argument("--E_list", default="2,3,4,6,8,10,12,14,16,18,20,22", help="embryo budgets per group")
    p.add_argument("--delta_max", type=float, default=0.10)
    p.add_argument("--delta_step", type=float, default=0.002)
    p.add_argument("--R", type=int, default=1200, help="simulated experiments per grid point")
    p.add_argument("--B_boot", type=int, default=800, help="bootstrap replicates per experiment")
    p.add_argument("--seed", type=int, default=20260310)
    p.add_argument("--y_min_plot", type=float, default=0.0)
    p.add_argument("--y_max_plot", type=float, default=23.0)
    p.add_argument("--y_tick_step", type=float, default=1.0)
    p.add_argument("--monotone_power", action=argparse.BooleanOptionalAction, default=True, help="running max of power over E")
    p.add_argument("--monotone_threshold", action=argparse.BooleanOptionalAction, default=True, help="running min of E over |Delta m|")

    p = command("cliplen", cmd_analyze_cliplen, "clip-length sensitivity (L = 4/12/24) and the 0/1/3/6 h context ladder", parent=asub)
    _add_outroot(p, help="main evaluation root (cnn_single summaries are used for the 0 h reference)")
    p.add_argument("--out_dir", type=Path, help="default: <outroot>/cliplen_sensitivity")
    _add_datasets(p)
    _add_models(p, default=("full",))
    p.add_argument("--clip_lens", type=parse_list, default=["4", "12", "24"])
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--max_eids", type=int, default=0)
    p.add_argument("--force", action="store_true")
    _add_runtime(p)

    p = command("stage-error", cmd_analyze_stage_error, "window-level error by Kimmel period (descriptive)", parent=asub)
    _add_outroot(p)
    p.add_argument("--out_csv", type=Path, help="default: <outroot>/stage_error/stage_error_by_bin.csv")
    _add_datasets(p)
    _add_models(p, default=("cnn_single", "full"))
    p.add_argument("--scheme", choices=["kimmel_start", "fixed"], default="kimmel_start")
    p.add_argument("--bins", default="", help="bin edges in hpf for --scheme fixed, e.g. 4.5,12.5,20.5,28.5,36.5,46.5")

    p = command("anchor", cmd_analyze_anchor, "re-aggregation under alternative anchors T0", parent=asub)
    _add_outroot(p)
    p.add_argument("--out_dir", type=Path, help="default: <outroot>/anchor_sensitivity")
    _add_datasets(p)
    _add_models(p)
    p.add_argument("--t0_list", default="4.0,4.5,5.0")

    p = command("stage-tempo", cmd_analyze_stage_tempo, "piecewise tempo slopes per Kimmel period with embryo-bootstrap CIs", parent=asub)
    _add_outroot(p)
    p.add_argument("--out_dir", type=Path, help="default: <outroot>/stage_tempo")
    _add_datasets(p)
    _add_models(p)
    p.add_argument("--n_boot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)

    p = command("figures", cmd_figures, "paper figures (Figs. 2-4, PNG + PDF) and <outroot>/paper_tables.md")
    _add_outroot(p)
    p.add_argument("--out_dir", type=Path, help="default: <outroot>/figures")
    p.add_argument("--dpi", type=int, default=350)

    p = command("saliency", cmd_saliency, "SmoothGrad saliency for five clips (qualitative)")
    _add_outroot(p, required=False, help="evaluation root: use the best-fitting 28.5 C test embryo")
    p.add_argument("--model", default="full", choices=MODELS)
    p.add_argument("--ckpt", type=Path, help="checkpoint (with --outroot the default is the released one for --model; required with --npy)")
    p.add_argument("--npy", type=Path, help="explicit processed stack instead of --outroot")
    p.add_argument("--out_dir", type=Path, help="output directory (with --npy)")
    p.add_argument("--starts", default="", help="clip starts, e.g. 0,42,84 (default: five evenly spaced)")
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--sg_N", type=int, default=20, help="noisy samples (1 = plain gradient)")
    p.add_argument("--sg_sigma", type=float, default=0.01, help="noise SD on the [0, 1] scale")
    p.add_argument("--hm_lo", type=float, default=90.0)
    p.add_argument("--hm_hi", type=float, default=99.5)
    p.add_argument("--hm_gamma", type=float, default=0.55)
    p.add_argument("--blur_k", type=int, default=9)
    p.add_argument("--alpha_thr", type=float, default=0.25)
    p.add_argument("--alpha_max", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu (default: DEVICE in .env, else auto)")
    p.add_argument("--amp", action="store_true", help="mixed precision for the gradients (default off)")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None, help="use EMA weights (default: USE_EMA in .env, else on)")

    sb = command("sbiad840", None, "Princeton S-BIAD840 cross-site validation and low-shot fine-tuning")
    ssub = sb.add_subparsers(dest="sbiad840_command", metavar="<step>", required=True)

    p = command("preprocess", cmd_sbiad840_preprocess, "import the Princeton PNG export as processed arrays + test splits", parent=ssub)
    p.add_argument("--src_root", type=Path, help="Princeton_Data directory with Dataset_C and Dataset_D (default: SBIAD840_SRC_ROOT)")
    p.add_argument("--out_root", type=Path, help="default: SBIAD840_DIR (data/sbiad840_aligned_4p5)")
    p.add_argument("--align_start_hpf", type=float, default=4.5)
    p.add_argument("--expect_t", type=int, default=EXPECT_T)
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--p_lo", type=float, default=1.0)
    p.add_argument("--p_hi", type=float, default=99.0)
    p.add_argument("--pad_to_expect", action="store_true", help="pad/trim to --expect_t frames (default: keep the native 168)")
    p.add_argument("--limit", type=int, default=0)

    p = command("summarize", cmd_sbiad840_summarize, "anchored and through-origin summary tables of an evaluation root", parent=ssub)
    _add_outroot(p)
    _add_datasets(p, default=SBIAD840_DATASETS)
    _add_models(p)

    p = command("compare-kimmelnet", cmd_sbiad840_compare, "ETF results next to KimmelNet's published values", parent=ssub)
    p.add_argument("--base_outroot", required=True, type=Path, help="summarised evaluation with standard windows")
    p.add_argument("--dense_outroot", required=True, type=Path, help="summarised dense cnn_single evaluation (clip_len 1, stride 1)")

    p = command("make-splits", cmd_sbiad840_make_splits, "low-shot fine-tuning splits of the Princeton 28.5 C pool", parent=ssub)
    p.add_argument("--source_split", type=Path, help="default: SPLIT_28C5_SBIAD840")
    p.add_argument("--out_dir", type=Path, help="default: <source split dir>/finetune")
    p.add_argument("--train_counts", default="12,24")
    p.add_argument("--val_count", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="28C5_sbiad840")

    from .sbiad840 import FINETUNE_STAGES

    p = command("finetune", cmd_sbiad840_finetune, "fine-tune a released checkpoint on Princeton 28.5 C embryos", parent=ssub)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--stage", required=True, choices=list(FINETUNE_STAGES), help="which parts of the model are trained")
    p.add_argument("--split_json", type=Path, help="default: <SPLIT_28C5_SBIAD840 dir>/finetune/28C5_sbiad840_ft12_v12_seed42.json")
    p.add_argument("--proc_dir", type=Path, help="default: PROC_28C5_SBIAD840")
    p.add_argument("--out_dir", type=Path, help="default: <RUNS_DIR>/finetune_<model>_<stage>_<timestamp>")
    p.add_argument("--init_ckpt", type=Path, help="default: the released checkpoint of --model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=0, help="0 = checkpoint value")
    p.add_argument("--val_batch_size", type=int, default=0, help="0 = checkpoint value")
    p.add_argument("--samples_per_embryo", type=int, default=0, help="0 = checkpoint value")
    p.add_argument("--jitter", type=int, default=-1, help="-1 = checkpoint value")
    p.add_argument("--cache_items", type=int, default=-1, help="-1 = checkpoint value")
    p.add_argument("--lr", type=float, default=0.0, help="0 = checkpoint value")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--mem_profile", default="", choices=["", *MEM_PROFILES], help="default: checkpoint value")
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--ema_start_ratio", type=float, default=0.1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="mixed precision (default: AMP in .env, else on)")
    p.add_argument("--init_use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema_eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry_run", action="store_true", help="print the resolved configuration only")

    p = command("eval-finetuned", cmd_sbiad840_eval_finetuned, "evaluate a fine-tuned checkpoint on the Princeton test sets", parent=ssub)
    p.add_argument("--ft_ckpt", required=True, type=Path)
    p.add_argument("--model", required=True, choices=MODELS)
    _add_outroot(p)
    _add_datasets(p, default=SBIAD840_DATASETS)
    p.add_argument("--split_28c5", type=Path, help="e.g. the fine-tune split, so that only held-out embryos are evaluated")
    p.add_argument("--split_25c", type=Path)
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=True, help="recompute existing inference JSON")
    _add_runtime(p)

    p = command("summarize-finetune", cmd_sbiad840_summarize_finetune, "comparison table of fine-tuning runs (Table 7)", parent=ssub)
    p.add_argument("--experiment", action="append", nargs=4, required=True, metavar=("LABEL", "RUN_DIR", "EVAL28_OUTROOT", "EVAL25_OUTROOT"))
    p.add_argument("--out_dir", type=Path, default=Path("runs/sbiad840_finetune_compare"))

    from .ablation import SETTINGS

    p = command("ablation", cmd_ablation, "augmentation-family ablation of ETF-full (Table 4)")
    _add_outroot(p, required=False, help="default: <RUNS_DIR>/min_aug_ablation_<timestamp>")
    p.add_argument("--steps", type=parse_list, default=["train", "evaluate", "summarize"])
    p.add_argument("--settings", type=parse_list, default=[], help=f"default: {','.join(SETTINGS)}")
    p.add_argument("--batch_size", type=int, help="default 88 (Table 4 runs on one 46 GB L40S)")
    p.add_argument("--val_batch_size", type=int, help="default 176")
    p.add_argument("--num_workers", type=int, help="default 16")
    p.add_argument("--epochs", type=int, help="default 300")
    p.add_argument("--patience", type=int, help="default 20")
    p.add_argument("--save_every", type=int, help="default 25")
    p.add_argument("--seed", type=int, help="default 42")
    p.add_argument("--mem_profile", choices=list(MEM_PROFILES), help="default lowmem")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="any other training option, e.g. --set lr=3e-4 --set cnn_base=24 (repeatable)")
    p.add_argument("--dry_run", action="store_true", help="write the configurations and print the commands only")
    p.add_argument("--device", default=None, help="training and evaluation device (default: DEVICE in .env, else auto)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="mixed precision for training and evaluation (default on)")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None, help="evaluate the EMA weights (default: USE_EMA in .env, else on)")
    return parser


def main(argv: list[str] | None = None) -> None:
    from .config import load_settings

    args = build_parser().parse_args(argv)
    args.func(args, load_settings(args.env_file))


if __name__ == "__main__":
    main()
