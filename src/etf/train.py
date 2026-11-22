"""Training, fine-tuning and clip-level evaluation.

Objective (paper Section 3.7), for two clips of the same embryo starting at s1, s2:
    L = lambda_abs * L_abs + lambda_diff * ramp(step) * L_diff
    L_abs  = mean of the L1 (or SmoothL1) errors of both clips against T0 + DT * s
    L_diff = SmoothL1((t_hat(s2) - t_hat(s1)) - DT * (s2 - s1))
``ramp`` rises linearly from 0 to 1 over the first ``cons_ramp_ratio`` of steps.

Multi-GPU training: ``torchrun --nproc_per_node N -m etf train ...``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from . import MODEL_NAME
from .config import VARIANTS, TrainConfig  # noqa: F401  (re-exported)
from .constants import CLIP_LEN, DT_H, MEM_PROFILES
from .data import DeterministicValDataset, PairQueueDataset, aug_params_dict, build_aug_params, filter_excluded, parse_aug_groups
from .model import EmbryoTempoFormer, build_model
from .utils import read_json, write_json


# ---------------------------------------------------------------------------
# Reproducibility, devices and DDP
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own deterministic seed (and reseed the dataset)."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(int(worker_seed))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if ddp_enabled() else 0


def is_main_process() -> bool:
    return ddp_rank() == 0


def ddp_init() -> tuple[torch.device, bool]:
    """Initialise torch.distributed when launched by torchrun; returns (device, ddp)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return torch.device("cuda", local_rank), True
        dist.init_process_group(backend="gloo")
        return torch.device("cpu"), True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False


@torch.no_grad()
def _all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _r2_from_sums(sum_y: torch.Tensor, sum_y2: torch.Tensor, sum_err2: torch.Tensor, n: torch.Tensor) -> float:
    """R^2 = 1 - SSE/SST from aggregated sums (nan if undefined)."""
    eps = 1e-12
    if float(n.item()) < 2:
        return float("nan")
    mean = sum_y / (n + eps)
    sst = sum_y2 - sum_y * mean
    if abs(float(sst.item())) < 1e-12:
        return float("nan")
    return float((1.0 - sum_err2 / (sst + eps)).item())


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# EMA and checkpoints
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of the model weights (shadow state_dict)."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        with torch.no_grad():
            self.shadow: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in _unwrap(model).state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in _unwrap(model).state_dict().items():
            if k not in self.shadow or not v.is_floating_point():
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Load EMA weights into ``model``; returns the previous weights for ``restore``."""
        m = _unwrap(model)
        backup = {k: v.detach().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(self.shadow, strict=True)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        _unwrap(model).load_state_dict(backup, strict=True)


def load_weights(model: nn.Module, ckpt: Mapping[str, Any], use_ema: bool) -> None:
    """Load ``ckpt['ema']`` (if requested and present) or ``ckpt['model']`` into ``model``."""
    state = ckpt["ema"] if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state, strict=True)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, ema, epoch: int, step: int, cfg: TrainConfig, best_mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.shadow if ema else None,
            "epoch": int(epoch),
            "step": int(step),
            "best_mae": float(best_mae),
            "cfg": dataclasses.asdict(cfg),
            "meta": {"model_name": MODEL_NAME, "saved_at": time.strftime("%Y%m%d-%H%M%S", time.localtime())},
        },
        path,
    )


def _load_training_state(path: str, model, optimizer, scheduler, scaler, ema) -> tuple[int, int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if obj is not None and state.get(key) is not None:
            obj.load_state_dict(state[key])
    if ema is not None and state.get("ema") is not None:
        ema.shadow = state["ema"]
    return int(state.get("epoch", 0)), int(state.get("step", 0)), float(state.get("best_mae", float("inf")))


def dump_run_config(out_dir: Path, cfg: TrainConfig) -> None:
    """Write run_config.json (config, memory profile, augmentation, environment)."""
    write_json(
        {
            "train_config": dataclasses.asdict(cfg),
            "mem_profile_detail": MEM_PROFILES.get(cfg.mem_profile, MEM_PROFILES["balanced"]),
            "augment": aug_params_dict(cfg.aug_disable_groups),
            "augment_disable_groups": sorted(parse_aug_groups(cfg.aug_disable_groups)),
            "env": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
                "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        },
        out_dir / "run_config.json",
    )


# ---------------------------------------------------------------------------
# Fine-tuning freeze policy
# ---------------------------------------------------------------------------
def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(flag)


def apply_finetune_policy(model: EmbryoTempoFormer, cfg: TrainConfig) -> dict[str, int]:
    """Freeze / unfreeze parts of the model in place; returns parameter counts."""
    if cfg.freeze_frame_encoder:
        _set_requires_grad(model.frame_enc, False)
    if cfg.freeze_temporal:
        _set_requires_grad(model.blocks, False)
        _set_requires_grad(model.norm, False)
        model.cls_token.requires_grad = False
    if cfg.freeze_head:
        _set_requires_grad(model.head, False)

    if cfg.unfreeze_frame_proj:
        _set_requires_grad(model.frame_enc.proj, True)
    n_frame = max(0, int(cfg.unfreeze_frame_tail_blocks))
    if n_frame > 0:  # projection + the last N CNN blocks
        _set_requires_grad(model.frame_enc.proj, True)
        blocks = list(model.frame_enc.blocks.children())
        for blk in blocks[-min(n_frame, len(blocks)):]:
            _set_requires_grad(blk, True)
    n_temporal = max(0, int(cfg.unfreeze_temporal_tail_blocks))
    if n_temporal > 0:  # last N Transformer blocks + the CLS/norm readout path
        model.cls_token.requires_grad = True
        _set_requires_grad(model.norm, True)
        blocks = list(model.blocks.children())
        for blk in blocks[-min(n_temporal, len(blocks)):]:
            _set_requires_grad(blk, True)

    return {
        "total_params": sum(int(p.numel()) for p in model.parameters()),
        "trainable_params": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _clips_to_device(x: torch.Tensor, device: torch.device, amp: bool) -> torch.Tensor:
    """uint8 clips -> device, normalised to [0, 1] (fp16 under CUDA AMP)."""
    dtype = torch.float16 if (amp and device.type == "cuda") else torch.float32
    return x.to(device=device, dtype=dtype, non_blocking=True).div_(255.0)


def _abs_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target) if kind == "smoothl1" else F.l1_loss(pred, target)


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda"))


def build_loaders(cfg: TrainConfig, ddp: bool):
    split = read_json(cfg.split_json)
    ds_train = PairQueueDataset(
        Path(cfg.proc_dir), filter_excluded(split["train"]), cfg.clip_len, True, build_aug_params(cfg.aug_disable_groups),
        cfg.seed, cfg.samples_per_embryo, cfg.jitter, cfg.cache_items,
    )
    ds_val = DeterministicValDataset(Path(cfg.proc_dir), filter_excluded(split["val"]), cfg.clip_len, cache_items=cfg.val_cache_items)
    train_sampler = DistributedSampler(ds_train, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if ddp else None

    g = torch.Generator()  # shared by both loaders (sampler order and worker seeds)
    g.manual_seed(cfg.seed)

    def loader(ds, batch_size: int, workers: int, **kw) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
                          worker_init_fn=seed_worker, generator=g, **kw)

    dl_train = loader(ds_train, cfg.batch_size, cfg.num_workers, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)
    dl_val = loader(ds_val, cfg.val_batch_size, max(0, cfg.num_workers // 2), shuffle=False, sampler=val_sampler, drop_last=False)
    return dl_train, dl_val, train_sampler


def train_one_epoch(model, dl_train, optimizer, scheduler, scaler, ema: Optional[EMA], device: torch.device, ddp: bool,
                    cfg: TrainConfig, epoch: int, train_sampler, global_step: int, total_optim_steps: int) -> tuple[dict[str, float], int]:
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    sums = torch.zeros(6, device=device)  # loss, abs, cons, |err|, err^2, n
    ramp_steps = int(cfg.cons_ramp_ratio * total_optim_steps)
    accum = max(1, int(cfg.grad_accum))
    use_scaler = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for local_iter, batch in enumerate(dl_train, start=1):
        x1 = _clips_to_device(batch["x1"], device, cfg.amp)
        x2 = _clips_to_device(batch["x2"], device, cfg.amp)
        t1 = batch["offset1"].to(device, non_blocking=True)
        t2 = batch["offset2"].to(device, non_blocking=True)
        s1 = batch["start1"].to(device, non_blocking=True).float()
        s2 = batch["start2"].to(device, non_blocking=True).float()
        B = x1.shape[0]

        with _autocast(device, cfg.amp):
            p1 = model(x1)
            p2 = model(x2)
            loss_abs = 0.5 * (_abs_loss(p1, t1, cfg.abs_loss_type) + _abs_loss(p2, t2, cfg.abs_loss_type))
            loss_cons = F.smooth_l1_loss(p2 - p1, DT_H * (s2 - s1))
            ramp = 1.0 if ramp_steps <= 0 else min(1.0, (global_step + 1) / float(ramp_steps))
            loss = cfg.lambda_abs * loss_abs + cfg.lambda_diff * (ramp * loss_cons)
            loss_scaled = loss / float(accum)

        if use_scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if local_iter % accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            stepped = True
            if use_scaler:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before  # False when the step was skipped (inf/nan)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if stepped and scheduler is not None:
                scheduler.step()
            if ema is not None and total_optim_steps > 0 and global_step / float(total_optim_steps) >= float(cfg.ema_start_ratio):
                ema.update(model)
            global_step += 1

        with torch.no_grad():
            err1, err2 = p1 - t1, p2 - t2
            sums[0] += loss.detach() * B
            sums[1] += loss_abs.detach() * B
            sums[2] += loss_cons.detach() * B
            sums[3] += err1.abs().sum() + err2.abs().sum()
            sums[4] += (err1**2).sum() + (err2**2).sum()
            sums[5] += float(2 * B)

    if ddp:
        _all_reduce_sum_(sums)
    n = float(sums[5].item())
    metrics = {
        "loss": float((sums[0] / (n / 2.0)).item()),
        "abs": float((sums[1] / (n / 2.0)).item()),
        "cons": float((sums[2] / (n / 2.0)).item()),
        "mae": float((sums[3] / n).item()),
        "rmse": float((sums[4] / n).sqrt().item()),
    }
    return metrics, global_step


@torch.no_grad()
def validate(model, dl_val, device: torch.device, ddp: bool, amp: bool) -> dict[str, float]:
    """Clip-level validation metrics (loss, mae, rmse, r2); clips are correlated within embryo."""
    model.eval()
    sums = torch.zeros(4, device=device)  # loss, |err|, err^2, n
    r2_sums = torch.zeros(4, device=device)  # sum_y, sum_y2, sum_err2, n
    for batch in dl_val:
        x = _clips_to_device(batch["x"], device, amp)
        y = batch["target"].to(device, non_blocking=True)
        with _autocast(device, amp):
            p = model(x)
            loss = F.l1_loss(p, y)
        err = p - y
        B = x.shape[0]
        sums[0] += loss * B
        sums[1] += err.abs().sum()
        sums[2] += (err**2).sum()
        sums[3] += float(B)
        r2_sums[0] += y.sum()
        r2_sums[1] += (y**2).sum()
        r2_sums[2] += (err**2).sum()
        r2_sums[3] += float(B)
    if ddp:
        _all_reduce_sum_(sums)
        _all_reduce_sum_(r2_sums)
    n = max(1.0, float(sums[3].item()))
    return {
        "loss": float((sums[0] / n).item()),
        "mae": float((sums[1] / n).item()),
        "rmse": float((sums[2] / n).sqrt().item()),
        "r2": _r2_from_sums(*r2_sums),
    }


HISTORY_COLUMNS = [
    "epoch", "step",
    "train_loss", "train_mae", "train_rmse", "train_abs", "train_cons",
    "val_loss", "val_mae", "val_rmse", "val_r2",
    "lr", "secs", "mem_profile", "gpu_mem_gb_rank0",
]


def train(cfg: TrainConfig) -> None:
    """Train (or fine-tune with ``init_ckpt``) and write best.pt / last.pt / history.csv to ``out_dir``."""
    if cfg.resume and cfg.init_ckpt:
        raise ValueError("Use either resume or init_ckpt, not both.")

    device, ddp = ddp_init()
    if not ddp and cfg.device != "auto":
        device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    seed_all(cfg.seed + ddp_rank())

    model = build_model(cfg).to(device)
    if cfg.init_ckpt:
        init_state = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)
        load_weights(model, init_state, use_ema=bool(cfg.init_use_ema))
        if is_main_process():
            src = "ema" if cfg.init_use_ema and init_state.get("ema") is not None else "model"
            print(f"[init_ckpt] loaded {src} weights from {cfg.init_ckpt}")

    counts = apply_finetune_policy(model, cfg)
    if counts["trainable_params"] <= 0:
        raise ValueError("No trainable parameters remain after applying the freeze policy.")
    if is_main_process():
        print(f"[trainable] {counts['trainable_params']:,} / {counts['total_params']:,} parameters")

    if ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=cfg.temporal_mode != "transformer")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    dl_train, dl_val, train_sampler = build_loaders(cfg, ddp=ddp)

    steps_per_epoch = max(1, math.ceil(len(dl_train) / max(1, int(cfg.grad_accum))))
    total_optim_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = min(max(1, int(cfg.warmup_ratio * total_optim_steps)), total_optim_steps)

    def lr_lambda(step: int) -> float:  # linear warm-up, then cosine decay to lr_min_ratio
        if step < warmup_steps:
            return (step + 1) / float(warmup_steps)
        if total_optim_steps == warmup_steps:
            return cfg.lr_min_ratio
        t = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    ema = EMA(_unwrap(model), decay=cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None

    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.csv"
    if is_main_process():
        dump_run_config(out_dir, cfg)

    start_epoch, global_step, best_mae = 0, 0, float("inf")
    if cfg.resume:
        se, global_step, best_mae = _load_training_state(cfg.resume, _unwrap(model), optimizer, scheduler, scaler, ema)
        start_epoch = se + 1
        if is_main_process():
            print(f"[resume] epoch={start_epoch} global_step={global_step} best_mae={best_mae:.6f}")

    if is_main_process() and not hist_path.exists():
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)

    epochs_without_improve = 0
    last_epoch = max(0, start_epoch - 1)
    for epoch in range(start_epoch, cfg.epochs):
        last_epoch = epoch
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_m, global_step = train_one_epoch(model, dl_train, optimizer, scheduler, scaler, ema, device, ddp, cfg,
                                               epoch, train_sampler, global_step, total_optim_steps)

        run_val = (epoch + 1) % max(1, int(cfg.val_every)) == 0 or epoch == cfg.epochs - 1
        val_m = {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
        if run_val:
            backup = ema.copy_to(_unwrap(model)) if (ema is not None and cfg.ema_eval) else None
            val_m = validate(model, dl_val, device, ddp, cfg.amp)
            if backup is not None:
                ema.restore(_unwrap(model), backup)

        secs = time.time() - t0
        lr_cur = scheduler.get_last_lr()[0]
        mem_gb = float(torch.cuda.max_memory_reserved() / (1024**3)) if device.type == "cuda" else float("nan")

        should_stop = False
        if is_main_process():
            with open(hist_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch, global_step,
                    train_m["loss"], train_m["mae"], train_m["rmse"], train_m["abs"], train_m["cons"],
                    val_m["loss"], val_m["mae"], val_m["rmse"], val_m["r2"],
                    lr_cur, secs, cfg.mem_profile, mem_gb,
                ])
            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch{epoch + 1}.pt", model, optimizer, scheduler, scaler, ema, epoch, global_step, cfg, best_mae)
            if run_val and val_m["mae"] + 1e-9 < best_mae:
                best_mae = float(val_m["mae"])
                epochs_without_improve = 0
                save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, scaler, ema, epoch, global_step, cfg, best_mae)
            elif run_val:
                epochs_without_improve += 1
            print(f"[epoch {epoch}] train_mae={train_m['mae']:.3f} val_mae={val_m['mae']:.3f} lr={lr_cur:.2e} "
                  f"mem={mem_gb:.1f}GB profile={cfg.mem_profile} val_run={int(run_val)}")
            if run_val and cfg.patience > 0 and epochs_without_improve >= int(cfg.patience):
                print(f"[early-stop] no val_mae improvement for {epochs_without_improve} epoch(s); "
                      f"stopping at epoch {epoch} with best_mae={best_mae:.6f}")
                should_stop = True

        if ddp:
            flag = torch.tensor([int(should_stop)], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            should_stop = bool(flag.item())
        if should_stop:
            break

    if is_main_process():
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, ema, last_epoch, global_step, cfg, best_mae)
    if ddp:
        dist.destroy_process_group()


def evaluate(ckpt: str, proc_dir: str, split_json: str, *, clip_len: int = CLIP_LEN, batch_size: int = 64, num_workers: int = 4,
             amp: bool = True, device: str = "auto", mem_profile: str = "lowmem", use_ema: bool = True) -> dict[str, float]:
    """Clip-level metrics of a checkpoint on the ``val`` split (descriptive only)."""
    dev = resolve_device(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(state.get("cfg", {}), mem_profile=mem_profile)
    model = build_model(cfg)
    load_weights(model, state, use_ema=use_ema)
    model.to(dev)
    ds_val = DeterministicValDataset(Path(proc_dir), filter_excluded(read_json(split_json)["val"]), clip_len=clip_len)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return validate(model, dl_val, dev, ddp=False, amp=amp)
