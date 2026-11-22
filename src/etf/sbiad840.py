"""Cross-site validation on the Princeton data set S-BIAD840 (paper Section 4.6, Table 7).

S-BIAD840 stores one directory per time point (``Dataset_C`` = 28.5 C,
``Dataset_D`` = 25 C), each with one PNG per embryo. ``preprocess_png`` rebuilds
one stack per embryo aligned to 4.5 hpf (Dataset_C drops 4 frames, Dataset_D
drops 8, leaving 168 frames to 46.25 hpf) and applies the ETF preprocessing.
The native length is kept by default (no synthetic padded tail).

For comparability with KimmelNet the summaries add the through-origin fit
y = m x (``m_origin``) next to ETF's anchored readouts.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CLIP_LEN, DT_H, EXPECT_T, IMG_SIZE, MODELS, SBIAD840_25C, SBIAD840_28C5, SBIAD840_DATASETS, STRIDE, T0_HPF
from .utils import fmt, markdown_table, parse_list, read_csv_rows, read_json, write_csv, write_json

PNG_RE = re.compile(r"^(?P<prefix>.+?\.nd2)-(?P<embryo>\d+)-(?P<timeidx>\d+)\.png$")
SOURCE_DATASETS = {"Dataset_C": "28C5", "Dataset_D": "25C"}

# KimmelNet results on the same data (Jones, Renshaw & Barry, Life Sci Alliance 2024; doi:10.26508/lsa.202302351).
KIMMELNET = {
    SBIAD840_28C5: {"dataset_label": "Dataset C", "prediction_error_mean_h": 1.30, "prediction_error_sd_h": 5.52, "slope": 0.923, "r2_line_fit": 0.644},
    SBIAD840_25C: {"dataset_label": "Dataset D", "prediction_error_mean_h": 1.18, "prediction_error_sd_h": 5.82, "slope": 0.773, "r2_line_fit": 0.533},
}

# Fine-tuning stages: which parts of a released checkpoint are trainable.
FINETUNE_STAGES: dict[str, dict[str, Any]] = {
    "head_only": dict(freeze_frame_encoder=True, freeze_temporal=True),
    "proj_head": dict(freeze_frame_encoder=True, freeze_temporal=True, unfreeze_frame_proj=True),
    "frame_tail1": dict(freeze_frame_encoder=True, freeze_temporal=True, unfreeze_frame_tail_blocks=1),
    "frame_tail2": dict(freeze_frame_encoder=True, freeze_temporal=True, unfreeze_frame_tail_blocks=2),
    "temporal_last1": dict(freeze_frame_encoder=True, freeze_temporal=True, unfreeze_temporal_tail_blocks=1),
    "temporal_last2": dict(freeze_frame_encoder=True, freeze_temporal=True, unfreeze_temporal_tail_blocks=2),
    "temporal_head": dict(freeze_frame_encoder=True),
    "full_trainable": dict(),
}
TEMPORAL_STAGES = {"temporal_last1", "temporal_last2", "temporal_head"}


# ---------------------------------------------------------------------------
# PNG import
# ---------------------------------------------------------------------------
def _index_png_folder(folder: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in folder.glob("*.png"):
        m = PNG_RE.match(p.name)
        if m is None:
            continue
        idx = int(m.group("embryo"))
        if idx in out:
            raise ValueError(f"duplicate embryo index {idx} in {folder}")
        out[idx] = p
    return out


def preprocess_png(src_root: str | Path, out_root: str | Path, *, align_start_hpf: float = 4.5, expect_t: int = EXPECT_T,
                   img_size: int = IMG_SIZE, p_lo: float = 1.0, p_hi: float = 99.0, pad_to_expect: bool = False, limit: int = 0) -> None:
    """Convert ``<src_root>/Dataset_{C,D}/<hpf>/*.png`` into processed arrays and test-only split files."""
    from PIL import Image

    from .preprocess import preprocess_stack, save_proc_npy

    src_root, out_root = Path(src_root), Path(out_root)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    for src_name, tag in SOURCE_DATASETS.items():
        ds_dir = src_root / src_name
        if not ds_dir.exists():
            raise FileNotFoundError(ds_dir)
        time_dirs = sorted((p for p in ds_dir.iterdir() if p.is_dir()), key=lambda p: float(p.name))
        keep_dirs = [p for p in time_dirs if float(p.name) >= align_start_hpf]
        if not keep_dirs:
            raise ValueError(f"no frames at/after {align_start_hpf} hpf in {ds_dir}")
        indices = [_index_png_folder(d) for d in keep_dirs]
        embryo_ids = sorted(set(indices[0]).intersection(*indices[1:]))
        if not embryo_ids:
            raise ValueError(f"no consistent embryo ids across kept dirs in {ds_dir}")

        proc_dir = out_root / f"processed_{tag}_sbiad840"
        meta_all: dict[str, Any] = {}
        for embryo_idx in embryo_ids[:limit] if limit > 0 else embryo_ids:
            files = [idx[embryo_idx] for idx in indices]
            frames = []
            for fp in files:
                with Image.open(fp) as im:
                    frames.append(np.asarray(im.convert("L")))
            raw = np.stack(frames, axis=0)
            proc, meta = preprocess_stack(raw, expect_t=expect_t, img_size=img_size, p_lo=p_lo, p_hi=p_hi, pad_to_expect=pad_to_expect)
            eid = f"SBIAD840_{tag}_E{embryo_idx:03d}"
            save_proc_npy(proc_dir, eid, proc)
            meta_all[eid] = {
                "source_dataset": src_name,
                "source_embryo_index": embryo_idx,
                "orig_time_start_hpf": float(time_dirs[0].name),
                "orig_time_end_hpf": float(time_dirs[-1].name),
                "kept_time_start_hpf": float(keep_dirs[0].name),
                "kept_time_end_hpf": float(keep_dirs[-1].name),
                "n_time_dirs_total": len(time_dirs),
                "n_time_dirs_kept": len(keep_dirs),
                "dropped_pre_frames": len(time_dirs) - len(keep_dirs),
                "raw_shape": list(raw.shape),
                "first_source_file": files[0].name,
                "last_source_file": files[-1].name,
                "preprocess": meta,
            }
        args = dict(src_root=str(src_root), dataset_dir=str(ds_dir), align_start_hpf=align_start_hpf, expect_t=expect_t,
                    img_size=img_size, p_lo=p_lo, p_hi=p_hi, pad_to_expect=bool(pad_to_expect), limit=limit)
        write_json({"meta": meta_all, "args": args}, proc_dir / "preprocess_meta.json")
        split = {"train": [], "val": [], "test": list(meta_all),
                 "meta": {"source_dataset": src_name, "condition_tag": tag, "test_only_external_domain": True,
                          "align_start_hpf": align_start_hpf, "dt_h": DT_H}}
        split_path = out_root / "splits" / f"{tag}_sbiad840_test.json"
        write_json(split, split_path)
        print(f"[sbiad840] {src_name} -> {proc_dir} ({len(meta_all)} embryos); split={split_path}")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def origin_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Through-origin fit y = m x (KimmelNet convention) and its residual statistics."""
    denom = float(np.dot(x, x))
    m = float(np.dot(x, y) / denom) if denom > 0 else float("nan")
    resid = y - m * x
    ss_res = float(np.sum(np.square(resid)))
    ss_tot = float(np.sum(np.square(y - float(np.mean(y)))))
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    return {
        "m_origin": m,
        "origin_resid_mean": float(np.mean(resid)),
        "origin_resid_sd": float(np.std(resid, ddof=0)),
        "origin_resid_rmse": float(np.sqrt(np.mean(np.square(resid)))),
        "origin_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "corr_r2": corr * corr,
    }


def summarize_external(outroot: str | Path, *, datasets: Sequence[str] = SBIAD840_DATASETS, models: Sequence[str] = MODELS,
                       out_csv: str | Path | None = None, out_md: str | Path | None = None) -> list[dict[str, Any]]:
    """sbiad840_external_summary.{csv,md}: anchored and through-origin readouts per dataset x model."""
    outroot = Path(outroot)
    rows = []
    for ds in datasets:
        for model in models:
            d = outroot / ds / model
            if not ((d / "summary.json").exists() and (d / "points.csv").exists() and (d / "json").exists()):
                continue
            s = read_json(d / "summary.json")
            gm, fa = s["global_metrics_points"], s["fit_anchor_T0"]
            pts = read_csv_rows(d / "points.csv")
            x = np.asarray([float(r["x_true"]) for r in pts], dtype=np.float64)
            y = np.asarray([float(r["y_pred"]) for r in pts], dtype=np.float64)
            t0 = np.asarray([float(json.loads(fp.read_text())["t0_final"]) for fp in sorted((d / "json").glob("*.json"))], dtype=np.float64)
            o = origin_fit(x, y)
            rows.append({
                "dataset": ds, "model": model, "n_json": int(len(t0)), "n_points": int(len(x)),
                "mae_h": float(gm["mae"]), "rmse_h": float(gm["rmse"]), "r2_points": float(gm["r2"]),
                "m_anchor": float(fa["m"]), "rmse_resid_h": float(fa["rmse_resid"]), "max_abs_resid_h": float(fa["max_abs_resid"]),
                "t0_final_mean_h": float(np.mean(t0)), "t0_final_median_h": float(np.median(t0)), "t0_final_sd_h": float(np.std(t0, ddof=0)),
                "m_origin": o["m_origin"], "origin_resid_mean_h": o["origin_resid_mean"], "origin_resid_sd_h": o["origin_resid_sd"],
                "origin_resid_rmse_h": o["origin_resid_rmse"], "origin_r2": o["origin_r2"], "corr_r2": o["corr_r2"],
            })
    if not rows:
        raise SystemExit(f"No aggregated S-BIAD840 results under {outroot}")
    write_csv(Path(out_csv) if out_csv else outroot / "sbiad840_external_summary.csv", rows, list(rows[0].keys()))
    pretty = [{
        "dataset": r["dataset"], "model": r["model"], "RMSE (h)": fmt(r["rmse_h"]), "R2": fmt(r["r2_points"]),
        "m_anchor": fmt(r["m_anchor"]), "rmse_resid (h)": fmt(r["rmse_resid_h"]), "t0 median (h)": fmt(r["t0_final_median_h"]),
        "m_origin": fmt(r["m_origin"]), "origin resid mean±sd (h)": f"{fmt(r['origin_resid_mean_h'])} ± {fmt(r['origin_resid_sd_h'])}",
        "origin R2": fmt(r["origin_r2"]), "corr^2": fmt(r["corr_r2"]),
    } for r in rows]
    md = markdown_table(pretty, list(pretty[0].keys()))
    (Path(out_md) if out_md else outroot / "sbiad840_external_summary.md").write_text(md, encoding="utf-8")
    print(f"[sbiad840] summary written under {outroot}")
    return rows


COMPARE_FIELDS = ["dataset", "source", "protocol", "RMSE (h)", "m_anchor", "rmse_resid (h)", "t0 median (h)", "m_origin",
                  "origin resid mean±sd (h)", "through-origin line-fit R2", "corr^2"]


def compare_kimmelnet(base_csv: str | Path, dense_csv: str | Path, out_csv: str | Path, out_md: str | Path) -> None:
    """ETF (standard windows and dense single-frame cnn_single) next to KimmelNet's published values."""
    def etf_row(dataset: str, source: str, protocol: str, r: dict[str, str]) -> dict[str, str]:
        return {"dataset": dataset, "source": source, "protocol": protocol, "RMSE (h)": fmt(r["rmse_h"]), "m_anchor": fmt(r["m_anchor"]),
                "rmse_resid (h)": fmt(r["rmse_resid_h"]), "t0 median (h)": fmt(r["t0_final_median_h"]), "m_origin": fmt(r["m_origin"]),
                "origin resid mean±sd (h)": f"{fmt(r['origin_resid_mean_h'])} ± {fmt(r['origin_resid_sd_h'])}",
                "through-origin line-fit R2": fmt(r["origin_r2"]), "corr^2": fmt(r["corr_r2"])}

    base, dense = read_csv_rows(base_csv), read_csv_rows(dense_csv)
    merged = []
    for ds in SBIAD840_DATASETS:
        lit = KIMMELNET[ds]
        merged.append({"dataset": ds, "source": f"KimmelNet {lit['dataset_label']}", "protocol": "paper dense single-image",
                       "RMSE (h)": "NA", "m_anchor": "NA", "rmse_resid (h)": "NA", "t0 median (h)": "NA", "m_origin": fmt(lit["slope"]),
                       "origin resid mean±sd (h)": f"{fmt(lit['prediction_error_mean_h'])} ± {fmt(lit['prediction_error_sd_h'])}",
                       "through-origin line-fit R2": fmt(lit["r2_line_fit"]), "corr^2": "-"})
        merged += [etf_row(ds, f"ETF {r['model']}", "current ETF windowing", r) for r in base if r["dataset"] == ds]
        merged += [etf_row(ds, "ETF cnn_single dense", "clip_len=1, stride=1", r) for r in dense if r["dataset"] == ds]
    write_csv(out_csv, merged, COMPARE_FIELDS)
    Path(out_md).write_text(markdown_table(merged, COMPARE_FIELDS), encoding="utf-8")
    print(f"[sbiad840] {out_csv}")


# ---------------------------------------------------------------------------
# Low-shot fine-tuning
# ---------------------------------------------------------------------------
def make_finetune_splits(source_split: str | Path, out_dir: str | Path, *, train_counts: str = "12,24", val_count: int = 12,
                         seed: int = 42, prefix: str = "28C5_sbiad840") -> list[Path]:
    """Random train/val/test splits of the Princeton 28.5 C pool (e.g. 12/12/72 and 24/12/60)."""
    src = read_json(source_split)
    pool = list(dict.fromkeys(src.get("train", []) + src.get("val", []) + src.get("test", [])))
    if not pool:
        raise ValueError(f"No embryo ids found in {source_split}")
    counts = [int(x) for x in parse_list(train_counts)]
    if not counts:
        raise ValueError(f"no training-set sizes in {train_counts!r}")
    random.Random(int(seed)).shuffle(pool)
    out_dir, manifest, paths = Path(out_dir), [], []
    for n_train in counts:
        if n_train <= 0 or n_train + val_count >= len(pool):
            raise ValueError(f"invalid split sizes train={n_train} val={val_count} for a pool of {len(pool)}")
        split = {"train": pool[:n_train], "val": pool[n_train:n_train + val_count], "test": pool[n_train + val_count:]}
        name = f"{prefix}_ft{n_train}_v{val_count}_seed{seed}.json"
        write_json(split, out_dir / name)
        paths.append(out_dir / name)
        manifest.append({"name": name, "train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"]),
                         "seed": int(seed), "source": str(source_split)})
        print(f"[sbiad840] {out_dir / name}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    write_json({"splits": manifest}, out_dir / f"{prefix}_manifest_seed{seed}.json")
    return paths


def finetune_config(model: str, stage: str, *, init_ckpt: str | Path, split_json: str, proc_dir: str, out_dir: str,
                    epochs: int = 30, num_workers: int = 8, batch_size: int = 0, val_batch_size: int = 0, samples_per_embryo: int = 0,
                    jitter: int = -1, cache_items: int = -1, grad_accum: int = 1, patience: int = 0, save_every: int = 1,
                    seed: int = 42, device: str = "auto", lr: float = 0.0, mem_profile: str = "", ema_decay: float = 0.0,
                    ema_start_ratio: float = 0.1, amp: bool = True, init_use_ema: bool = True, ema_eval: bool = False):
    """TrainConfig for fine-tuning: architecture and optimiser settings from the checkpoint, fresh optimiser state.

    Batch sizes, samples per embryo, jitter, cache size and learning rate default to the checkpoint's values
    (pass a positive value, or >= 0 for jitter/cache_items, to override).
    """
    import torch

    from .config import TrainConfig

    if stage not in FINETUNE_STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(FINETUNE_STAGES)}")
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False).get("cfg")
    if not isinstance(ck, dict):
        raise SystemExit(f"Checkpoint {init_ckpt} does not contain a cfg dict")
    temporal_mode = str(ck.get("temporal_mode", "transformer"))
    if stage in TEMPORAL_STAGES and temporal_mode != "transformer":
        raise SystemExit(f"Stage '{stage}' requires temporal_mode=transformer, got {temporal_mode!r} ({model})")

    def pick(value, key, minimum=1):
        return value if value >= minimum else ck[key]

    return TrainConfig(
        proc_dir=proc_dir, split_json=split_json, out_dir=out_dir,
        clip_len=int(ck.get("clip_len", CLIP_LEN)), img_size=int(ck.get("img_size", IMG_SIZE)), expect_t=int(ck.get("expect_t", EXPECT_T)),
        batch_size=int(pick(batch_size, "batch_size")), val_batch_size=int(pick(val_batch_size, "val_batch_size")),
        num_workers=int(num_workers), samples_per_embryo=int(pick(samples_per_embryo, "samples_per_embryo")),
        jitter=int(pick(jitter, "jitter", 0)), aug_disable_groups="", cache_items=int(pick(cache_items, "cache_items", 0)),
        val_cache_items=16, epochs=int(epochs), lr=float(lr if lr > 0 else ck["lr"]),
        weight_decay=float(ck.get("weight_decay", 0.01)), warmup_ratio=float(ck.get("warmup_ratio", 0.01)),
        lr_min_ratio=float(ck.get("lr_min_ratio", 0.05)), max_grad_norm=float(ck.get("max_grad_norm", 1.0)), grad_accum=int(grad_accum),
        model_dim=int(ck.get("model_dim", 128)), model_depth=int(ck.get("model_depth", 4)), model_heads=int(ck.get("model_heads", 4)),
        model_mlp_ratio=float(ck.get("model_mlp_ratio", 2.0)), drop=float(ck.get("drop", 0.1)), attn_drop=float(ck.get("attn_drop", 0.0)),
        temporal_drop_p=float(ck.get("temporal_drop_p", 0.0)), temporal_mode=temporal_mode, cnn_base=int(ck.get("cnn_base", 24)),
        cnn_expand=int(ck.get("cnn_expand", 2)), cnn_se_reduction=int(ck.get("cnn_se_reduction", 4)),
        mem_profile=mem_profile or str(ck.get("mem_profile", "balanced")),
        lambda_abs=float(ck.get("lambda_abs", 1.0)), lambda_diff=float(ck.get("lambda_diff", 1.0)),
        cons_ramp_ratio=float(ck.get("cons_ramp_ratio", 0.2)), abs_loss_type=str(ck.get("abs_loss_type", "l1")),
        amp=bool(amp), ema_decay=float(ema_decay), ema_start_ratio=float(ema_start_ratio), ema_eval=bool(ema_eval),
        seed=int(seed), device=device, resume="", init_ckpt=str(init_ckpt), init_use_ema=bool(init_use_ema),
        save_every=int(save_every), patience=int(patience), val_every=1, **FINETUNE_STAGES[stage],
    )


def eval_finetuned(ft_ckpt: str | Path, model: str, outroot: str | Path, *, datasets: dict[str, tuple[Path, Path]],
                   clip_len: int = CLIP_LEN, stride: int = STRIDE, dt: float = DT_H, t0: float = T0_HPF, device: str = "auto",
                   amp: bool = True, use_ema: bool = True, force: bool = True) -> None:
    """Inference + aggregation + summary for one fine-tuned checkpoint on the Princeton test sets."""
    from .inference import infer_all
    from .tempo import aggregate_outroot

    infer_all(outroot=outroot, datasets=datasets, checkpoints={model: Path(ft_ckpt)}, clip_len=clip_len, stride=stride,
              device=device, amp=amp, use_ema=use_ema, force=force)
    aggregate_outroot(outroot, list(datasets), [model], dt=dt, t0=t0, force=True)
    summarize_external(outroot, datasets=list(datasets), models=[model])


def _trainable_params(run_config: dict[str, Any]) -> tuple[int, int]:
    from .config import TrainConfig
    from .model import build_model
    from .train import apply_finetune_policy

    cfg = TrainConfig.from_dict(run_config["train_config"])
    counts = apply_finetune_policy(build_model(cfg), cfg)
    return counts["total_params"], counts["trainable_params"]


def summarize_finetune(experiments: Sequence[tuple[str, str, str, str]], out_csv: str | Path, out_md: str | Path) -> list[dict[str, Any]]:
    """Table of fine-tuning runs (label, run dir, held-out 28.5 C eval root, 25 C eval root) with parameter counts."""
    rows = []
    for label, run_dir, eval28, eval25 in experiments:
        run_dir = Path(run_dir)
        run_cfg = read_json(run_dir / "run_config.json")
        tc = run_cfg["train_config"]
        split = read_json(Path(tc["split_json"]).expanduser().resolve())
        total, trainable = _trainable_params(run_cfg)
        init_name = Path(tc["init_ckpt"]).name
        model = next((m for m in MODELS if m in init_name or m in run_dir.name), None)
        if model is None:
            raise SystemExit(f"Could not infer the model tag from {run_dir} / {init_name}")
        common = {
            "experiment": label, "model": model, "stage": label[len(model) + 1:] if "_" in label else label,
            "fine_tune_dataset": "SBIAD840_28C5", "fine_tune_split": Path(tc["split_json"]).name,
            "train_embryos": len(split["train"]), "val_embryos": len(split["val"]), "heldout_28c5_embryos": len(split["test"]),
            "total_params": total, "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total if total else 0.0,
            "init_ckpt": init_name,
        }
        for ds, root in ((SBIAD840_28C5, eval28), (SBIAD840_25C, eval25)):
            match = [r for r in read_csv_rows(Path(root) / "sbiad840_external_summary.csv") if r["dataset"] == ds and r["model"] == model]
            if len(match) != 1:
                raise SystemExit(f"Expected one row for {ds}/{model} in {root}, got {len(match)}")
            r = match[0]
            rows.append({**common, "eval_dataset": ds, **{k: float(r[k]) for k in (
                "mae_h", "rmse_h", "r2_points", "m_anchor", "rmse_resid_h", "t0_final_median_h", "m_origin",
                "origin_resid_mean_h", "origin_resid_sd_h", "origin_r2", "corr_r2")}})
    write_csv(out_csv, rows, list(rows[0].keys()))
    headers = ["experiment", "eval_dataset", "params", "split", "RMSE (h)", "m_origin", "origin resid mean±sd (h)", "t0 median (h)",
               "through-origin line-fit R2", "corr^2"]
    pretty = [dict(zip(headers, [
        r["experiment"], r["eval_dataset"], f"{r['trainable_params']}/{r['total_params']} ({fmt(r['trainable_pct'], 1)}%)",
        f"{r['train_embryos']}/{r['val_embryos']}/{r['heldout_28c5_embryos']}", fmt(r["rmse_h"]), fmt(r["m_origin"]),
        f"{fmt(r['origin_resid_mean_h'])} ± {fmt(r['origin_resid_sd_h'])}", fmt(r["t0_final_median_h"]), fmt(r["origin_r2"]),
        fmt(r["corr_r2"])])) for r in rows]
    Path(out_md).write_text(markdown_table(pretty, headers), encoding="utf-8")
    print(f"[sbiad840] {out_csv}")
    return rows
