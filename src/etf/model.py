"""EmbryoTempoFormer network.

A clip [B, L, 1, H, W] is encoded frame by frame by a lightweight CNN into one
token per frame; the tokens are aggregated over time (identity / mean pooling /
Transformer with a CLS token and rotary position embeddings) and regressed to a
developmental time in hpf.

Attribute names of modules that hold parameters (``frame_enc``, ``blocks``,
``norm``, ``head``, ``cls_token`` and their children) define the state_dict keys
of the released checkpoints and must not be renamed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from .constants import MEM_PROFILES, TEMPORAL_MODES


def gn_groups(ch: int, max_groups: int = 8) -> int:
    """Largest GroupNorm group count <= max_groups that divides ``ch``."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4, min_hidden: int = 8):
        super().__init__()
        hidden = max(channels // reduction, min_hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class DSConvSEBlock(nn.Module):
    """Depthwise-separable conv block with GroupNorm and squeeze-excitation."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, expand_ratio: int = 2, se_reduction: int = 4):
        super().__init__()
        mid = c_in * expand_ratio
        self.use_res = stride == 1 and c_in == c_out
        self.pw1 = nn.Conv2d(c_in, mid, 1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups(mid), mid)
        self.dw = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups(mid), mid)
        self.pw2 = nn.Conv2d(mid, c_out, 1, bias=False)
        self.gn3 = nn.GroupNorm(gn_groups(c_out), c_out)
        self.se = SEBlock(c_out, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gn1(self.pw1(x)))
        y = F.silu(self.gn2(self.dw(y)))
        y = self.se(self.gn3(self.pw2(y)))
        return x + y if self.use_res else y


class FrameEncoderLite(nn.Module):
    """Per-frame CNN encoder: [N, 1, H, W] in [0, 1] -> [N, d_model] tokens."""

    def __init__(
        self,
        d_model: int,
        base: int = 24,
        expand_ratio: int = 2,
        se_reduction: int = 4,
        ckpt_cnn: bool = True,
        ckpt_segments: int = 2,
    ):
        super().__init__()
        self.ckpt_cnn = bool(ckpt_cnn)
        self.ckpt_segments = int(ckpt_segments)

        def block(c_in: int, c_out: int, stride: int) -> DSConvSEBlock:
            return DSConvSEBlock(c_in, c_out, stride=stride, expand_ratio=expand_ratio, se_reduction=se_reduction)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_groups(base), base),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            block(base, base, 2),
            block(base, base * 2, 2),
            block(base * 2, base * 2, 1),
            block(base * 2, base * 4, 2),
            block(base * 4, base * 4, 1),
            block(base * 4, base * 5, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base * 5, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.ckpt_cnn and self.training:
            segments = max(1, min(self.ckpt_segments, len(self.blocks)))
            x = checkpoint_sequential(self.blocks, segments, x, use_reentrant=False)
        else:
            x = self.blocks(x)
        return self.proj(self.pool(x).flatten(1))


class RoPE1D(nn.Module):
    """Rotary position embedding applied to queries and keys."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = int(head_dim)
        self.base = float(base)

    def build_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, device=device).float() / half))
        pos = torch.arange(seq_len, device=device).float()
        ang = torch.outer(pos, inv_freq)  # [L, half]
        return ang.cos().to(dtype), ang.sin().to(dtype)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class TemporalBlock(nn.Module):
    """Pre-norm Transformer block over frame tokens [B, L, D] (RoPE on Q/K, SDPA)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1, attn_drop: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.attn_drop = float(attn_drop)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = RoPE1D(head_dim=head_dim)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.build_cos_sin(L, device=x.device, dtype=x.dtype)
        q = self.rope.apply(q, cos, sin)
        k = self.rope.apply(k, cos, sin)

        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.proj_drop(self.out(a))
        return x + self.mlp(self.norm2(x))


class EmbryoTempoFormer(nn.Module):
    """Clip [B, L, 1, H, W] in [0, 1] -> predicted developmental time [B] (hpf).

    temporal_mode:
      identity    -- use only the first frame (single-image baseline, ``cnn_single``)
      meanpool    -- average frame tokens (``meanpool``)
      transformer -- CLS token + Transformer blocks (``nocons`` and ``full``)
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        temporal_drop_p: float = 0.0,
        temporal_mode: str = "transformer",
        cnn_base: int = 24,
        cnn_expand: int = 2,
        cnn_se_reduction: int = 4,
        frame_chunk: int = 8,
        ckpt_frame: bool = False,
        ckpt_cnn: bool = True,
        ckpt_segments: int = 2,
    ):
        super().__init__()
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"Unknown temporal_mode={temporal_mode!r}; expected one of {TEMPORAL_MODES}")
        self.d_model = int(d_model)
        self.temporal_drop_p = float(temporal_drop_p)
        self.temporal_mode = str(temporal_mode)
        self.frame_chunk = int(frame_chunk)
        self.ckpt_frame = bool(ckpt_frame)

        self.frame_enc = FrameEncoderLite(
            d_model=self.d_model,
            base=int(cnn_base),
            expand_ratio=int(cnn_expand),
            se_reduction=int(cnn_se_reduction),
            ckpt_cnn=bool(ckpt_cnn),
            ckpt_segments=int(ckpt_segments),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            TemporalBlock(self.d_model, int(n_heads), float(mlp_ratio), float(dropout), float(attn_drop))
            for _ in range(int(depth))
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [B, L, 1, H, W] into tokens [B, L, D], ``frame_chunk`` frames at a time."""
        B, L, C, H, W = x.shape
        chunk = self.frame_chunk if self.frame_chunk > 0 else L
        toks = []
        for t0 in range(0, L, chunk):
            t1 = min(L, t0 + chunk)
            xs = x[:, t0:t1].reshape(B * (t1 - t0), C, H, W)
            if self.training and self.ckpt_frame:
                ts = checkpoint(self.frame_enc, xs, use_reentrant=False)
            else:
                ts = self.frame_enc(xs)
            toks.append(ts.reshape(B, t1 - t0, self.d_model))
        return torch.cat(toks, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,L,1,H,W], got {tuple(x.shape)}")
        B = x.shape[0]
        if self.temporal_mode == "identity":
            x = x[:, :1]

        tok = self._encode_frames(x)  # [B, L, D]

        # Temporal token dropout (training-time regularisation), rescaled to keep the expectation.
        if self.training and self.temporal_drop_p > 0 and tok.shape[1] > 1:
            mask = (torch.rand(B, tok.shape[1], 1, device=tok.device) > self.temporal_drop_p).to(tok.dtype)
            tok = tok * mask / (1.0 - self.temporal_drop_p)

        if self.temporal_mode == "meanpool":
            return self.head(self.norm(tok.mean(dim=1))).squeeze(-1)
        if self.temporal_mode == "identity":
            return self.head(self.norm(tok[:, 0])).squeeze(-1)

        h = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h)[:, 0]).squeeze(-1)


def build_model(cfg: Any) -> EmbryoTempoFormer:
    """Build the network from any object with TrainConfig-style attributes."""
    mp = MEM_PROFILES.get(cfg.mem_profile, MEM_PROFILES["balanced"])
    return EmbryoTempoFormer(
        d_model=cfg.model_dim,
        depth=cfg.model_depth,
        n_heads=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        dropout=cfg.drop,
        attn_drop=cfg.attn_drop,
        temporal_drop_p=cfg.temporal_drop_p,
        temporal_mode=cfg.temporal_mode,
        cnn_base=cfg.cnn_base,
        cnn_expand=cfg.cnn_expand,
        cnn_se_reduction=cfg.cnn_se_reduction,
        frame_chunk=int(mp["frame_chunk"]),
        ckpt_frame=bool(mp["ckpt_frame"]),
        ckpt_cnn=bool(mp["ckpt_cnn"]),
        ckpt_segments=int(mp["ckpt_segments"]),
    )
