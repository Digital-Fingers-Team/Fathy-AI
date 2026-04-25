from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings with optional Dynamic NTK and YaRN scaling."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        yarn_original_max_position_embeddings: int = 4096,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale: float = 1.0,
        yarn_mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.yarn_original_max_position_embeddings = yarn_original_max_position_embeddings
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_mscale = yarn_mscale
        self.yarn_mscale_all_dim = yarn_mscale_all_dim

        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        return 1.0 / (self.base ** (indices / self.dim))

    def _effective_base(self, seq_len: int) -> float:
        if self.scaling_type == "dynamic_ntk" and seq_len > self.max_position_embeddings:
            scale = self.scaling_factor * seq_len / self.max_position_embeddings
            return self.base * (scale - (self.scaling_factor - 1.0)) ** (self.dim / (self.dim - 2))
        return self.base

    def _yarn_position_scale(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.scaling_type != "yarn":
            return torch.ones(seq_len, device=device, dtype=torch.float32)

        if seq_len <= self.yarn_original_max_position_embeddings:
            return torch.ones(seq_len, device=device, dtype=torch.float32)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        progress = (t - self.yarn_original_max_position_embeddings).clamp(min=0)
        progress = progress / max(seq_len - self.yarn_original_max_position_embeddings, 1)

        low = self.yarn_beta_slow
        high = self.yarn_beta_fast
        blend = low + (high - low) * progress
        return 1.0 + (blend - 1.0) * (self.scaling_factor - 1.0)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.scaling_type == "dynamic_ntk":
            base = self._effective_base(seq_len)
            idx = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            inv_freq = 1.0 / (base ** (idx / self.dim))
        else:
            inv_freq = self.inv_freq.to(device=device)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling_type == "yarn":
            position_scale = self._yarn_position_scale(seq_len, device)
            t = t / position_scale

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        if self.scaling_type == "yarn":
            mscale = self.yarn_mscale + self.yarn_mscale_all_dim * (self.dim / 128)
            emb = emb * mscale

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self.max_seq_len_cached = seq_len
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = seq_len or x.shape[-2]
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self._build_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            return cos, sin

        cos = self.cos_cached.index_select(0, position_ids.view(-1)).reshape(
            *position_ids.shape, self.dim
        )
        sin = self.sin_cached.index_select(0, position_ids.view(-1)).view(
            *position_ids.shape, self.dim
        )
        return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
