from __future__ import annotations

from typing import Optional
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (stable version for training + SFT)."""

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
        idx = torch.arange(0, self.dim, 2, dtype=torch.float32)
        return 1.0 / (self.base ** (idx / self.dim))

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        inv_freq = self.inv_freq.to(device)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # scaling (safe fallback)
        if self.scaling_type == "yarn":
            scale = torch.ones_like(t)
            t = t / scale

        freqs = torch.outer(t, inv_freq)  # [seq, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, dim]

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len_cached = seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        seq_len = seq_len or x.shape[-2]
        device, dtype = x.device, x.dtype

        # rebuild cache if needed
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            self._build_cache(seq_len, device, dtype)

        cos_cache = self.cos_cached
        sin_cache = self.sin_cached

        # -----------------------------
        # CASE 1: no position_ids
        # -----------------------------
        if position_ids is None:
            cos = cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            return cos, sin

        # -----------------------------
        # CASE 2: with position_ids (FIXED)
        # -----------------------------
        position_ids = position_ids.to(device)

        flat_pos = position_ids.reshape(-1)  # SAFE (NOT view)

        cos = torch.index_select(cos_cache, 0, flat_pos)
        sin = torch.index_select(sin_cache, 0, flat_pos)

        cos = cos.reshape(*position_ids.shape, self.dim)
        sin = sin.reshape(*position_ids.shape, self.dim)

        return cos.unsqueeze(1), sin.unsqueeze(1)


# -----------------------------
# helpers
# -----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k