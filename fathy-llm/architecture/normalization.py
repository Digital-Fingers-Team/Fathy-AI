from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm variant without mean centering."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm.to(input_dtype)) * self.weight
