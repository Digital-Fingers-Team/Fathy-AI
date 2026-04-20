from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


class SwiGLUFFN(nn.Module):
    """Bias-free SwiGLU MLP with configurable intermediate width."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if intermediate_size is None:
            intermediate_size = _round_up_to_multiple(
                int((8 * hidden_size) / 3),
                multiple_of,
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
