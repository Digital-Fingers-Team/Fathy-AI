"""Reward model used for preference optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RewardModelConfig:
    hidden_size: int
    dropout: float = 0.1


class RewardModel(nn.Module):
    """Backbone + scalar reward head returning one score per sequence."""

    def __init__(self, backbone: nn.Module, config: RewardModelConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    @staticmethod
    def _extract_hidden_states(outputs) -> Tensor:
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, dict) and "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        if isinstance(outputs, tuple) and outputs:
            return outputs[0]
        raise ValueError("Backbone output does not contain `last_hidden_state`.")

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self._extract_hidden_states(outputs)

        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).clamp(min=1)
            last_indices = lengths - 1
            pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]

        return self.reward_head(pooled).squeeze(-1)


def preference_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Pairwise ranking objective: -log(sigmoid(r_chosen - r_rejected))."""

    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
