"""Supervised fine-tuning loop built on top of the reusable pretraining trainer."""

from __future__ import annotations

import torch

from .pretrain import PreTrainer, TrainingLoopConfig


class SFTTrainer(PreTrainer):
    """SFT trainer shares pretraining mechanics but tags checkpoints/logging as `sft`."""

    def __init__(self, model: torch.nn.Module, train_dataset, config: TrainingLoopConfig) -> None:
        super().__init__(model=model, train_dataset=train_dataset, config=config)

    def train(self) -> None:
        super().train(phase="sft")
