"""Supervised fine-tuning loop built on top of the reusable pretraining trainer."""

from __future__ import annotations

import torch

from .pretrain import PretrainTrainer, TrainingLoopConfig


class SFTTrainer(PretrainTrainer):
    """SFT trainer shares pretraining mechanics but tags checkpoints/logging as `sft`."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: TrainingLoopConfig,
        device: torch.device,
        logger=None,
    ) -> None:
        super().__init__(model, optimizer, scheduler, config, device, logger=logger)

    def train(self, dataloader):
        return super().train(dataloader, phase="sft")
