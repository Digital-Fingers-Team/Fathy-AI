"""Reusable training loop for LM pretraining/SFT with AMP and grad accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainingLoopConfig:
    epochs: int
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    log_every: int = 10
    checkpoint_every: int = 500
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"


class PretrainTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: TrainingLoopConfig,
        device: torch.device,
        logger: Callable[[dict], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger or (lambda _: None)

        amp_enabled = config.mixed_precision and device.type == "cuda"
        self.autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, step: int, phase: str = "pretrain") -> None:
        checkpoint = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "config": self.config.__dict__,
        }
        path = Path(self.config.checkpoint_dir) / f"{phase}_step_{step}.pt"
        torch.save(checkpoint, path)

    def train(self, dataloader: DataLoader, phase: str = "pretrain") -> None:
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        global_step = 0
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        for epoch in range(self.config.epochs):
            for step, batch in enumerate(dataloader):
                global_step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.scaler.is_enabled(),
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss / accum_steps

                self.scaler.scale(loss).backward()

                should_step = global_step % accum_steps == 0
                is_last_step = step == len(dataloader) - 1
                if should_step or is_last_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

                if global_step % self.config.log_every == 0:
                    self.logger(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "loss": float(loss.item() * accum_steps),
                        }
                    )

                if global_step % self.config.checkpoint_every == 0:
                    self._save_checkpoint(global_step, phase=phase)
