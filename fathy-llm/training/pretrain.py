"""Reusable pretraining loop with checkpointing, spike recovery, and logging."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PretrainingDataset, language_modeling_collate_fn
from .scheduler import SchedulerConfig, build_cosine_scheduler


@dataclass
class TrainingLoopConfig:
    epochs: int
    batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision_bf16: bool = True
    gradient_checkpointing: bool = False
    log_every: int = 10
    checkpoint_every: int = 1000
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    total_steps: int | None = None
    warmup_steps: int = 0
    min_lr_scale: float = 0.0
    loss_spike_window: int = 50
    loss_spike_threshold: float = 3.0
    resume_from_latest: bool = True
    use_wandb: bool = False
    wandb_project: str = "fathy-llm"
    wandb_run_name: str | None = None
    jsonl_log_path: str = "training_log.jsonl"


class PreTrainer:
    def __init__(self, model: torch.nn.Module, train_dataset: PretrainingDataset | dict[str, Any], config: TrainingLoopConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(train_dataset, PretrainingDataset):
            self.train_dataset = train_dataset
        else:
            self.train_dataset = PretrainingDataset(**train_dataset)

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=not getattr(self.train_dataset, "streaming", False),
            num_workers=self.config.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=language_modeling_collate_fn,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.total_steps = self._resolve_total_steps()
        scheduler_cfg = SchedulerConfig(
            total_steps=self.total_steps,
            warmup_steps=self.config.warmup_steps,
            min_lr_scale=self.config.min_lr_scale,
        )
        self.scheduler = build_cosine_scheduler(self.optimizer, scheduler_cfg)

        amp_enabled = self.config.mixed_precision_bf16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.autocast_dtype = torch.bfloat16
        self.autocast_enabled = amp_enabled

        self.global_step = 0
        self._recent_losses: deque[float] = deque(maxlen=max(3, self.config.loss_spike_window))
        self._last_stable_checkpoint: Path | None = None
        self._wandb_run = self._init_logging_backend()

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.resume_from_latest:
            latest = self._find_latest_checkpoint()
            if latest is not None:
                self.load_checkpoint(latest)

    def _resolve_total_steps(self) -> int:
        if self.config.total_steps is not None:
            return max(1, self.config.total_steps)

        if not hasattr(self.dataloader, "__len__"):
            raise ValueError("total_steps must be provided for datasets without a known length.")

        steps_per_epoch = max(1, len(self.dataloader))
        accum = max(1, self.config.gradient_accumulation_steps)
        optimizer_steps_per_epoch = (steps_per_epoch + accum - 1) // accum
        return max(1, self.config.epochs * optimizer_steps_per_epoch)

    def _init_logging_backend(self):
        if not self.config.use_wandb:
            return None

        try:
            import wandb  # type: ignore

            return wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, config=self.config.__dict__)
        except Exception:
            return None

    def _latest_step_from_name(self, path: Path) -> int:
        stem = path.stem
        if "_step_" not in stem:
            return -1
        try:
            return int(stem.rsplit("_step_", maxsplit=1)[-1])
        except ValueError:
            return -1

    def _find_latest_checkpoint(self) -> Path | None:
        candidates = [p for p in self.checkpoint_dir.glob("pretrain_step_*.pt") if p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=self._latest_step_from_name)

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_checkpoint(self, step: int, phase: str = "pretrain") -> Path:
        checkpoint = {
            "global_step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "rng_state": self._rng_state(),
            "config": self.config.__dict__,
        }
        path = self.checkpoint_dir / f"{phase}_step_{step}.pt"
        torch.save(checkpoint, path)
        self._last_stable_checkpoint = path
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        scaler_state = checkpoint.get("scaler")
        if scaler_state and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)
        self._restore_rng_state(checkpoint.get("rng_state"))
        self.global_step = int(checkpoint.get("global_step", 0))
        self._last_stable_checkpoint = Path(path)
        return self.global_step

    def _is_loss_spike(self, loss_value: float) -> bool:
        if len(self._recent_losses) < 3:
            return False
        baseline = median(self._recent_losses)
        return baseline > 0 and loss_value > baseline * self.config.loss_spike_threshold

    def _log(self, metrics: dict[str, Any]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
            return

        log_path = Path(self.config.jsonl_log_path)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")

    def train(self, phase: str = "pretrain") -> None:
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(1, self.config.gradient_accumulation_steps)
        last_log_time = time.perf_counter()
        last_log_step = self.global_step

        for epoch in range(self.config.epochs):
            for step, batch in enumerate(self.dataloader):
                self.global_step += 1
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    outputs = self.model(**batch)
                    raw_loss = float(outputs.loss.detach().item())
                    loss = outputs.loss / accum_steps

                if self._is_loss_spike(raw_loss) and self._last_stable_checkpoint is not None:
                    self.load_checkpoint(self._last_stable_checkpoint)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                self._recent_losses.append(raw_loss)

                should_step = self.global_step % accum_steps == 0
                is_last_batch = step == len(self.dataloader) - 1
                grad_norm_value = 0.0

                if should_step or is_last_batch:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                if self.global_step % self.config.log_every == 0:
                    now = time.perf_counter()
                    elapsed = max(1e-6, now - last_log_time)
                    step_delta = max(1, self.global_step - last_log_step)
                    tokens = int(batch["input_ids"].numel() * step_delta)
                    lr = self.optimizer.param_groups[0]["lr"]
                    gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
                    metrics = {
                        "phase": phase,
                        "epoch": epoch,
                        "step": step,
                        "global_step": self.global_step,
                        "loss": raw_loss,
                        "lr": lr,
                        "tokens_per_sec": tokens / elapsed,
                        "gpu_mem": gpu_mem,
                        "grad_norm": grad_norm_value,
                    }
                    self._log(metrics)
                    last_log_time = now
                    last_log_step = self.global_step

                if self.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(self.global_step, phase=phase)

        if self._wandb_run is not None:
            self._wandb_run.finish()


PretrainTrainer = PreTrainer
