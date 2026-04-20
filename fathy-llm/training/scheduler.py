"""Warmup + cosine/linear scheduler utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SchedulerConfig:
    total_steps: int
    warmup_steps: int = 0
    min_lr_scale: float = 0.0


def _warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(max(1, warmup_steps)))


def cosine_decay_lr_lambda(step: int, config: SchedulerConfig) -> float:
    warm = _warmup_scale(step, config.warmup_steps)
    if step < config.warmup_steps:
        return warm

    decay_steps = max(1, config.total_steps - config.warmup_steps)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(config.min_lr_scale, cosine)


def linear_decay_lr_lambda(step: int, config: SchedulerConfig) -> float:
    warm = _warmup_scale(step, config.warmup_steps)
    if step < config.warmup_steps:
        return warm

    decay_steps = max(1, config.total_steps - config.warmup_steps)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    linear = 1.0 - progress
    return max(config.min_lr_scale, linear)


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_decay_lr_lambda(step, config))


def build_linear_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: linear_decay_lr_lambda(step, config))
