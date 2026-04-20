"""Training primitives for pretraining, SFT, reward modeling, and RLHF."""

from .dataset import PaddingConfig, RLHFHyperparameters
from .pretrain import PretrainTrainer, TrainingLoopConfig
from .reward_model import RewardModel, RewardModelConfig, preference_loss
from .rlhf import RLHFConfig, RLHFTrainer
from .scheduler import (
    SchedulerConfig,
    build_cosine_scheduler,
    build_linear_scheduler,
    cosine_decay_lr_lambda,
    linear_decay_lr_lambda,
)
from .sft import SFTTrainer

__all__ = [
    "PaddingConfig",
    "RLHFConfig",
    "RLHFHyperparameters",
    "RLHFTrainer",
    "PretrainTrainer",
    "RewardModel",
    "RewardModelConfig",
    "SFTTrainer",
    "SchedulerConfig",
    "TrainingLoopConfig",
    "build_cosine_scheduler",
    "build_linear_scheduler",
    "cosine_decay_lr_lambda",
    "linear_decay_lr_lambda",
    "preference_loss",
]
"""Training utilities for Fathy LLM."""
