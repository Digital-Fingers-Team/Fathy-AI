"""Dataset collators/loaders for pretrain, SFT, preference, and RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


@dataclass
class RLHFHyperparameters:
    """Shared knobs frequently tuned during RLHF optimization."""

    kl_penalty_weight: float = 0.02
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_bonus: float = 0.01


@dataclass
class PaddingConfig:
    pad_token_id: int = 0
    label_pad_token_id: int = -100


def _to_tensor(value: Any, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _pad_1d(items: list[Any], dtype: torch.dtype, padding_value: int | float) -> Tensor:
    tensors = [_to_tensor(item, dtype=dtype).view(-1) for item in items]
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def pretrain_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    return {
        "input_ids": _pad_1d([x["input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "attention_mask": _pad_1d([x["attention_mask"] for x in batch], torch.long, 0),
        "labels": _pad_1d([x["labels"] for x in batch], torch.long, padding.label_pad_token_id),
    }


def sft_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    return pretrain_collate_fn(batch, padding=padding)


def preference_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    return {
        "chosen_input_ids": _pad_1d([x["chosen_input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "chosen_attention_mask": _pad_1d([x["chosen_attention_mask"] for x in batch], torch.long, 0),
        "rejected_input_ids": _pad_1d([x["rejected_input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "rejected_attention_mask": _pad_1d([x["rejected_attention_mask"] for x in batch], torch.long, 0),
    }


def rl_rollout_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    collated = {
        "prompts": _pad_1d([x["prompts"] for x in batch], torch.long, padding.pad_token_id),
        "prompt_attention_mask": _pad_1d([x["prompt_attention_mask"] for x in batch], torch.long, 0),
        "responses": _pad_1d([x["responses"] for x in batch], torch.long, padding.pad_token_id),
        "response_attention_mask": _pad_1d([x["response_attention_mask"] for x in batch], torch.long, 0),
    }

    optional_float_fields = ("advantages", "returns", "old_log_probs", "values")
    for name in optional_float_fields:
        if name in batch[0]:
            collated[name] = _pad_1d([x[name] for x in batch], torch.float32, 0.0)

    return collated


def create_dataloader(
    dataset: Any,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
