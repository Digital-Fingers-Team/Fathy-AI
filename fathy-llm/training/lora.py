"""LoRA helper utilities used by SFT and RLHF stages."""

from __future__ import annotations

import torch


def apply_lora(
    model: torch.nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> torch.nn.Module:
    """Apply PEFT LoRA adapters when available; otherwise return the original model."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        return get_peft_model(model, peft_cfg)
    except Exception:
        return model
