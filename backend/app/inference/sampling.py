from __future__ import annotations

from collections.abc import Iterable

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: Iterable[int] | torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty using CTRL-style token logit adjustment."""
    if penalty <= 0:
        raise ValueError("repetition penalty must be > 0")
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    token_ids = (
        generated_token_ids
        if isinstance(generated_token_ids, torch.Tensor)
        else torch.tensor(list(generated_token_ids), device=logits.device)
    )
    token_ids = token_ids.to(dtype=torch.long, device=logits.device).unique()
    if token_ids.numel() == 0:
        return adjusted

    selected = adjusted[..., token_ids]
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    adjusted[..., token_ids] = selected
    return adjusted


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_p >= 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    removal_mask = torch.zeros_like(sorted_mask)
    removal_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(removal_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_token_ids: Iterable[int] | torch.Tensor | None = None,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample/choose next token from logits of shape ``[batch, vocab]``."""
    processed = logits

    if repetition_penalty != 1.0 and generated_token_ids is not None:
        processed = apply_repetition_penalty(processed, generated_token_ids, repetition_penalty)

    if not do_sample or temperature == 0:
        return torch.argmax(processed, dim=-1)

    processed = apply_temperature(processed, temperature)
    processed = top_k_filtering(processed, top_k)
    processed = top_p_filtering(processed, top_p)

    probs = torch.softmax(processed, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
