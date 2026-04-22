"""LoRA helper utilities used by SFT and RLHF stages."""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
)


class LoRALayer(nn.Module):
    """Wrap a ``nn.Linear`` with trainable low-rank LoRA adapters."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALayer expects nn.Linear, got {type(base_layer)!r}.")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # LoRA uses B @ A where A is down projection and B is up projection.
        self.A = nn.Parameter(torch.empty(rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Recommended initialization: A ~ Kaiming, B = 0, so the adapter starts as no-op.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        adapter_input = self.lora_dropout(x)
        adapter_output = adapter_input @ self.A.t()
        adapter_output = adapter_output @ self.B.t()
        return base_output + (adapter_output * self.scale)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return an ``nn.Linear`` with LoRA weights folded into base weights."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        merged.weight.copy_(self.base_layer.weight)
        delta_weight = (self.B @ self.A) * self.scale
        merged.weight.add_(delta_weight.to(dtype=merged.weight.dtype, device=merged.weight.device))
        if self.base_layer.bias is not None:
            merged.bias.copy_(self.base_layer.bias)
        return merged



def _set_module_by_path(root: nn.Module, module_path: str, module: nn.Module) -> None:
    parts = module_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)



def _iter_trainable_ratio(model: nn.Module) -> tuple[int, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_params, total_params



def apply_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> nn.Module:
    """Replace matching linear layers with ``LoRALayer`` and freeze non-LoRA params."""
    if not target_modules:
        raise ValueError("target_modules must be non-empty.")

    target_set = set(target_modules)
    replaced_paths: list[str] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        short_name = module_name.rsplit(".", maxsplit=1)[-1]
        if short_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Target module '{module_name}' matched, but it is {type(module)!r} not nn.Linear."
            )

        _set_module_by_path(
            root=model,
            module_path=module_name,
            module=LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced_paths.append(module_name)

    if not replaced_paths:
        raise ValueError(
            "No matching linear modules found for LoRA injection. "
            f"Looked for {sorted(target_set)}."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.A.requires_grad = True
            module.B.requires_grad = True

    trainable_params, total_params = _iter_trainable_ratio(model)
    if trainable_params == 0:
        raise RuntimeError("LoRA application resulted in zero trainable parameters.")

    LOGGER.info(
        "Applied LoRA to %d modules (rank=%d, alpha=%.3f, dropout=%.3f).",
        len(replaced_paths),
        rank,
        alpha,
        dropout,
    )
    LOGGER.info("LoRA target modules: %s", sorted(target_set))
    LOGGER.info(
        "Trainable params: %d / %d (%.4f%%).",
        trainable_params,
        total_params,
        (100.0 * trainable_params / total_params),
    )

    return model


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters into base linear weights and unwrap LoRA layers."""
    merged_paths: list[str] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, LoRALayer):
            continue

        merged_linear = module.merged_linear()
        _set_module_by_path(model, module_name, merged_linear)
        merged_paths.append(module_name)

    if not merged_paths:
        LOGGER.warning("merge_lora_weights called, but no LoRALayer modules were found.")
    else:
        LOGGER.info("Merged and removed %d LoRA modules for inference export.", len(merged_paths))

    return model
