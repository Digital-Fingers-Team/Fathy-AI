from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import torch


@dataclass(slots=True)
class LayerKVCache:
    """Key/value cache for one transformer layer.

    Tensor shape convention: ``[batch, heads, seq, head_dim]``.
    """

    keys: torch.Tensor
    values: torch.Tensor

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """Append sequence positions in-place on the ``seq`` axis."""
        self.keys = append_along_sequence(self.keys, new_keys)
        self.values = append_along_sequence(self.values, new_values)

    def slice_sequence(self, start: int | None = None, end: int | None = None) -> "LayerKVCache":
        """Return sequence-sliced cache view ``[..., start:end, :]``."""
        return LayerKVCache(self.keys[..., start:end, :], self.values[..., start:end, :])

    def select_batch(self, batch_indices: torch.Tensor) -> "LayerKVCache":
        """Return batch-selected cache rows."""
        return LayerKVCache(
            self.keys.index_select(0, batch_indices),
            self.values.index_select(0, batch_indices),
        )

    def select_head(self, head_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return key/value tensors for a single head index."""
        return self.keys[:, head_idx, ...], self.values[:, head_idx, ...]


@dataclass(slots=True)
class KVCache:
    """Container for all layer caches keyed by layer index."""

    layers: Dict[int, LayerKVCache] = field(default_factory=dict)

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self.layers

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        return self.layers[layer_idx]

    def __setitem__(self, layer_idx: int, layer_cache: LayerKVCache) -> None:
        self.layers[layer_idx] = layer_cache

    def append_layer(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """Append layer cache if present; otherwise initialize the layer entry."""
        if layer_idx not in self.layers:
            self.layers[layer_idx] = LayerKVCache(new_keys, new_values)
            return
        self.layers[layer_idx].append(new_keys, new_values)

    def slice_sequence(self, start: int | None = None, end: int | None = None) -> "KVCache":
        return KVCache({layer_idx: cache.slice_sequence(start, end) for layer_idx, cache in self.layers.items()})

    def select_batch(self, batch_indices: torch.Tensor) -> "KVCache":
        return KVCache({layer_idx: cache.select_batch(batch_indices) for layer_idx, cache in self.layers.items()})

    def iter_layers(self) -> Iterable[tuple[int, LayerKVCache]]:
        return self.layers.items()


def append_along_sequence(existing: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    """Append K/V blocks along sequence dimension (``-2``)."""
    if existing is None:
        return new
    if existing.size()[:-2] != new.size()[:-2] or existing.size(-1) != new.size(-1):
        raise ValueError(
            "KV shapes are incompatible for append: "
            f"existing={tuple(existing.size())}, new={tuple(new.size())}"
        )
    return torch.cat((existing, new), dim=-2)


def slice_layer_cache(
    layer_cache: LayerKVCache,
    start: int | None = None,
    end: int | None = None,
    batch_indices: torch.Tensor | None = None,
) -> LayerKVCache:
    """Utility helper for sequence and optional batch slicing in one call."""
    sliced = layer_cache.slice_sequence(start, end)
    if batch_indices is not None:
        sliced = sliced.select_batch(batch_indices)
    return sliced
