from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configuration container for the Fathy transformer architecture."""

    vocab_size: int = 50_304
    max_position_embeddings: int = 4_096
    hidden_size: int = 1_024
    intermediate_size: Optional[int] = None
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling_type: Optional[str] = None  # None | "dynamic_ntk" | "yarn"
    rope_scaling_factor: float = 1.0
    rope_yarn_original_max_position_embeddings: int = 4_096
    rope_yarn_beta_fast: float = 32.0
    rope_yarn_beta_slow: float = 1.0
    rope_yarn_mscale: float = 1.0
    rope_yarn_mscale_all_dim: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_bias: bool = False
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        implied_head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.head_dim is None:
            object.__setattr__(self, "head_dim", implied_head_dim)
        elif self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        elif self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )

        if self.intermediate_size is not None and self.intermediate_size <= self.hidden_size:
            raise ValueError("intermediate_size should be greater than hidden_size")

        if self.rope_scaling_type not in {None, "dynamic_ntk", "yarn"}:
            raise ValueError(
                "rope_scaling_type must be one of None, 'dynamic_ntk', or 'yarn'"
            )
        if self.rope_scaling_factor <= 0:
            raise ValueError("rope_scaling_factor must be positive")


FATHY_SMALL = ModelConfig(
    hidden_size=768,
    num_layers=16,
    num_attention_heads=12,
    num_key_value_heads=4,
    max_position_embeddings=2_048,
)

FATHY_MEDIUM = ModelConfig(
    hidden_size=1_024,
    num_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=4_096,
)

FATHY_LARGE = ModelConfig(
    hidden_size=1_536,
    num_layers=32,
    num_attention_heads=24,
    num_key_value_heads=8,
    max_position_embeddings=8_192,
)
