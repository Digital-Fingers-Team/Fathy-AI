from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .positional import RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, n_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout
        self.use_flash_attention = config.use_flash_attention

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
            yarn_original_max_position_embeddings=config.rope_yarn_original_max_position_embeddings,
            yarn_beta_fast=config.rope_yarn_beta_fast,
            yarn_beta_slow=config.rope_yarn_beta_slow,
            yarn_mscale=config.rope_yarn_mscale,
            yarn_mscale_all_dim=config.rope_yarn_mscale_all_dim,
        )

    def _shape(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        bsz, seq_len, _ = tensor.shape
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), self.num_kv_heads)
        value_states = self._shape(self.v_proj(hidden_states), self.num_kv_heads)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        kv_seq_len = key_states.shape[-2] + past_len
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        next_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        use_flash = (
            self.use_flash_attention
            and hasattr(F, "scaled_dot_product_attention")
            and attention_mask is None
        )

        if use_flash:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)
            causal_mask = torch.full(
                (q_len, key_states.shape[-2]),
                torch.finfo(scores.dtype).min,
                device=scores.device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + key_states.shape[-2] - q_len)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, next_key_value
