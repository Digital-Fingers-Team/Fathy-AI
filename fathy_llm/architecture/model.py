from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .feedforward import SwiGLUFFN
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present_key_value


class FathyCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _build_position_ids(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.LongTensor:
        past_len = 0
        if past_key_values and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[-2]

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )
        return position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(input_ids, past_key_values)

        all_hidden_states = [] if output_hidden_states else None
        next_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_past = None if past_key_values is None else past_key_values[idx]

            if self.gradient_checkpointing and self.training and not use_cache:

                def custom_forward(*inputs: torch.Tensor) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
                    return layer(
                        inputs[0],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                    )

                hidden_states, _ = checkpoint(custom_forward, hidden_states, use_reentrant=False)
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )

            if use_cache:
                next_past_key_values.append(present)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_past_key_values,
            "hidden_states": all_hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.LongTensor:
        self.eval()
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            model_inputs = generated[:, -1:] if past_key_values is not None else generated
            outputs = self(
                input_ids=model_inputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if use_cache:
                past_key_values = outputs["past_key_values"]

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> "FathyCausalLM":
        checkpoint_dict = torch.load(path, map_location=map_location)
        config = ModelConfig(**checkpoint_dict["config"])
        model = cls(config)
        model.load_state_dict(checkpoint_dict["state_dict"], strict=strict)
        return model

    def to_hf_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key
            new_key = new_key.replace("embed_tokens", "model.embed_tokens")
            new_key = new_key.replace("layers", "model.layers")
            new_key = new_key.replace("input_norm", "input_layernorm")
            new_key = new_key.replace("post_attn_norm", "post_attention_layernorm")
            new_key = new_key.replace("attn", "self_attn")
            new_key = new_key.replace("norm", "model.norm") if new_key.startswith("norm") else new_key
            if new_key.startswith("lm_head"):
                pass
            mapped[new_key] = value
        return mapped

    def load_hf_state_dict(self, hf_state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        reverse_mapped: dict[str, torch.Tensor] = {}
        for key, value in hf_state_dict.items():
            new_key = key
            new_key = new_key.replace("model.embed_tokens", "embed_tokens")
            new_key = new_key.replace("model.layers", "layers")
            new_key = new_key.replace("input_layernorm", "input_norm")
            new_key = new_key.replace("post_attention_layernorm", "post_attn_norm")
            new_key = new_key.replace("self_attn", "attn")
            new_key = new_key.replace("model.norm", "norm")
            reverse_mapped[new_key] = value
        self.load_state_dict(reverse_mapped, strict=strict)
