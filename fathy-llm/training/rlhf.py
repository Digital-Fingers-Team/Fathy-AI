"""RLHF trainer with rollout generation and PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class RLHFConfig:
    kl_penalty_weight: float = 0.02
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_bonus: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0


class RLHFTrainer:
    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        critic_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: RLHFConfig,
        device: torch.device,
        logger: Callable[[dict], None] | None = None,
    ) -> None:
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.reward_model = reward_model.to(device)
        self.critic_model = critic_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger or (lambda _: None)

        self.reference_model.eval()
        self.reward_model.eval()

    @staticmethod
    def _gather_log_probs(logits: Tensor, token_ids: Tensor) -> Tensor:
        return F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _extract_values(critic_outputs) -> Tensor:
        if hasattr(critic_outputs, "logits"):
            return critic_outputs.logits.squeeze(-1)
        if isinstance(critic_outputs, dict):
            if "values" in critic_outputs:
                return critic_outputs["values"].squeeze(-1)
            if "logits" in critic_outputs:
                return critic_outputs["logits"].squeeze(-1)
        if isinstance(critic_outputs, Tensor):
            return critic_outputs.squeeze(-1)
        if isinstance(critic_outputs, tuple) and critic_outputs:
            return critic_outputs[0].squeeze(-1)
        raise ValueError("Critic model output must expose `values`, `logits`, or a tensor.")

    def generate_rollouts(
        self,
        prompts: Tensor,
        prompt_attention_mask: Tensor,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        generation_kwargs = generation_kwargs or {}
        prompts = prompts.to(self.device)
        prompt_attention_mask = prompt_attention_mask.to(self.device)

        with torch.no_grad():
            sequences = self.policy_model.generate(
                input_ids=prompts,
                attention_mask=prompt_attention_mask,
                **generation_kwargs,
            )

            response_len = sequences.size(1) - prompts.size(1)
            if response_len <= 0:
                raise ValueError("Generation must append at least one token to the prompt.")

            responses = sequences[:, -response_len:]
            full_attention_mask = torch.ones_like(sequences, device=self.device)
            response_attention_mask = torch.ones_like(responses, device=self.device)

            policy_logits = self.policy_model(input_ids=sequences, attention_mask=full_attention_mask).logits
            ref_logits = self.reference_model(input_ids=sequences, attention_mask=full_attention_mask).logits
            policy_logits = policy_logits[:, -response_len - 1 : -1, :]
            ref_logits = ref_logits[:, -response_len - 1 : -1, :]

            old_log_probs = self._gather_log_probs(policy_logits, responses)
            ref_log_probs = self._gather_log_probs(ref_logits, responses)
            token_kl = old_log_probs - ref_log_probs
            sequence_kl = token_kl.sum(dim=-1)

            sequence_rewards = self.reward_model(input_ids=sequences, attention_mask=full_attention_mask)
            penalized_rewards = sequence_rewards - self.config.kl_penalty_weight * sequence_kl

            values = self._extract_values(self.critic_model(input_ids=sequences, attention_mask=full_attention_mask))
            response_values = values[:, -response_len:]
            final_values = response_values[:, -1]
            advantages = penalized_rewards - final_values
            returns = advantages + final_values

        return {
            "prompts": prompts,
            "prompt_attention_mask": prompt_attention_mask,
            "responses": responses,
            "response_attention_mask": response_attention_mask,
            "sequences": sequences,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "values": response_values,
            "token_kl": token_kl,
            "sequence_kl": sequence_kl,
            "penalized_rewards": penalized_rewards,
        }

    def ppo_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        sequences = batch["sequences"].to(self.device)
        responses = batch["responses"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        response_len = responses.size(1)
        full_attention_mask = torch.ones_like(sequences, device=self.device)
        logits = self.policy_model(input_ids=sequences, attention_mask=full_attention_mask).logits
        logits = logits[:, -response_len - 1 : -1, :]

        new_log_probs = self._gather_log_probs(logits, responses)
        ratios = torch.exp((new_log_probs - old_log_probs).sum(dim=-1))

        clipped_ratios = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        value_outputs = self.critic_model(input_ids=sequences, attention_mask=full_attention_mask)
        values = self._extract_values(value_outputs)
        final_values = values[:, -1]
        value_loss = F.mse_loss(final_values, returns)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_bonus * entropy

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "mean_advantage": float(advantages.mean().item()),
            "mean_return": float(returns.mean().item()),
        }

    def train_step(
        self,
        prompts: Tensor,
        prompt_attention_mask: Tensor,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        rollout = self.generate_rollouts(prompts, prompt_attention_mask, generation_kwargs=generation_kwargs)

        metrics: dict[str, float] = {}
        for _ in range(self.config.ppo_epochs):
            metrics = self.ppo_step(rollout)

        metrics["mean_kl"] = float(rollout["sequence_kl"].mean().item())
        metrics["mean_reward"] = float(rollout["penalized_rewards"].mean().item())
        self.logger(metrics)
        return metrics
