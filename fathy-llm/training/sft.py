"""Supervised fine-tuning loop with validation, early stopping, and optional LoRA."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from architecture.config import ModelConfig
from architecture.model import FathyCausalLM
from tokenizer.tokenizer import FathyTokenizer

from .dataset import InstructionDataset, language_modeling_collate_fn
from .lora import apply_lora
from .pretrain import PreTrainer, TrainingLoopConfig


@dataclass
class SFTConfig(TrainingLoopConfig):
    """SFT-specific config with safer defaults for instruction tuning."""

    learning_rate: float = 1e-5
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class SFTTrainer(PreTrainer):
    """SFT trainer that supports train/val datasets, eval, and early stopping."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: InstructionDataset,
        val_dataset: InstructionDataset | None,
        config: SFTConfig,
    ) -> None:
        self.phase = "sft"
        self.config = config

        if self.config.use_lora:
            model = apply_lora(
                model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )

        super().__init__(model=model, train_dataset=train_dataset, config=config)

        self.val_dataset = val_dataset
        self.val_dataloader = None
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=language_modeling_collate_fn,
            )

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        if self.config.resume_from_latest:
            latest = self._find_latest_checkpoint()
            if latest is not None:
                self.load_checkpoint(latest)

    def _find_latest_checkpoint(self) -> Path | None:
        candidates = [p for p in self.checkpoint_dir.glob("sft_step_*.pt") if p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=self._latest_step_from_name)

    def save_checkpoint(self, step: int, phase: str = "sft") -> Path:
        path = super().save_checkpoint(step=step, phase=phase)
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["early_stop_counter"] = self.early_stop_counter
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        step = super().load_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu")
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        return step

    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        if self.val_dataloader is None:
            return math.nan

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            total_loss += float(loss.detach().item())
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        self._log({"phase": self.phase, "epoch": epoch, "global_step": self.global_step, "val_loss": avg_loss})
        self.model.train()
        return avg_loss

    def train(self) -> None:
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(1, self.config.gradient_accumulation_steps)

        for epoch in range(self.config.epochs):
            for step, batch in enumerate(self.dataloader):
                self.global_step += 1
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                    outputs = self.model(**batch)
                    loss_tensor = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                    raw_loss = float(loss_tensor.detach().item())
                    loss = loss_tensor / accum_steps

                if self._is_loss_spike(raw_loss) and self._last_stable_checkpoint is not None:
                    self.load_checkpoint(self._last_stable_checkpoint)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                self._recent_losses.append(raw_loss)

                should_step = self.global_step % accum_steps == 0
                is_last_batch = step == len(self.dataloader) - 1

                if should_step or is_last_batch:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    if self.global_step % self.config.log_every == 0:
                        self._log(
                            {
                                "phase": self.phase,
                                "epoch": epoch,
                                "step": step,
                                "global_step": self.global_step,
                                "loss": raw_loss,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "grad_norm": grad_norm_value,
                            }
                        )

            if (epoch + 1) % max(1, self.config.eval_every_epochs) == 0 and self.val_dataloader is not None:
                val_loss = self.evaluate(epoch=epoch)
                improved = val_loss + self.config.early_stopping_min_delta < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.config.early_stopping_patience:
                        self.save_checkpoint(self.global_step, phase=self.phase)
                        break

            if (epoch + 1) % max(1, self.config.checkpoint_every_epochs) == 0:
                self.save_checkpoint(self.global_step, phase=self.phase)

        if self._wandb_run is not None:
            self._wandb_run.finish()


def _sample_dataset(dataset: Dataset[dict[str, Any]], ratio: float, seed: int) -> Dataset[dict[str, Any]]:
    if ratio >= 1.0:
        return dataset
    total = len(dataset)
    keep = max(1, int(total * ratio))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return Subset(dataset, indices[:keep])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT with optional validation and early stopping")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--train_data", type=str, required=True, help="Train instruction JSONL")
    parser.add_argument("--eval_data", type=str, default=None, help="Validation instruction JSONL")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML model config")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--run_name", type=str, default="fathy-sft")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume_from_latest", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use ~1% subset for train/eval")
    args, _ = parser.parse_known_args()
    return args


def _load_model_config(config_path: str | None) -> ModelConfig:
    if not config_path:
        return ModelConfig()

    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        model_cfg = payload.get("model", {}) if isinstance(payload, dict) else {}
        allowed = set(ModelConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in model_cfg.items() if k in allowed}
        return ModelConfig(**kwargs)
    except Exception:
        return ModelConfig()


def main() -> None:
    args = _parse_args()

    tokenizer = FathyTokenizer.from_file(args.tokenizer)
    model = FathyCausalLM(_load_model_config(args.config))

    train_ds: Dataset[dict[str, Any]] = InstructionDataset(
        jsonl_path=args.train_data,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    val_ds: Dataset[dict[str, Any]] | None = None
    if args.eval_data:
        val_ds = InstructionDataset(
            jsonl_path=args.eval_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
        )

    if args.quick:
        train_ds = _sample_dataset(train_ds, ratio=0.01, seed=13)
        if val_ds is not None:
            val_ds = _sample_dataset(val_ds, ratio=0.01, seed=17)

    config = SFTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.output_dir,
        eval_every_epochs=args.eval_every_epochs,
        early_stopping_patience=args.patience,
        resume_from_latest=args.resume_from_latest,
        use_lora=args.use_lora,
    )

    trainer = SFTTrainer(model=model, train_dataset=train_ds, val_dataset=val_ds, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
