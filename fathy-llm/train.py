"""Unified entrypoint for Fathy LLM training stages.

Stages:
- all
- data
- tokenizer
- pretrain
- sft
- rlhf
- eval
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml



@dataclass
class RunContext:
    config: dict[str, Any]
    data_dir: Path
    checkpoint_dir: Path
    resume: bool
    lora: bool
    quick: bool
    wandb: bool
    wandb_project: str


def load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load shared YAML config for all stages."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping/dict.")

    return payload


def _stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    shared = config.get("shared", {})
    stage_cfg = config.get(stage, {})
    merged = {**shared, **stage_cfg}
    return merged if isinstance(merged, dict) else {}


def _sample_jsonl_to_ratio(source_file: Path, output_file: Path, ratio: float, seed: int = 13) -> int:
    rng = random.Random(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    with source_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            if rng.random() <= ratio:
                dst.write(line)
                kept += 1

    return kept


def run_data_stage(ctx: RunContext) -> dict[str, Any]:
    """Clean/process datasets. Supports quick mode by downsampling to ~1%."""
    cfg = _stage_config(ctx.config, "data")
    input_dir = Path(cfg.get("input_dir", ctx.data_dir / "raw"))
    output_dir = Path(cfg.get("output_dir", ctx.data_dir / "processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from data.preprocess import DataPreprocessor

    preprocessor = DataPreprocessor()
    results = preprocessor.process_all(str(input_dir), str(output_dir), apply_deduplication=cfg.get("dedupe", True))

    if ctx.quick:
        quick_dir = output_dir / "quick"
        quick_dir.mkdir(parents=True, exist_ok=True)
        sampled_counts: dict[str, int] = {}
        for file_path in output_dir.glob("*.jsonl"):
            kept = _sample_jsonl_to_ratio(file_path, quick_dir / file_path.name, ratio=0.01)
            sampled_counts[file_path.name] = kept
        results["quick_sampled"] = sampled_counts

    return {"stage": "data", "input_dir": str(input_dir), "output_dir": str(output_dir), "results": results}


def run_tokenizer_stage(ctx: RunContext) -> dict[str, Any]:
    """Train tokenizer from prepared text files."""
    cfg = _stage_config(ctx.config, "tokenizer")
    corpus_files = [Path(p) for p in cfg.get("files", [])]
    if not corpus_files:
        corpus_files = sorted((ctx.data_dir / "processed").glob("*.txt"))

    if not corpus_files:
        raise FileNotFoundError("Tokenizer stage requires at least one .txt corpus file.")

    from tokenizer.tokenizer import FathyTokenizer

    tokenizer = FathyTokenizer.train(
        files=corpus_files,
        vocab_size=int(cfg.get("vocab_size", 50_000 if ctx.quick else 100_000)),
        min_frequency=int(cfg.get("min_frequency", 2)),
    )

    out_path = Path(cfg.get("output_path", ctx.checkpoint_dir / "tokenizer.json"))
    tokenizer.save(out_path)
    return {
        "stage": "tokenizer",
        "files": [str(p) for p in corpus_files],
        "output_path": str(out_path),
        "quick": ctx.quick,
    }


def run_pretrain_stage(ctx: RunContext) -> dict[str, Any]:
    """Pretraining stage contract. Quick mode is interpreted as ~1% sample schedule."""
    cfg = _stage_config(ctx.config, "pretrain")
    return {
        "stage": "pretrain",
        "quick": ctx.quick,
        "data_ratio": 0.01 if ctx.quick else 1.0,
        "resume": ctx.resume,
        "checkpoint_dir": str(ctx.checkpoint_dir),
        "config": cfg,
    }


def run_sft_stage(ctx: RunContext) -> dict[str, Any]:
    """SFT stage contract. Quick mode uses short schedule + ~1% sample."""
    cfg = _stage_config(ctx.config, "sft")
    return {
        "stage": "sft",
        "quick": ctx.quick,
        "data_ratio": 0.01 if ctx.quick else 1.0,
        "resume": ctx.resume,
        "lora": ctx.lora,
        "config": cfg,
    }


def run_rlhf_stage(ctx: RunContext) -> dict[str, Any]:
    """RLHF stage contract. Quick mode limits rollout count to a smoke-sized run."""
    cfg = _stage_config(ctx.config, "rlhf")
    return {
        "stage": "rlhf",
        "quick": ctx.quick,
        "rollout_ratio": 0.01 if ctx.quick else 1.0,
        "resume": ctx.resume,
        "config": cfg,
    }


def run_eval_stage(ctx: RunContext) -> dict[str, Any]:
    """Evaluation stage contract. Quick mode runs reduced benchmark slices."""
    cfg = _stage_config(ctx.config, "eval")
    return {
        "stage": "eval",
        "quick": ctx.quick,
        "benchmark_ratio": 0.01 if ctx.quick else 1.0,
        "checkpoint_dir": str(ctx.checkpoint_dir),
        "config": cfg,
    }


def _persist_stage_result(checkpoint_dir: Path, stage_result: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_name = stage_result["stage"]
    output_path = checkpoint_dir / f"{stage_name}_stage_result.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stage_result, file, ensure_ascii=False, indent=2)


STAGE_RUNNERS = {
    "data": run_data_stage,
    "tokenizer": run_tokenizer_stage,
    "pretrain": run_pretrain_stage,
    "sft": run_sft_stage,
    "rlhf": run_rlhf_stage,
    "eval": run_eval_stage,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fathy LLM unified stage runner")
    parser.add_argument("stage", choices=["all", "data", "tokenizer", "pretrain", "sft", "rlhf", "eval"])
    parser.add_argument("--config", type=str, default=None, help="Path to shared YAML config")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint/output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest stage checkpoint")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA for SFT stage")
    parser.add_argument("--quick", action="store_true", help="Run each stage on roughly 1%% of data")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="fathy-llm", help="wandb project name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    ctx = RunContext(
        config=config,
        data_dir=Path(args.data_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        resume=args.resume,
        lora=args.lora,
        quick=args.quick,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    stage_order = ["data", "tokenizer", "pretrain", "sft", "rlhf", "eval"] if args.stage == "all" else [args.stage]

    for stage in stage_order:
        result = STAGE_RUNNERS[stage](ctx)
        if ctx.wandb:
            result["wandb"] = {"enabled": True, "project": ctx.wandb_project}
        _persist_stage_result(ctx.checkpoint_dir, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
