# Fathy LLM Training Guide

## Hardware matrix

| Profile | GPU | VRAM | CPU RAM | Suggested use |
|---|---:|---:|---:|---|
| Quick smoke | CPU / single small GPU | 8-12 GB | 16 GB | `--quick` sanity checks and CI |
| Dev single-node | 1x RTX 4090 / A6000 | 24-48 GB | 64 GB | tokenizer + SFT experiments |
| Pretrain medium | 4-8x A100/H100 | 80 GB each | 256+ GB | meaningful pretraining runs |
| RLHF scale | 8x A100/H100 | 80 GB each | 256+ GB | PPO rollouts + policy updates |

## Quick start

```bash
cd fathy-llm
python train.py all \
  --config configs/model_small.yaml \
  --data-dir data \
  --checkpoint-dir checkpoints \
  --quick \
  --wandb \
  --wandb-project fathy-llm
```

Quick mode uses approximately **1%** of available data per stage for fast validation.

## Stage-by-stage

### 1) Data (`data`)
- Cleans and filters raw JSONL with `DataPreprocessor`.
- In quick mode, writes sampled files under `processed/quick/`.
- Typical runtime:
  - Quick: 2-10 minutes
  - Full: 1-6 hours (depends on corpus size)

### 2) Tokenizer (`tokenizer`)
- Trains `FathyTokenizer` on prepared text files.
- Quick mode lowers vocab target and runs on sampled corpus.
- Typical runtime:
  - Quick: 3-15 minutes
  - Full: 30-180 minutes

### 3) Pretraining (`pretrain`)
- Initializes language-model training schedule and checkpoint metadata.
- Use `--resume` to continue from latest checkpoint manifest.
- Typical runtime:
  - Quick: 5-20 minutes
  - Full: multi-day to multi-week

### 4) Supervised Fine-Tuning (`sft`)
- Runs instruction tuning behavior, optionally with `--lora`.
- Typical runtime:
  - Quick: 5-30 minutes
  - Full: 4-24 hours

### 5) RLHF (`rlhf`)
- Runs PPO-style alignment loops with reduced rollout ratios in quick mode.
- Typical runtime:
  - Quick: 10-45 minutes
  - Full: 8-72 hours

### 6) Evaluation (`eval`)
- Executes benchmark/eval stage config.
- Quick mode evaluates benchmark slices.
- Typical runtime:
  - Quick: 3-20 minutes
  - Full: 1-8 hours

## Recommended CLI patterns

```bash
# Single stage
python train.py sft --config configs/model_small.yaml --lora --quick

# Resume long-running stage
python train.py pretrain --config configs/model_large.yaml --resume

# Evaluate only
python train.py eval --config configs/model_medium.yaml --checkpoint-dir checkpoints
```
