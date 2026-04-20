# Fathy LLM Training Stack

This directory contains a **3-stage Claude-like training pipeline** for building Arabic-first (and multilingual) assistant checkpoints:

1. **Stage 1: Pretraining** (next-token prediction on large-scale corpora)
2. **Stage 2: SFT** (instruction tuning on curated prompts/responses)
3. **Stage 3: RLHF / preference optimization** (helpfulness + harmlessness shaping)

---

## Repository Layout

```text
fathy-llm/
├── configs/
│   ├── model_small.yaml
│   ├── model_medium.yaml
│   └── model_large.yaml
├── scripts/
│   ├── train_pretrain.sh
│   ├── train_sft.sh
│   └── train_rlhf.sh
├── tests/
│   ├── test_gqa_attention.py
│   ├── test_rmsnorm.py
│   └── test_tokenizer.py
└── requirements.txt
```

---

## 3-Stage Pipeline (Claude-like)

### 1) Pretraining

Use large, mixed-domain corpora (Arabic + English + code + web) with careful deduplication and contamination checks.

```bash
bash fathy-llm/scripts/train_pretrain.sh
```

Config knobs:
- Architecture and context size in `configs/model_*.yaml`
- Optimizer, LR schedule, batch sizing in `training`
- Safety priors in `constitutional`

### 2) Supervised Fine-Tuning (SFT)

Fine-tune from pretrain checkpoints with instruction/chat datasets that emphasize:
- factual QA
- reasoning style preferences
- Arabic dialect and MSA robustness
- refusal style consistency

```bash
bash fathy-llm/scripts/train_sft.sh
```

### 3) RLHF / Preference Optimization

Optimize policy quality using pairwise preferences and reward modeling or direct preference optimization.

```bash
bash fathy-llm/scripts/train_rlhf.sh
```

---

## Quick-Start Serving Command

After training (or with an interoperable HF checkpoint), you can quickly serve locally using Transformers:

```bash
python -m transformers.commands.serving \
  --model /checkpoints/fathy/rlhf/latest \
  --task text-generation \
  --host 0.0.0.0 \
  --port 8080
```

Alternative for vLLM (recommended for throughput):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /checkpoints/fathy/rlhf/latest \
  --dtype bfloat16 \
  --tensor-parallel-size 2
```

---

## Checkpoint and Hugging Face Interoperability

### Save in standard HF format

Make sure your trainer exports:
- `config.json`
- `model.safetensors` (or sharded safetensors)
- tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, vocab files)
- optional `generation_config.json`

### Load with `AutoModelForCausalLM`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/checkpoints/fathy/rlhf/latest"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
```

### Push to Hugging Face Hub

```bash
huggingface-cli login
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
m=AutoModelForCausalLM.from_pretrained('/checkpoints/fathy/rlhf/latest'); \
t=AutoTokenizer.from_pretrained('/checkpoints/fathy/rlhf/latest'); \
m.push_to_hub('your-org/fathy-llm'); t.push_to_hub('your-org/fathy-llm')"
```

---

## Distributed Training Guidance

### Single node (8 GPUs)

```bash
NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 bash fathy-llm/scripts/train_pretrain.sh
```

### Multi-node

Set node-specific env vars before each launch:

```bash
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

# On node 0
export NODE_RANK=0
bash fathy-llm/scripts/train_sft.sh

# On node 1
export NODE_RANK=1
bash fathy-llm/scripts/train_sft.sh
```

### Torchrun vs DeepSpeed

All scripts support `DIST_BACKEND=torchrun|deepspeed`.

- `torchrun`: simpler and good for DDP/FSDP baselines
- `deepspeed`: preferred for ZeRO-2/3 and larger models

Example:

```bash
DIST_BACKEND=deepspeed DEEPSPEED_CONFIG=fathy-llm/configs/deepspeed_rlhf_zero3.json \
  bash fathy-llm/scripts/train_rlhf.sh
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r fathy-llm/requirements.txt
```

---

## Run Tests

```bash
pytest -q fathy-llm/tests
```

> Note: architecture tests expect your codebase to expose classes like `GroupedQueryAttention` and `RMSNorm` under standard import paths (see test files).
