#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Supervised fine-tuning launch
# -----------------------------

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

CONFIG_PATH="${CONFIG_PATH:-fathy-llm/configs/model_medium.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-fathy-llm/training/sft.py}"
TRAIN_DATA="${TRAIN_DATA:-/data/sft/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-/data/sft/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints/fathy/sft}"
RUN_NAME="${RUN_NAME:-fathy-sft}"
DIST_BACKEND="${DIST_BACKEND:-torchrun}"
QUICK="${QUICK:-0}"
QUICK_ARG=""
if [[ "${QUICK}" == "1" ]]; then
  QUICK_ARG="--quick"
fi

if [[ "${DIST_BACKEND}" == "deepspeed" ]]; then
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-fathy-llm/configs/deepspeed_sft_zero3.json}"
  deepspeed \
    --num_nodes "${NNODES}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --train_data "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    ${QUICK_ARG} \
    --deepspeed "${DEEPSPEED_CONFIG}"
else
  torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --train_data "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    ${QUICK_ARG}
fi
