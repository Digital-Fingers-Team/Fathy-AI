#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Distributed pretraining launch
# -----------------------------

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# Cluster / process topology
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Experiment arguments
CONFIG_PATH="${CONFIG_PATH:-fathy-llm/configs/model_small.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-fathy-llm/training/pretrain.py}"
DATA_PATH="${DATA_PATH:-/data/pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints/fathy/pretrain}"
RUN_NAME="${RUN_NAME:-fathy-pretrain}"

# Backend: torchrun or deepspeed
DIST_BACKEND="${DIST_BACKEND:-torchrun}"
QUICK="${QUICK:-0}"
QUICK_ARG=""
if [[ "${QUICK}" == "1" ]]; then
  QUICK_ARG="--quick"
fi

if [[ "${DIST_BACKEND}" == "deepspeed" ]]; then
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-fathy-llm/configs/deepspeed_pretrain_zero3.json}"
  deepspeed \
    --num_nodes "${NNODES}" \
    --num_gpus "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    --data_path "${DATA_PATH}" \
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
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    ${QUICK_ARG}
fi
