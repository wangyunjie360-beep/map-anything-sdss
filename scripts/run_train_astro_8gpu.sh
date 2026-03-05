#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Defaults (override via env vars if needed)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29677}"
RUN_NAME="${RUN_NAME:-astro_sdss_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/astro_sdss_v2_20gb_curated/experiments}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

# For num_views=2, max_num_of_imgs_per_gpu=2 means ~1 paired sample per GPU.
MAX_NUM_OF_IMGS_PER_GPU="${MAX_NUM_OF_IMGS_PER_GPU:-2}"

mkdir -p "${HYDRA_RUN_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "[run_train_astro_8gpu] REPO_DIR=${REPO_DIR}"
echo "[run_train_astro_8gpu] CUDA_VISIBLE_DEVICES=${GPUS}"
echo "[run_train_astro_8gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[run_train_astro_8gpu] MASTER_PORT=${MASTER_PORT}"
echo "[run_train_astro_8gpu] HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "[run_train_astro_8gpu] MAX_NUM_OF_IMGS_PER_GPU=${MAX_NUM_OF_IMGS_PER_GPU}"
echo "[run_train_astro_8gpu] EXTRA_ARGS=$*"

CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --config-name train_astro_sdss \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  train_params.max_num_of_imgs_per_gpu="${MAX_NUM_OF_IMGS_PER_GPU}" \
  "$@"
