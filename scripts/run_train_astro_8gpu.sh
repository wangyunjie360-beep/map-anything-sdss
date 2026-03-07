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

# For num_views=2, max_num_of_imgs_per_gpu=16 means ~8 paired samples per GPU.
MAX_NUM_OF_IMGS_PER_GPU="${MAX_NUM_OF_IMGS_PER_GPU:-16}"
DINO_STRICT_CHECK="${DINO_STRICT_CHECK:-1}"

mkdir -p "${HYDRA_RUN_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

if [[ "${DINO_STRICT_CHECK}" == "1" ]]; then
  : "${DINO_LOCAL_REPO:?DINO_LOCAL_REPO is required when DINO_STRICT_CHECK=1}"
  : "${DINO_LOCAL_CKPT:?DINO_LOCAL_CKPT is required when DINO_STRICT_CHECK=1}"
  if [[ ! -d "${DINO_LOCAL_REPO}" ]]; then
    echo "[run_train_astro_8gpu] ERROR: DINO_LOCAL_REPO does not exist: ${DINO_LOCAL_REPO}" >&2
    exit 1
  fi
  if [[ ! -f "${DINO_LOCAL_REPO}/hubconf.py" ]]; then
    echo "[run_train_astro_8gpu] ERROR: hubconf.py missing in DINO_LOCAL_REPO: ${DINO_LOCAL_REPO}" >&2
    exit 1
  fi
  if [[ ! -f "${DINO_LOCAL_CKPT}" ]]; then
    echo "[run_train_astro_8gpu] ERROR: DINO_LOCAL_CKPT does not exist: ${DINO_LOCAL_CKPT}" >&2
    exit 1
  fi
fi

echo "[run_train_astro_8gpu] REPO_DIR=${REPO_DIR}"
echo "[run_train_astro_8gpu] CUDA_VISIBLE_DEVICES=${GPUS}"
echo "[run_train_astro_8gpu] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[run_train_astro_8gpu] MASTER_PORT=${MASTER_PORT}"
echo "[run_train_astro_8gpu] HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "[run_train_astro_8gpu] MAX_NUM_OF_IMGS_PER_GPU=${MAX_NUM_OF_IMGS_PER_GPU}"
echo "[run_train_astro_8gpu] DINO_STRICT_CHECK=${DINO_STRICT_CHECK}"
echo "[run_train_astro_8gpu] DINO_LOCAL_REPO=${DINO_LOCAL_REPO:-<unset>}"
echo "[run_train_astro_8gpu] DINO_LOCAL_CKPT=${DINO_LOCAL_CKPT:-<unset>}"
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
