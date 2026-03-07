#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/wyj/map-anything-sdss"
EXP_ROOT="${ROOT_DIR}/astro_sdss_v2_20gb_curated/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${EXP_ROOT}/astro_sdss_v2_${TIMESTAMP}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DINO_LOCAL_REPO="${DINO_LOCAL_REPO:-/root/wyj/dinov2}"
export DINO_LOCAL_CKPT="${DINO_LOCAL_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth}"
export MASTER_PORT="${MASTER_PORT:-29688}"

mkdir -p "${EXP_DIR}"

echo "EXP_DIR=${EXP_DIR}"
echo "DINO_LOCAL_REPO=${DINO_LOCAL_REPO}"
echo "DINO_LOCAL_CKPT=${DINO_LOCAL_CKPT}"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" scripts/train.py \
  --config-name train_astro_sdss_v2 \
  hydra.run.dir="${EXP_DIR}" \
  train_params.max_num_of_imgs_per_gpu="${MAX_NUM_OF_IMGS_PER_GPU:-16}" \
  train_params.accum_iter="${ACCUM_ITER:-4}"
