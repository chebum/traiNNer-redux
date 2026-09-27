#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python -m scripts.data_preparation.mine_detail_crops \
  --source texture=/media/ivann/26F2-16E7/raw_images/unsplash_textures \
  --output datasets/mined_hr_512_texture \
  --crop-size 512 \
  --candidates 24 \
  --crops-per-image 2 \
  --quality 96 \
  --workers 8 \
  --seed 1234

uv run python -m scripts.data_preparation.create_clean_lq \
  --input datasets/mined_hr_512_texture \
  --output datasets/mined_lq_x2_texture \
  --scale 2 \
  --workers 8

uv run python -m scripts.data_preparation.split_mined_crops \
  --root datasets/mined_hr_512_texture \
  --paired-root datasets/mined_lq_x2_texture \
  --category texture \
  --val-fraction 0.02 \
  --seed 1234
