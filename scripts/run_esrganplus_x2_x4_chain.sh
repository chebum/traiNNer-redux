#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

SOURCE_X4=/home/ivann/projects/AdcSR/weight/realesrgan/RealESRGAN_x4plus.pth
CONVERTED_X4=experiments/pretrained_models/RealESRGAN_x4plus_to_ESRGANplus.safetensors

test -s "$SOURCE_X4"
test -s experiments/2x_ESRGANplus_calibrated_degradation_pilot/models/net_g_ema_2500.safetensors
test -s experiments/2x_ESRGANplus_calibrated_degradation_pilot/models/resume_models/net_d_2500.safetensors

uv run python scripts/convert_realesrgan_to_esrganplus.py \
  --input "$SOURCE_X4" \
  --output "$CONVERTED_X4" \
  --verify

if [[ ! -d datasets/mined_lq_x4_texture/val/texture ]]; then
  /bin/bash scripts/data_preparation/prepare_clean_x4_pairs.sh
fi

uv run python train.py \
  -opt configs/train_esrganplus_x2_calibrated_degradation_full_epoch.yml

test -s experiments/2x_ESRGANplus_calibrated_degradation_full_epoch/models/net_g_ema_latest.safetensors

uv run python train.py \
  -opt configs/train_esrganplus_x4_clean_full_epoch.yml

test -s experiments/4x_ESRGANplus_clean_full_epoch/models/net_g_ema_latest.safetensors
test -s experiments/4x_ESRGANplus_clean_full_epoch/models/resume_models/net_d_latest.safetensors

uv run python train.py \
  -opt configs/train_esrganplus_x4_calibrated_degradation_full_epoch.yml
