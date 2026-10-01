#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

for scale in 2 4; do
  test -s "experiments/${scale}x_ESRGANplus_calibrated_degradation_full_epoch/models/net_g_ema_latest.safetensors"
  test -s "experiments/${scale}x_ESRGANplus_calibrated_degradation_full_epoch/models/resume_models/net_d_latest.safetensors"
done

if [[ ! -d datasets/mined_lq_x4_texture/val/texture ]]; then
  /bin/bash scripts/data_preparation/prepare_clean_x4_pairs.sh
fi

uv run python train.py \
  -opt configs/train_esrganplus_x2_calibrated_degradation_full_epoch.yml

test -s experiments/2x_ESRGANplus_calibrated_degradation_full_epoch_v2/models/net_g_ema_latest.safetensors

uv run python train.py \
  -opt configs/train_esrganplus_x4_calibrated_degradation_full_epoch.yml
