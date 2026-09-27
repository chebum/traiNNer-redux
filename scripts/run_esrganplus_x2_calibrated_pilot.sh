#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -s datasets/mined_hr_512_texture/manifest.jsonl
test -d datasets/mined_hr_512_texture/train/texture
test -d datasets/mined_hr_512_texture/val/texture
test -d datasets/mined_lq_x2_texture/val/texture

exec uv run python train.py \
  -opt configs/train_esrganplus_x2_calibrated_degradation_pilot.yml
