#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

for category in text photo nature anime texture; do
  uv run python scripts/data_preparation/create_clean_lq.py \
    --input "datasets/mined_hr_512_${category}" \
    --output "datasets/mined_lq_x4_${category}" \
    --scale 4 \
    --workers 8
done
