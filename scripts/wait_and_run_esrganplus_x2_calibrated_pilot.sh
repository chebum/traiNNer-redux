#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while [[ ! -d datasets/mined_hr_512_texture/val/texture || \
         ! -d datasets/mined_lq_x2_texture/val/texture ]]; do
  if ! systemctl --user is-active --quiet esrganplus-prepare-textures.service; then
    echo "Texture preparation stopped before producing a complete split" >&2
    exit 1
  fi
  sleep 30
done

exec /bin/bash scripts/run_esrganplus_x2_calibrated_pilot.sh
