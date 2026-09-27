#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ivann/projects/esrgan_plus_train"
uv_bin="/home/ivann/.local/bin/uv"

cd "$repo_dir"

# Experiment 2 is already running as its own detached service.  Wait for its
# authoritative service state before evaluating its final checkpoint.
while systemctl --user is-active --quiet esrganplus-texture-exp2.service; do
    sleep 30
done

test -f experiments/2x_ESRGANplus_texture_exp2/models/net_g_ema_2500.safetensors
"$uv_bin" run python -m scripts.evaluate_stochastic_checkpoint \
    --checkpoint experiments/2x_ESRGANplus_texture_exp2/models/net_g_ema_2500.safetensors \
    --output experiments/stochastic_eval/texture_exp2_2500 \
    --samples-per-category 2 \
    --seeds 4

# Dataset ablation of experiment 1: perceptual + GAN, photo/nature only.
"$uv_bin" run python train.py --auto_resume \
    -opt configs/train_esrganplus_x2_texture_exp3_nongraphic.yml
test -f experiments/2x_ESRGANplus_texture_exp3_nongraphic/models/net_g_ema_2500.safetensors
"$uv_bin" run python -m scripts.evaluate_stochastic_checkpoint \
    --checkpoint experiments/2x_ESRGANplus_texture_exp3_nongraphic/models/net_g_ema_2500.safetensors \
    --output experiments/stochastic_eval/texture_exp3_nongraphic_2500 \
    --samples-per-category 2 \
    --seeds 4

# Dataset ablation of experiment 2: explicit diversity, photo/nature only.
"$uv_bin" run python train.py --auto_resume \
    -opt configs/train_esrganplus_x2_texture_exp4_nongraphic_diversity.yml
test -f experiments/2x_ESRGANplus_texture_exp4_nongraphic_diversity/models/net_g_ema_2500.safetensors
"$uv_bin" run python -m scripts.evaluate_stochastic_checkpoint \
    --checkpoint experiments/2x_ESRGANplus_texture_exp4_nongraphic_diversity/models/net_g_ema_2500.safetensors \
    --output experiments/stochastic_eval/texture_exp4_nongraphic_diversity_2500 \
    --samples-per-category 2 \
    --seeds 4
