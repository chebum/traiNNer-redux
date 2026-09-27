#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ivann/projects/esrgan_plus_train"
uv_bin="/home/ivann/.local/bin/uv"

cd "$repo_dir"

# Do not contend for the GPU with the already-running fourth ablation and its
# fixed-seed evaluation.
while systemctl --user is-active --quiet esrganplus-texture-ablation-queue.service; do
    sleep 30
done

test -f experiments/stochastic_eval/texture_exp4_nongraphic_diversity_2500/metrics.json

"$uv_bin" run python train.py --auto_resume \
    -opt configs/train_esrganplus_x2_texture_exp5_learned_gan.yml

checkpoint="experiments/2x_ESRGANplus_texture_exp5_learned_gan/models/net_g_ema_2500.safetensors"
test -f "$checkpoint"
"$uv_bin" run python -m scripts.evaluate_stochastic_checkpoint \
    --checkpoint "$checkpoint" \
    --output experiments/stochastic_eval/texture_exp5_learned_gan_2500 \
    --samples-per-category 2 \
    --seeds 4 \
    --noise-style learned_additive

# Continue the best architecture with domain-routed objectives. This remains in
# the same detached, sleep-inhibited service after the short-run evaluation.
"$uv_bin" run python train.py --auto_resume \
    -opt configs/train_esrganplus_x2_texture_best_15k.yml
