# SSIU stochastic-texture fail-fast session

Both configs select `device: mps` and run in FP32. AMP is intentionally disabled
because the current training AMP path is CUDA-specific.

The two runs start from the same sharp deterministic checkpoint and differ only
where necessary to test stochastic texture generation:

- `train_ssiu_x2_clean_control_failfast.yml`: deterministic clean fine-tune.
- `train_ssiu_x2_clean_stochastic_failfast.yml`: zero-near-initialized
  high-frequency texture branch, two sampled forwards, and anti-collapse losses.

Both run for 15,000 iterations, validate and checkpoint every 1,000 iterations,
and use a balanced 64-image validation subset (16 per content category). Run
them sequentially on MPS so timing and memory pressure remain comparable.

```bash
uv run python train.py -opt configs/train_ssiu_x2_clean_control_failfast.yml
uv run python train.py -opt configs/train_ssiu_x2_clean_stochastic_failfast.yml
```

To keep the Mac awake and launch the stochastic run only after the control exits
successfully:

```bash
caffeinate -i zsh -c 'uv run python train.py -opt configs/train_ssiu_x2_clean_control_failfast.yml && uv run python train.py -opt configs/train_ssiu_x2_clean_stochastic_failfast.yml'
```

## Early decision points

Inspect iterations 1,000, 3,000, and 5,000 before committing to the full run.
The stochastic TensorBoard logs include:

- `stochastic_hf_delta`: seed-dependent high-frequency RGB difference;
- `stochastic_detail_hf_delta`: the same difference weighted toward locations
  where the HR target actually contains edges or texture;
- `stochastic_lf_delta`: seed-dependent low-frequency RGB difference;
- `l_g_stochastic_diversity`: nonzero while variation is below 0.5/255;
- `l_g_stochastic_low_frequency`: penalizes structural/color drift;

Only the first random sample receives the normal fidelity stack in each run.
Across iterations it is an unbiased sample from the same generator distribution;
this keeps the expected fidelity weighting aligned with the one-sample control.
The second stochastic forward is used only to estimate seed variation, so the
stochastic run uses roughly twice the generator compute per iteration.

Stop the stochastic run early if any of these hold by iteration 3,000-5,000:

- `stochastic_hf_delta` remains near zero and the diversity loss does not fall;
- visible seed differences are predominantly grain, ringing, or edge jitter;
- `stochastic_lf_delta` grows with color or geometry changes;
- fixed-seed PSNR drops by more than about 0.3 dB versus the control without a
  convincing texture improvement.

Continue to 15,000 only if high-frequency variation is content-aligned and the
fixed-seed fidelity gap remains acceptable. Always compare the same seed-zero
primary image; the extra seed images are diagnostic and are not oracle-selected.

The existing mined validation split was created per crop rather than per source
image. It is adequate for a relative A/B signal because both runs use the same
subset, but absolute benchmark numbers should not be published from it.

The newly found `clean_control_x2_best_epoch3.pt` is intentionally not used:
this trainer accepts `.pth` and `.safetensors`, and that original training
checkpoint also needs its parameter names converted. Both A/B runs use the
sharper deployment safetensors checkpoint instead.

The perceptual loss initializes torchvision VGG19 weights. Ensure those weights
are cached or that the first launch has network access before starting both
runs.

## Texture-only LPIPS restart

`train_ssiu_x2_clean_stochastic_lpips_failfast.yml` is a fresh experiment from
the original sharp deterministic checkpoint, not from the 15,000-step
stochastic model. It freezes every deterministic SSIU parameter and trains only
`texture_head.*`. Complex FFT and full-resolution pixel losses are removed;
learned LPIPS supervises perceptual plausibility, while an 8x average loss and
the existing low-frequency seed penalty protect color and geometry.

The run lasts 3,000 iterations and validates every 250. Stop early if seed
differences remain below one 8-bit level by iteration 1,000, or if the amplified
difference maps show spatially uniform grain instead of content-aligned detail.

```bash
caffeinate -i uv run python train.py -opt configs/train_ssiu_x2_clean_stochastic_lpips_failfast.yml
```
