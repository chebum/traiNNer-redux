# SSIU matched 3k GAN factorial ablation

All trained models started from
`checkpoints/SSIU/clean_control_x2_best_epoch3.pt` and used the same data,
seed, discriminator, losses, optimizer, learning-rate schedule, and 3,000-step
budget. Evaluation uses the 14 images in `validation_images`, with stochastic
feature noise disabled.

| Model | Noise | Shortcuts | PSNR | SSIM | LPIPS | HF L1 | HF energy ratio |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| Clean checkpoint | No | No | 30.7356 | 0.906236 | 0.157776 | 0.018888 | 0.728873 |
| 3k GAN control | No | No | 31.4967 | 0.910216 | 0.081846 | 0.016859 | 0.859501 |
| Per-module noise | Yes | No | 31.5043 | **0.910422** | 0.081758 | 0.016840 | 0.859041 |
| Per-residual noise | Yes | No | 31.4927 | 0.910132 | 0.081828 | 0.016866 | 0.859768 |
| Plus shortcuts | No | Yes | 31.5006 | 0.910139 | 0.081411 | 0.016857 | **0.860442** |
| Plus shortcuts + per-module noise | Yes | Yes | **31.5094** | 0.910338 | **0.081380** | **0.016840** | 0.860043 |

The matched no-noise/no-shortcut control reduces LPIPS from 0.157776 to
0.081846 (-48.12%). This establishes that nearly all of the previously
observed improvement comes from the common GAN/perceptual fine-tuning rather
than from noise or the new shortcuts.

## Matched feature effects

| Comparison | PSNR | SSIM | LPIPS | LPIPS wins |
|---|---:|---:|---:|---:|
| Add per-module noise, no shortcuts | +0.0076 dB | +0.000206 | -0.000089 (-0.108%) | 14/14 |
| Add shortcuts, no noise | +0.0040 dB | -0.000077 | -0.000436 (-0.532%) | 13/14 |
| Add noise to shortcut model | +0.0087 dB | +0.000198 | -0.000030 (-0.037%) | 10/14 |
| Add shortcuts to per-module-noise model | +0.0051 dB | -0.000084 | -0.000377 (-0.462%) | 13/14 |

Per-module noise accounts for only 0.12% of the clean-to-control LPIPS gain,
and shortcuts account for 0.57%. The combined model improves LPIPS by 0.000466
(-0.569%) over the matched GAN control, or 0.61% of the clean-to-control gain.
The LPIPS interaction is +0.000058, so the two small gains are mildly
sub-additive rather than mutually reinforcing at 3,000 steps.

Per-residual noise is effectively neutral relative to the GAN control:
-0.000018 LPIPS (-0.023%), with -0.0039 dB PSNR.

The optional paths therefore have small, consistent effects, especially for
LPIPS, but the experiment does not support attributing the large quality jump
to either feature. Ordinary adversarial/perceptual continuation is the dominant
cause. Full per-image measurements are in `metrics.json`; rendered comparisons
are in `sheets/`.
