# Weighted x2/x4 restoration comparison

Deterministic evaluation (`noise_mode=disabled`) used 100 held-out crops with
the following domain weights: 20% text, 10% anime, 24% photos, 23% nature, and
23% textures. Clean inputs use PIL bicubic downsampling; degraded inputs use
the calibrated prior with seed 1234. LPIPS-Alex is the primary metric and lower
is better.

| Use case | Best ESRGAN+ | LPIPS | Best SSIU | LPIPS | Choice |
|---|---|---:|---|---:|---|
| x2 clean | full 8000 | 0.060602 | plus-shortcuts-noise 3000 | **0.056437** | SSIU |
| x2 degraded | full 8000 | **0.137579** | degradation 2500 | 0.174267 | ESRGAN+ |
| x4 clean | clean 7500 | **0.118725** | degradation 2500 | 0.175849 | ESRGAN+ |
| x4 degraded | degradation 8000 | **0.158581** | degradation 2500 | 0.205584 | ESRGAN+ |

The x2 SSIU clean winner also has higher PSNR than ESRGAN+ (34.5251 versus
32.5041 dB). For degraded inputs, ESRGAN+ improves LPIPS over the best SSIU by
21.1% at x2 and 22.9% at x4, which is substantial enough to justify its roughly
21x larger parameter count.

If only one model is deployed per scale, use the degradation-trained ESRGAN+
step 8000 for both x2 and x4. The specialized SSIU x2 model wins only for clean
inputs; ESRGAN+ becomes the better combined choice when more than about 7.2%
of x2 traffic is degraded. At x4, ESRGAN+ wins clean and degraded LPIPS.

Full per-checkpoint and per-image JSON outputs remain local under
`benchmark_results/x2_x4_weighted_100/` because benchmark artifacts are
excluded from version control.
