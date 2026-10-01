# x2/x4 restoration degradation recipe

The active calibrated x2/x4 full-epoch configs use
`pipeline_version: 2` in `CalibratedDegradationDataset`. The ground-truth crop is
unchanged; all synthetic degradations apply only to the low-quality input.

15% of samples retain clean area-downsample replay. For the remaining 85%, the
operation order is:

1. Optional centered directional motion blur at HR resolution (20%). Its length
   is 0.3–1.2 output pixels, multiplied by the restoration scale before filtering;
   the angle is uniform over 0–180 degrees.
2. Strong JPEG encode/decode at HR resolution (90%, quality 15–65).
3. Downsample to the target x2/x4 size using area (35%), bicubic (45%), or Lanczos
   (20%) interpolation.
4. Optional independent RGB Gaussian noise (40%, sigma 0–1.5 on the 0–255 scale).
5. Optional edge-preserving bilateral texture smoothing (30%, diameter 7,
   color sigma 8–30, spatial sigma 1–2.5 output pixels).
6. Optional final JPEG encode/decode (65%). Quality bands are 30–55 (35%), 56–80
   (45%), and 81–95 (20%).

These probabilities are conditional on entering the degraded branch. The two
JPEG decisions are independent, giving both JPEG → resize and JPEG → resize →
JPEG examples, plus some final-JPEG-only and JPEG-free examples. Both JPEG
operations use real OpenCV encoding/decoding.

There is no Gaussian, box, or sinc preblur in v2. Resizing still applies its
normal reconstruction/antialiasing filters. Bilateral smoothing approximates
texture suppression; it is not a complete camera denoiser or facial beauty filter.
The strengths and probabilities are starting settings, not fitted measurements
of the example images.

Historical saved experiment configs omit `pipeline_version` and continue to use
the original v1 recipe, including its blur and JPEG-after-resize behavior. Their
saved configs and checkpoints are preserved. Training uses the updated files in
`configs/`; evaluation must use a v2 config to assess the new degradation recipe.
Existing clean-only validation remains unchanged and does not measure restoration
of these new corruptions.

Both configs initialize generator/EMA from their respective completed restoration
run's `models/net_g_ema_latest.safetensors` and discriminator from
`models/resume_models/net_d_latest.safetensors`. They start fresh optimizer and
scheduler state (`resume_state: null`) and write to experiment names ending in
`_v2`, preserving the source checkpoint directories. The x2/x4 launcher runs
these two restoration stages directly. The temporary pilot config and launcher
have been removed.
