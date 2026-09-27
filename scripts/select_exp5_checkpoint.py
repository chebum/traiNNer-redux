"""Rank ESRGAN+ checkpoints on a fixed clean x2 reconstruction set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor
from traiNNer.archs.esrganplus_arch import ESRGANPlus
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.squeeze(0)
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    paths = sorted(
        path
        for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    device = torch.device("cuda")
    lpips = LPIPS(net="alex").to(device).eval()
    pairs: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]] = []
    for path in paths:
        gt_pil = Image.open(path).convert("RGB")
        gt_pil = gt_pil.crop(
            (0, 0, gt_pil.width - gt_pil.width % 2, gt_pil.height - gt_pil.height % 2)
        )
        lq_pil = gt_pil.resize(
            (gt_pil.width // 2, gt_pil.height // 2), Image.Resampling.BICUBIC
        )
        gt = pil_to_tensor(gt_pil).float().div(255).unsqueeze(0).to(device)
        lq = pil_to_tensor(lq_pil).float().div(255).unsqueeze(0).to(device)
        pairs.append((lq, gt, to_uint8(gt)))

    results: list[dict[str, object]] = []
    for checkpoint in args.checkpoint:
        model = ESRGANPlus(
            scale=2,
            noise_mode="always",
            noise_style="learned_additive",
            noise_after_rrdb=False,
        ).to(device)
        model.load_state_dict(
            load_file(str(checkpoint), device=str(device)), strict=True
        )
        model.eval()
        per_image = []
        for path, (lq, gt, gt_image) in zip(paths, pairs, strict=True):
            sample_rows = []
            for sample in range(args.samples):
                torch.manual_seed(args.seed + sample)
                torch.cuda.manual_seed_all(args.seed + sample)
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    output = model(lq)
                output_image = to_uint8(output)
                with torch.inference_mode():
                    perceptual = float(lpips(output.float(), gt, normalize=True).item())
                sample_rows.append(
                    {
                        "psnr_rgb": float(
                            calculate_psnr(output_image, gt_image, crop_border=2)
                        ),
                        "ssim_rgb": float(
                            calculate_ssim(output_image, gt_image, crop_border=2)
                        ),
                        "mae_rgb": float(
                            np.abs(
                                output_image.astype(np.float32)
                                - gt_image.astype(np.float32)
                            ).mean()
                            / 255.0
                        ),
                        "lpips_alex": perceptual,
                    }
                )
            per_image.append(
                {
                    "image": path.name,
                    **{
                        metric: float(np.mean([row[metric] for row in sample_rows]))
                        for metric in (
                            "psnr_rgb",
                            "ssim_rgb",
                            "mae_rgb",
                            "lpips_alex",
                        )
                    },
                }
            )
        metrics = ("psnr_rgb", "ssim_rgb", "mae_rgb", "lpips_alex")
        result = {
            "checkpoint": str(checkpoint),
            "mean": {
                metric: float(np.mean([row[metric] for row in per_image]))
                for metric in metrics
            },
            "images": per_image,
        }
        results.append(result)
        print(json.dumps(result["mean"]), checkpoint.name, flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
