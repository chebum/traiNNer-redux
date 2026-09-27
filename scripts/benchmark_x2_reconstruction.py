"""Compare x2 SR checkpoints by reconstructing bicubic-downsampled images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from spandrel import ModelLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from traiNNer.archs.esrganplus_arch import ESRGANPlus
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    paths = sorted(
        path
        for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {args.input}")

    device = torch.device("cuda")
    baseline = ModelLoader().load_from_file(args.baseline).to(device).eval()
    if baseline.scale != 2:
        raise ValueError(f"Expected x2 baseline, got x{baseline.scale}")

    trained = ESRGANPlus(
        scale=2,
        noise_mode="always",
        noise_style="learned_additive",
        noise_after_rrdb=False,
    ).to(device)
    trained.load_state_dict(
        load_file(str(args.trained), device=str(device)), strict=True
    )
    trained.eval()

    lpips = LPIPS(net="alex").to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("gt", "lq", "bicubic", "baseline", "trained"):
        (args.output / directory).mkdir(exist_ok=True)

    rows: list[dict[str, float | str]] = []
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for path in paths:
        gt_pil = Image.open(path).convert("RGB")
        width, height = gt_pil.size
        # Ensure downsampling and exact x2 reconstruction have identical dimensions.
        gt_pil = gt_pil.crop((0, 0, width - width % 2, height - height % 2))
        lq_pil = gt_pil.resize(
            (gt_pil.width // 2, gt_pil.height // 2), Image.Resampling.BICUBIC
        )
        bicubic_pil = lq_pil.resize(gt_pil.size, Image.Resampling.BICUBIC)
        lq = pil_to_tensor(lq_pil).float().div(255).unsqueeze(0).to(device)
        gt = pil_to_tensor(gt_pil).float().div(255).unsqueeze(0).to(device)

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            outputs = {
                "bicubic": pil_to_tensor(bicubic_pil)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(device),
                "baseline": baseline(lq),
                "trained": trained(lq),
            }

        gt_image = tensor_to_uint8(gt)
        to_pil_image(gt[0].cpu()).save(args.output / "gt" / f"{path.stem}.png")
        lq_pil.save(args.output / "lq" / f"{path.stem}.png")

        for model_name, output in outputs.items():
            output_image = tensor_to_uint8(output)
            to_pil_image(output[0].float().clamp(0, 1).cpu()).save(
                args.output / model_name / f"{path.stem}.png"
            )
            with torch.inference_mode():
                lpips_value = float(lpips(output.float(), gt, normalize=True).item())
            rows.append(
                {
                    "image": path.name,
                    "model": model_name,
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
                    "lpips_alex": lpips_value,
                }
            )
        print(f"processed {path.name}", flush=True)

    metrics = ("psnr_rgb", "ssim_rgb", "mae_rgb", "lpips_alex")
    summary = {
        model_name: {
            metric: float(
                np.mean(
                    [float(row[metric]) for row in rows if row["model"] == model_name]
                )
            )
            for metric in metrics
        }
        for model_name in ("bicubic", "baseline", "trained")
    }
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
