"""Compare clean and calibrated-degradation restoration quality of checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor
from traiNNer.archs.esrganplus_arch import ESRGANPlus
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.data.calibrated_degradation_dataset import degrade_image
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from traiNNer.utils.options import yaml_load
from traiNNer.utils.rng import RNG

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH") from error
    return label, Path(path)


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


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    top, left = (height - size) // 2, (width - size) // 2
    return np.ascontiguousarray(image[top : top + size, left : left + size])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=parse_checkpoint, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-root", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    options, _ = yaml_load(str(args.config))
    train_dataset = options.datasets["train"]
    if train_dataset.degradation is None or not isinstance(
        train_dataset.dataroot_gt, list
    ):
        raise ValueError("Config must define calibrated train degradation and GT roots")
    degradation = dict(train_dataset.degradation)
    degradation["degradation_probability"] = 1.0

    RNG.init_rng(args.seed)
    random.seed(args.seed)
    sources: list[tuple[str, Path]] = []
    for train_root in train_dataset.dataroot_gt:
        root = Path(train_root.replace("/train/", "/val/"))
        category = root.name
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        sources.extend((category, path) for path in paths[: args.samples_per_root])

    device = torch.device("cuda")
    lpips = LPIPS(net="alex").to(device).eval()
    pairs = []
    for category, path in sources:
        with Image.open(path) as opened:
            image = np.asarray(opened.convert("RGB"))
        gt_image = center_crop(image, 256)
        clean_image = cv2_resize_area(gt_image, 2)
        degraded_image = degrade_image(gt_image, 2, degradation)
        pairs.append((category, path.name, gt_image, clean_image, degraded_image))

    all_results: list[dict[str, object]] = []
    for label, checkpoint in args.checkpoint:
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
        rows = []
        for index, (category, name, gt_image, clean_image, degraded_image) in enumerate(
            pairs
        ):
            gt = (
                pil_to_tensor(Image.fromarray(gt_image))
                .float()
                .div(255)
                .unsqueeze(0)
                .to(device)
            )
            for suite, lq_image in (
                ("clean", clean_image),
                ("degraded", degraded_image),
            ):
                lq = (
                    pil_to_tensor(Image.fromarray(lq_image))
                    .float()
                    .div(255)
                    .unsqueeze(0)
                    .to(device)
                )
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    output = model(lq)
                output_image = to_uint8(output)
                with torch.inference_mode():
                    perceptual = float(lpips(output.float(), gt, normalize=True).item())
                rows.append(
                    {
                        "category": category,
                        "image": name,
                        "suite": suite,
                        "psnr_rgb": float(
                            calculate_psnr(output_image, gt_image, crop_border=2)
                        ),
                        "ssim_rgb": float(
                            calculate_ssim(output_image, gt_image, crop_border=2)
                        ),
                        "lpips_alex": perceptual,
                    }
                )
        summary = {}
        for suite in ("clean", "degraded"):
            selected = [row for row in rows if row["suite"] == suite]
            summary[suite] = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in ("psnr_rgb", "ssim_rgb", "lpips_alex")
            }
        text_rows = [row for row in rows if row["category"] == "text"]
        summary["text"] = {
            suite: {
                metric: float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in text_rows
                            if row["suite"] == suite
                        ]
                    )
                )
                for metric in ("psnr_rgb", "ssim_rgb", "lpips_alex")
            }
            for suite in ("clean", "degraded")
        }
        all_results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "summary": summary,
                "images": rows,
            }
        )
        print(label, json.dumps(summary), flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


def cv2_resize_area(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image, (width // scale, height // scale), interpolation=cv2.INTER_AREA
    )


if __name__ == "__main__":
    main()
