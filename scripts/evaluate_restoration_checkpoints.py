"""Compare clean and calibrated-degradation restoration quality of checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor
from traiNNer.archs.esrganplus_arch import ESRGANPlus
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.archs.ssiu_arch import SSIU
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


def parse_category_count(value: str) -> tuple[str, int]:
    try:
        category, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "category count must be CATEGORY=COUNT"
        ) from error
    if not category or count < 0:
        raise argparse.ArgumentTypeError("category count must be non-negative")
    return category, count


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
    parser.add_argument(
        "--model-type",
        choices=("esrganplus", "ssiu"),
        default="esrganplus",
    )
    parser.add_argument("--samples-per-root", type=int, default=16)
    parser.add_argument(
        "--category-count",
        type=parse_category_count,
        action="append",
        help=(
            "Seeded sample quota as CATEGORY=COUNT. Repeat for each category. "
            "Overrides --samples-per-root."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--noise-mode",
        choices=("train", "always", "disabled"),
        default="always",
        help="Generator noise behavior during evaluation.",
    )
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

    if options.network_g is None:
        raise ValueError("Config must define network_g")
    network_options = dict(options.network_g)
    network_type = str(network_options.pop("type", "")).lower()
    if args.model_type == "esrganplus" and network_type != "esrganplus":
        raise ValueError(f"Expected an ESRGANPlus generator, got {network_type!r}")
    network_options["scale"] = options.scale
    network_options["noise_mode"] = args.noise_mode

    RNG.init_rng(args.seed)
    random.seed(args.seed)
    available: dict[str, list[Path]] = {}
    for train_root in train_dataset.dataroot_gt:
        root = Path(train_root.replace("/train/", "/val/"))
        category = root.name
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        available[category] = paths

    sources: list[tuple[str, Path]] = []
    if args.category_count:
        quotas = dict(args.category_count)
        for category, count in quotas.items():
            paths = available.get(category)
            if paths is None:
                raise ValueError(f"Unknown validation category: {category}")
            if len(paths) < count:
                raise ValueError(
                    f"Requested {count} {category} images, but only {len(paths)} exist"
                )
            category_rng = random.Random(f"{args.seed}:{category}")
            sources.extend((category, path) for path in category_rng.sample(paths, count))
    else:
        for category, paths in available.items():
            sources.extend(
                (category, path) for path in paths[: args.samples_per_root]
            )

    device = torch.device("cuda")
    lpips = LPIPS(net="alex").to(device).eval()
    pairs = []
    for category, path in sources:
        with Image.open(path) as opened:
            full_image = opened.convert("RGB")
            image = np.asarray(full_image)
            clean_full = np.asarray(
                full_image.resize(
                    (full_image.width // options.scale, full_image.height // options.scale),
                    Image.Resampling.BICUBIC,
                )
            )
        gt_image = center_crop(image, 256)
        clean_image = center_crop(clean_full, 256 // options.scale)
        degraded_image = degrade_image(gt_image, options.scale, degradation)
        pairs.append((category, path.name, gt_image, clean_image, degraded_image))

    all_results: list[dict[str, object]] = []
    for label, checkpoint in args.checkpoint:
        state = load_checkpoint(checkpoint, device)
        if args.model_type == "esrganplus":
            model = ESRGANPlus(**network_options).to(device)
        else:
            runtime_keys = [key.removeprefix("module.") for key in state]
            feature_noise_blocks = {
                key.split(".")[1]
                for key in runtime_keys
                if key.startswith("feature_noise.")
            }
            model = SSIU(
                scale=options.scale,
                noise_mode=args.noise_mode,
                stochastic_feature_blocks=len(feature_noise_blocks),
                stochastic_internal_residuals=any(
                    ".residual_noise." in key for key in runtime_keys
                ),
                plus_residuals=any(
                    ".plus_input_projection." in key for key in runtime_keys
                ),
            ).to(device)
        model.load_state_dict(state, strict=True)
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
        for category in sorted({str(row["category"]) for row in rows}):
            category_rows = [row for row in rows if row["category"] == category]
            summary[category] = {
                suite: {
                    metric: float(
                        np.mean(
                            [
                                float(row[metric])
                                for row in category_rows
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
                "scale": options.scale,
                "model_type": args.model_type,
                "sample_count": len(sources),
                "summary": summary,
                "images": rows,
            }
        )
        print(label, json.dumps(summary), flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


def load_checkpoint(
    checkpoint: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        return load_file(str(checkpoint), device=str(device))
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}")
    for key in ("params_ema", "params", "state_dict", "model_state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, dict):
            loaded = candidate
            break
    if not all(isinstance(value, torch.Tensor) for value in loaded.values()):
        raise TypeError(f"Could not find a tensor state dict in {checkpoint}")
    return loaded
if __name__ == "__main__":
    main()
