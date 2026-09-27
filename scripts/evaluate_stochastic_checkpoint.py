"""Evaluate seeded ESRGAN+ variation on fixed validation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from traiNNer.archs.esrganplus_arch import ESRGANPlus
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from traiNNer.metrics.stochastic import stochastic_output_diagnostics


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(image.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


def to_image(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.squeeze(0)
        .detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (240, 28), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESRGANPlus(scale=2, noise_sigma=0.1, noise_mode="always").to(device)
    model.load_state_dict(
        load_file(str(args.checkpoint), device=str(device)), strict=True
    )
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    categories = ("text", "photo", "nature", "anime")
    for category in categories:
        lq_root = Path(f"datasets/mined_lq_x2_{category}/val/{category}")
        gt_root = Path(f"datasets/mined_hr_512_{category}/val/{category}")
        for lq_path in sorted(lq_root.glob("*.png"))[: args.samples_per_category]:
            gt_path = gt_root / f"{lq_path.stem}.webp"
            lq_image = load_rgb(lq_path)
            gt_image = load_rgb(gt_path)
            lq = to_tensor(lq_image, device)
            outputs: list[torch.Tensor] = []
            with torch.inference_mode():
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    outputs.append(model(lq))

            diagnostics = stochastic_output_diagnostics(outputs, lq)
            seed_images = [to_image(output) for output in outputs]
            psnr = [
                calculate_psnr(image, gt_image, crop_border=2) for image in seed_images
            ]
            ssim = [
                calculate_ssim(image, gt_image, crop_border=2) for image in seed_images
            ]
            pairwise = [
                np.abs(seed_images[left].astype(np.float32) - seed_images[right]).mean()
                / 255.0
                for left in range(args.seeds)
                for right in range(left + 1, args.seeds)
            ]
            result = {
                "category": category,
                "image": lq_path.name,
                **diagnostics,
                "pairwise_rgb_l1": float(np.mean(pairwise)),
                "psnr_mean": float(np.mean(psnr)),
                "psnr_std": float(np.std(psnr)),
                "ssim_mean": float(np.mean(ssim)),
                "ssim_std": float(np.std(ssim)),
            }
            all_results.append(result)

            panels = [
                add_label(image, f"seed {seed}")
                for seed, image in enumerate(seed_images)
            ]
            sheet = np.concatenate(panels, axis=1)
            cv2.imwrite(
                str(args.output / f"{category}_{lq_path.stem}_seeds.png"),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            )

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "images": all_results,
    }
    numeric_keys = [
        key for key, value in all_results[0].items() if isinstance(value, float)
    ]
    summary["mean"] = {
        key: float(np.mean([float(result[key]) for result in all_results]))
        for key in numeric_keys
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
