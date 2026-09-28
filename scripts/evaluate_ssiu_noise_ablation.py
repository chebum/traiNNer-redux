"""Compare deterministic SSIU outputs for short architecture ablations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.archs.ssiu_arch import SSIU
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_tensor(image: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(image.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


def tensor_to_image(value: Tensor) -> np.ndarray:
    return (
        value.squeeze(0)
        .detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def unwrap_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint root: {type(checkpoint).__name__}")
    for key in (
        "model_state_dict",
        "state_dict",
        "params_ema",
        "params",
        "model",
        "net",
    ):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            return nested
    return checkpoint


def load_model(path: Path, device: torch.device) -> SSIU:
    if path.suffix == ".safetensors":
        raw_state = load_file(str(path), device="cpu")
    elif path.suffix in (".pt", ".pth"):
        raw_state = unwrap_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    else:
        raise ValueError(f"Unsupported checkpoint extension: {path}")

    mapper = SSIU(scale=2)
    state = mapper.map_state_dict(raw_state)
    per_module = any(key.startswith("feature_noise.") for key in state)
    per_residual = any(".residual_noise." in key for key in state)
    plus_residuals = any(
        ".plus_input_projection." in key or key.endswith(".plus_long_skip_gain")
        for key in state
    )
    model = SSIU(
        scale=2,
        stochastic_feature_blocks=9 if per_module else 0,
        stochastic_internal_residuals=per_residual,
        plus_residuals=plus_residuals,
        noise_mode="disabled",
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def high_pass(value: Tensor, kernel_size: int = 5) -> Tensor:
    radius = kernel_size // 2
    low = F.avg_pool2d(
        F.pad(value, (radius,) * 4, mode="reflect"),
        kernel_size,
        stride=1,
    )
    return value - low


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (260, 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def mean_metrics(rows: list[dict[str, Any]], model_name: str) -> dict[str, float]:
    keys = ("psnr", "ssim", "lpips", "hf_l1", "hf_energy_ratio", "baseline_l1")
    selected = [row for row in rows if row["model"] == model_name]
    return {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gan-control", type=Path)
    parser.add_argument("--per-module", type=Path)
    parser.add_argument("--per-residual", type=Path)
    parser.add_argument("--plus-only", type=Path)
    parser.add_argument("--plus-noise", type=Path)
    parser.add_argument("--images", type=Path, default=Path("validation_images"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoints = {"baseline": args.baseline}
    for name, path in (
        ("gan_control", args.gan_control),
        ("per_module", args.per_module),
        ("per_residual", args.per_residual),
        ("plus_only", args.plus_only),
        ("plus_noise", args.plus_noise),
    ):
        if path is not None:
            checkpoints[name] = path
    models = {name: load_model(path, device) for name, path in checkpoints.items()}
    lpips = LPIPS(net="alex", eval_mode=True).to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    for name in models:
        (args.output / name).mkdir(exist_ok=True)
    (args.output / "sheets").mkdir(exist_ok=True)

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not image_paths:
        raise ValueError(f"No images found in {args.images}")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for image_path in image_paths:
            gt_image = load_rgb(image_path)
            gt = image_to_tensor(gt_image, device)
            lq = F.interpolate(
                gt,
                scale_factor=0.5,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            bicubic = F.interpolate(
                lq,
                size=gt.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
            outputs = {name: model(lq).clamp(0, 1) for name, model in models.items()}
            baseline = outputs["baseline"]
            gt_hf = high_pass(gt)
            gt_hf_energy = gt_hf.abs().mean().clamp_min(1e-8)

            panels = [add_label(tensor_to_image(bicubic), "bicubic")]
            for name, output in outputs.items():
                output_image = tensor_to_image(output)
                cv2.imwrite(
                    str(args.output / name / f"{image_path.stem}.png"),
                    cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR),
                )
                lpips_value = lpips(output, gt, normalize=True).mean().item()
                output_hf = high_pass(output)
                rows.append(
                    {
                        "image": image_path.name,
                        "model": name,
                        "psnr": float(
                            calculate_psnr(output_image, gt_image, crop_border=2)
                        ),
                        "ssim": float(
                            calculate_ssim(output_image, gt_image, crop_border=2)
                        ),
                        "lpips": float(lpips_value),
                        "hf_l1": float(F.l1_loss(output_hf, gt_hf).item()),
                        "hf_energy_ratio": float(
                            (output_hf.abs().mean() / gt_hf_energy).item()
                        ),
                        "baseline_l1": float(F.l1_loss(output, baseline).item()),
                    }
                )
                panels.append(add_label(output_image, name))
            panels.append(add_label(gt_image, "ground truth"))
            sheet = np.concatenate(panels, axis=1)
            cv2.imwrite(
                str(args.output / "sheets" / f"{image_path.stem}.png"),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            )

    summary = {
        "device": str(device),
        "images": len(image_paths),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "mean": {name: mean_metrics(rows, name) for name in models},
        "per_image": rows,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
