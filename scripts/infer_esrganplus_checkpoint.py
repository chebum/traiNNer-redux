"""Run fixed-seed ESRGAN+ inference on a directory of images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from traiNNer.archs.esrganplus_arch import ESRGANPlus

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    if args.samples < 1:
        raise ValueError("--samples must be at least one")

    paths = sorted(
        path
        for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No supported images found in {args.input}")

    device = torch.device("cuda")
    model = ESRGANPlus(
        scale=2,
        noise_mode="always",
        noise_style="learned_additive",
        noise_after_rrdb=False,
    ).to(device)
    model.load_state_dict(
        load_file(str(args.checkpoint), device=str(device)), strict=True
    )
    model.eval()

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        for sample_index in range(args.samples):
            sample_seed = args.seed + sample_index
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            sample_dir = args.output / f"seed_{sample_seed}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for path in paths:
                image = pil_to_tensor(Image.open(path).convert("RGB")).float().div(255)
                start = time.perf_counter()
                result = model(image.unsqueeze(0).to(device))
                torch.cuda.synchronize()
                output_path = sample_dir / f"{path.stem}.png"
                to_pil_image(result[0].float().clamp(0, 1).cpu()).save(output_path)
                print(
                    f"seed {sample_seed}: {path.name} -> {output_path} "
                    f"({time.perf_counter() - start:.2f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
