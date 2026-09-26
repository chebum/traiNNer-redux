"""Create clean, losslessly stored bicubic LR pairs from mined HR crops."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


def downscale(source: Path, destination: Path, scale: int) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if image.width % scale or image.height % scale:
            raise ValueError(f"{source} dimensions are not divisible by {scale}")
        size = (image.width // scale, image.height // scale)
        result = image.resize(size, Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # LR is always lossless: lossy WebP/JPEG here would turn a clean SR task
    # into an accidental restoration task.
    result.save(destination, format="PNG", compress_level=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale < 2:
        raise ValueError("scale must be at least 2")
    paths = sorted(
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    def process(path: Path) -> None:
        relative = path.relative_to(args.input).with_suffix(".png")
        downscale(path, args.output / relative, args.scale)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(process, paths), start=1):
            if index % 1000 == 0:
                print(f"{index}/{len(paths)}", flush=True)
    print(f"wrote {len(paths)} clean x{args.scale} LR images to {args.output}")


if __name__ == "__main__":
    main()
