"""Mine reusable, detail-rich square crops from heterogeneous image folders.

Text/graphic crops are stored as lossless WebP. Other categories default to
high-quality WebP to reduce size without teaching the model strong compression
artifacts. A JSONL manifest records every score and source coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TEXT_CATEGORIES = {"text", "infographic", "graphics"}


@dataclass(frozen=True)
class CropScore:
    score: float
    sharpness: float
    edge_coverage: float
    entropy: float
    flat_fraction: float


@dataclass(frozen=True)
class CropRecord:
    output: str
    source: str
    category: str
    box: tuple[int, int, int, int]
    score: float
    sharpness: float
    edge_coverage: float
    entropy: float
    flat_fraction: float
    lossless: bool


def score_crop(image: Image.Image, category: str) -> CropScore:
    """Score local detail while rejecting blur and mostly-flat backgrounds."""
    thumb = np.asarray(
        ImageOps.grayscale(image).resize((160, 160), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    lap = cv2.Laplacian(thumb, cv2.CV_32F)
    sharpness = float(lap.var())
    edges = cv2.Canny(thumb, 60, 160)
    edge_coverage = float(np.count_nonzero(edges) / edges.size)
    hist = np.bincount(thumb.ravel(), minlength=256).astype(np.float64)
    probabilities = hist[hist > 0] / hist.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    local_std = cv2.blur(thumb.astype(np.float32) ** 2, (9, 9))
    local_std -= cv2.blur(thumb.astype(np.float32), (9, 9)) ** 2
    flat_fraction = float(np.mean(local_std < 12.0))

    # Text needs dense, sharp strokes; natural images benefit more from broad
    # texture coverage and tonal diversity. Flat-area penalty rejects skies,
    # bokeh, and isolated anime characters on solid backgrounds.
    if category in TEXT_CATEGORIES:
        score = (
            math.log1p(sharpness) * 1.7
            + edge_coverage * 20.0
            + entropy * 0.35
            - flat_fraction * 4.0
        )
    else:
        score = (
            math.log1p(sharpness)
            + edge_coverage * 12.0
            + entropy * 0.55
            - flat_fraction * 5.0
        )
    return CropScore(score, sharpness, edge_coverage, entropy, flat_fraction)


def candidate_boxes(
    width: int, height: int, crop_size: int, candidates: int, seed: int
) -> list[tuple[int, int, int, int]]:
    if width < crop_size or height < crop_size:
        return []
    max_x, max_y = width - crop_size, height - crop_size
    target = min(candidates, (max_x + 1) * (max_y + 1))
    boxes = {
        (x, y, x + crop_size, y + crop_size)
        for x in (0, max_x // 2, max_x)
        for y in (0, max_y // 2, max_y)
    }
    rng = random.Random(seed)
    while len(boxes) < target:
        x = rng.randint(0, max_x) if max_x else 0
        y = rng.randint(0, max_y) if max_y else 0
        boxes.add((x, y, x + crop_size, y + crop_size))
    return sorted(boxes)


def accept(score: CropScore, category: str) -> bool:
    if category in TEXT_CATEGORIES:
        return score.sharpness >= 80 and score.edge_coverage >= 0.025
    # The flat-fraction condition is particularly important for anime.
    max_flat = 0.74 if category == "anime" else 0.82
    return (
        score.sharpness >= 45
        and score.edge_coverage >= 0.018
        and score.entropy >= 4.4
        and score.flat_fraction <= max_flat
    )


def source_seed(path: Path, seed: int) -> int:
    digest = hashlib.blake2b(str(path).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") ^ seed


def mine_image(  # noqa: PLR0917
    path: Path,
    category: str,
    crop_size: int,
    candidates: int,
    crops_per_image: int,
    seed: int,
) -> list[tuple[Image.Image, tuple[int, int, int, int], CropScore]]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    ranked = []
    for box in candidate_boxes(
        image.width, image.height, crop_size, candidates, source_seed(path, seed)
    ):
        crop = image.crop(box)
        crop_score = score_crop(crop, category)
        if accept(crop_score, category):
            ranked.append((crop_score.score, crop, box, crop_score))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for _, crop, box, crop_score in ranked:
        # Avoid retaining several almost-identical overlapping crops.
        if any(
            intersection_over_union(box, old_box) > 0.55 for _, old_box, _ in selected
        ):
            continue
        selected.append((crop, box, crop_score))
        if len(selected) == crops_per_image:
            break
    return selected


def intersection_over_union(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def parse_source(value: str) -> tuple[str, Path]:
    try:
        category, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source must be CATEGORY=PATH") from error
    path = Path(raw_path).expanduser().resolve()
    if not category or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid source: {value}")
    return category.lower(), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--max-sources-per-category", type=int, default=0)
    parser.add_argument("--quality", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    rng = random.Random(args.seed)
    total = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for category, root in args.source:
            paths = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )
            rng.shuffle(paths)
            if args.max_sources_per_category > 0:
                paths = paths[: args.max_sources_per_category]
            category_dir = args.output / category
            category_dir.mkdir(exist_ok=True)

            def process(
                path: Path, active_category: str = category
            ) -> tuple[Path, list] | None:
                try:
                    crops = mine_image(
                        path,
                        active_category,
                        args.crop_size,
                        args.candidates,
                        args.crops_per_image,
                        args.seed,
                    )
                except (OSError, ValueError):
                    return None
                return path, crops

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(process, paths)
                for source_index, result in enumerate(results, start=1):
                    if result is None:
                        continue
                    path, crops = result
                    if source_index % 250 == 0:
                        print(
                            f"{category}: {source_index}/{len(paths)} sources, "
                            f"{total} total crops",
                            flush=True,
                        )
                    for index, (crop, box, crop_score) in enumerate(crops):
                        identity = f"{path}:{box}".encode()
                        stem = hashlib.blake2b(identity, digest_size=10).hexdigest()
                        output = category_dir / f"{stem}_{index}.webp"
                        lossless = category in TEXT_CATEGORIES
                        save_options = {
                            "format": "WEBP",
                            "method": 6,
                            "lossless": lossless,
                        }
                        if not lossless:
                            save_options["quality"] = args.quality
                        crop.save(output, **save_options)
                        record = CropRecord(
                            output=str(output.relative_to(args.output)),
                            source=str(path),
                            category=category,
                            box=box,
                            lossless=lossless,
                            **asdict(crop_score),
                        )
                        manifest.write(
                            json.dumps(asdict(record), sort_keys=True) + "\n"
                        )
                        total += 1
            print(f"{category}: examined {len(paths)} sources", flush=True)
    print(f"wrote {total} crops and {manifest_path}")


if __name__ == "__main__":
    main()
