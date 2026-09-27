"""Partition mined crops by source image and keep paired LR files aligned."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def is_validation(source_identity: str | Path, fraction: float, seed: int) -> bool:
    """Assign a source image (not an individual crop) to validation."""
    digest = hashlib.blake2b(
        f"{seed}:{source_identity}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "little") / (2**64 - 1)
    return value < fraction


def _find_crop(root: Path, category: str, filename: str) -> Path:
    candidates = [
        root / category / filename,
        root / "train" / category / filename,
        root / "val" / category / filename,
    ]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one copy of {filename} below {root}, found {existing}"
        )
    return existing[0]


def split_category(
    root: Path,
    category: str,
    fraction: float,
    seed: int,
    paired_root: Path | None = None,
) -> tuple[int, int]:
    """Move HR crops and optional paired LR images using source-level assignment."""
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing crop manifest: {manifest_path}")

    records = []
    with manifest_path.open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            if record.get("category") == category:
                records.append(record)

    moves: list[tuple[Path, Path]] = []
    assignments: list[bool] = []
    for record in records:
        filename = Path(record["output"]).name
        use_val = is_validation(record["source"], fraction, seed)
        split = "val" if use_val else "train"
        source = _find_crop(root, category, filename)
        destination = root / split / category / filename
        moves.append((source, destination))

        if paired_root is not None:
            paired_filename = Path(filename).with_suffix(".png").name
            paired_source = _find_crop(paired_root, category, paired_filename)
            paired_destination = paired_root / split / category / paired_filename
            moves.append((paired_source, paired_destination))

        assignments.append(use_val)

    # Resolve and validate the complete migration before changing the dataset.
    # This prevents a missing pair late in the manifest from causing a partial split.
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination and destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing crop: {destination}")
    for source, destination in moves:
        if source != destination:
            shutil.move(source, destination)

    train_count = val_count = 0
    for use_val in assignments:
        if use_val:
            val_count += 1
        else:
            train_count += 1
    return train_count, val_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    train_count, val_count = split_category(
        args.root, args.category, args.val_fraction, args.seed, args.paired_root
    )
    print(f"{args.category}: {train_count} train, {val_count} val")


if __name__ == "__main__":
    main()
