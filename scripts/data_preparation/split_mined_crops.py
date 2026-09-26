"""Deterministically partition mined category folders into train and val."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def is_validation(path: Path, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        f"{seed}:{path.name}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "little") / (2**64 - 1)
    return value < fraction


def split_category(root: Path, category: str, fraction: float, seed: int) -> tuple[int, int]:
    source = root / category
    train = root / "train" / category
    val = root / "val" / category
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    train_count = val_count = 0
    for path in sorted(source.glob("*.webp")):
        use_val = is_validation(path, fraction, seed)
        destination = (val if use_val else train) / path.name
        shutil.move(path, destination)
        if use_val:
            val_count += 1
        else:
            train_count += 1
    return train_count, val_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    train_count, val_count = split_category(
        args.root, args.category, args.val_fraction, args.seed
    )
    print(f"{args.category}: {train_count} train, {val_count} val")


if __name__ == "__main__":
    main()
