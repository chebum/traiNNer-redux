import json
from pathlib import Path

import pytest
from scripts.data_preparation.split_mined_crops import is_validation, split_category


def test_split_is_stable_and_seeded() -> None:
    paths = [Path(f"crop_{index}.webp") for index in range(1000)]
    first = [is_validation(path, 0.1, 7) for path in paths]
    assert first == [is_validation(path, 0.1, 7) for path in paths]
    assert first != [is_validation(path, 0.1, 8) for path in paths]
    assert 70 <= sum(first) <= 130


def test_split_groups_crops_by_source_and_moves_lr_pair(tmp_path: Path) -> None:
    hr_root = tmp_path / "hr"
    lr_root = tmp_path / "lr"
    category = "photo"
    records = [
        {
            "source": "/source/a.jpg",
            "output": f"{category}/a_0.webp",
            "category": category,
        },
        {
            "source": "/source/a.jpg",
            "output": f"{category}/a_1.webp",
            "category": category,
        },
        {
            "source": "/source/b.jpg",
            "output": f"{category}/b_0.webp",
            "category": category,
        },
    ]
    (hr_root / category).mkdir(parents=True)
    (lr_root / category).mkdir(parents=True)
    (hr_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    for stem in ("a_0", "a_1", "b_0"):
        (hr_root / category / f"{stem}.webp").touch()
        (lr_root / category / f"{stem}.png").touch()

    split_category(hr_root, category, 0.5, 7, paired_root=lr_root)

    a_split = "val" if is_validation("/source/a.jpg", 0.5, 7) else "train"
    b_split = "val" if is_validation("/source/b.jpg", 0.5, 7) else "train"
    for stem, split in (("a_0", a_split), ("a_1", a_split), ("b_0", b_split)):
        assert (hr_root / split / category / f"{stem}.webp").is_file()
        assert (lr_root / split / category / f"{stem}.png").is_file()


def test_split_preflights_all_pairs_before_moving(tmp_path: Path) -> None:
    hr_root = tmp_path / "hr"
    lr_root = tmp_path / "lr"
    category = "photo"
    record = {
        "source": "/source/a.jpg",
        "output": f"{category}/a.webp",
        "category": category,
    }
    (hr_root / category).mkdir(parents=True)
    (lr_root / category).mkdir(parents=True)
    (hr_root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    original_hr = hr_root / category / "a.webp"
    original_hr.touch()

    with pytest.raises(FileNotFoundError):
        split_category(hr_root, category, 0.5, 7, paired_root=lr_root)

    assert original_hr.is_file()
    assert not list((hr_root / "train").rglob("*.webp"))
    assert not list((hr_root / "val").rglob("*.webp"))
