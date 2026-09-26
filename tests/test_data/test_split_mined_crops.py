from pathlib import Path

from scripts.data_preparation.split_mined_crops import is_validation


def test_split_is_stable_and_seeded() -> None:
    paths = [Path(f"crop_{index}.webp") for index in range(1000)]
    first = [is_validation(path, 0.1, 7) for path in paths]
    assert first == [is_validation(path, 0.1, 7) for path in paths]
    assert first != [is_validation(path, 0.1, 8) for path in paths]
    assert 70 <= sum(first) <= 130
