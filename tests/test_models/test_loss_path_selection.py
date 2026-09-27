from pathlib import Path

from traiNNer.models.sr_model import select_path_indices


def test_charbonnier_path_selector_uses_only_text_samples(tmp_path: Path) -> None:
    paths = [
        str(tmp_path / "train" / "text" / "infographic.webp"),
        str(tmp_path / "train" / "photo" / "street.webp"),
    ]
    assert select_path_indices(paths, ("/text/",)) == [0]
    assert select_path_indices(paths, ("/anime/",)) == []
