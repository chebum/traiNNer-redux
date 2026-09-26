from pathlib import Path

from PIL import Image
from scripts.data_preparation.create_clean_lq import downscale


def test_downscale_is_lossless_png_with_expected_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.webp"
    destination = tmp_path / "nested" / "lq.png"
    Image.effect_noise((512, 512), 40).convert("RGB").save(source, lossless=True)
    downscale(source, destination, 2)
    with Image.open(destination) as result:
        assert result.size == (256, 256)
        assert result.format == "PNG"


def test_downscale_rejects_nondivisible_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (511, 512)).save(source)
    try:
        downscale(source, tmp_path / "lq.png", 2)
    except ValueError as error:
        assert "not divisible" in str(error)
    else:
        raise AssertionError("Expected ValueError")
