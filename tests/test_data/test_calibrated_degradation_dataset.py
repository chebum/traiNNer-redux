import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from traiNNer.data.calibrated_degradation_dataset import (
    CalibratedDegradationDataset,
    degrade_image,
    sample_jpeg_quality,
)
from traiNNer.utils.redux_options import DatasetOptions


def test_clean_replay_is_area_downsample() -> None:
    image = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    result = degrade_image(image, 2, {"degradation_probability": 0.0})
    expected = cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(result, expected)


def test_jpeg_quality_mixture_uses_only_configured_bands() -> None:
    random.seed(1234)
    options = {
        "jpeg_quality": [1, 2],
        "jpeg_quality_mixture": [
            {"quality": [75, 79], "probability": 0.15},
            {"quality": [90, 92], "probability": 0.35},
            {"quality": [93, 95], "probability": 0.37},
            {"quality": [96, 100], "probability": 0.13},
        ],
    }
    qualities = [sample_jpeg_quality(options) for _ in range(1000)]
    assert all(
        75 <= quality <= 79
        or 90 <= quality <= 92
        or 93 <= quality <= 95
        or 96 <= quality <= 100
        for quality in qualities
    )
    assert all(
        any(low <= quality <= high for quality in qualities)
        for low, high in [(75, 79), (90, 92), (93, 95), (96, 100)]
    )


def test_dataset_returns_aligned_x2_pair(tmp_path: Path) -> None:
    root = tmp_path / "texture"
    root.mkdir()
    Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8)).save(root / "sample.png")
    options = DatasetOptions(
        name="calibrated",
        type="calibrateddegradationdataset",
        dataroot_gt=[str(root)],
        phase="train",
        scale=2,
        gt_size=32,
        use_hflip=False,
        use_rot=False,
        degradation={"degradation_probability": 0.0},
    )
    sample = CalibratedDegradationDataset(options)[0]
    assert sample["gt"].shape == (3, 32, 32)
    assert sample["lq"].shape == (3, 16, 16)
    assert sample["gt_path"].endswith("sample.png")
