import random
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image
from traiNNer.data.calibrated_degradation_dataset import (
    CalibratedDegradationDataset,
    degrade_image,
    jpeg_roundtrip,
    motion_blur_kernel,
    sample_jpeg_quality,
)
from traiNNer.utils.options import yaml_load
from traiNNer.utils.redux_options import DatasetOptions


@pytest.mark.parametrize("version", [1, 2])
def test_clean_replay_is_area_downsample(version: int) -> None:
    image = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    result = degrade_image(
        image, 2, {"pipeline_version": version, "degradation_probability": 0.0}
    )
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


def new_recipe() -> dict:
    return {
        "pipeline_version": 2,
        "degradation_probability": 1.0,
        "motion_blur_probability": 0.0,
        "pre_resize_jpeg_probability": 1.0,
        "pre_resize_jpeg_quality": [20, 20],
        "resize_probabilities": {"bicubic": 1.0},
        "noise_probability": 0.0,
        "texture_smoothing_probability": 0.0,
        "jpeg_probability": 0.0,
        "jpeg_quality": [40, 40],
    }


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("final_jpeg", [False, True])
def test_jpeg_resize_order(scale: int, final_jpeg: bool) -> None:
    image = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    options = new_recipe()
    options["jpeg_probability"] = float(final_jpeg)
    expected = cv2.resize(
        jpeg_roundtrip(image, 20),
        (64 // scale, 64 // scale),
        interpolation=cv2.INTER_CUBIC,
    )
    if final_jpeg:
        expected = jpeg_roundtrip(expected, 40)
    np.testing.assert_array_equal(degrade_image(image, scale, options), expected)


def test_motion_kernel_is_directional_and_preserves_brightness() -> None:
    kernel = motion_blur_kernel(4.0, 0.0)
    np.testing.assert_allclose(kernel.sum(), 1.0)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-7)
    coordinates = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    horizontal_variance = (kernel * coordinates[None, :] ** 2).sum()
    vertical_variance = (kernel * coordinates[:, None] ** 2).sum()
    assert horizontal_variance > 1.0
    assert vertical_variance == 0.0


@pytest.mark.parametrize("scale", [2, 4])
def test_optional_filters_surround_resize_and_jpeg(scale: int) -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    options = new_recipe()
    options.update(
        motion_blur_probability=1.0,
        motion_blur_length=[0.8, 0.8],
        texture_smoothing_probability=1.0,
        texture_smoothing_diameter=7,
        texture_smoothing_sigma_color=[20, 20],
        texture_smoothing_sigma_space=[2, 2],
        jpeg_probability=1.0,
    )
    operations = []

    def filter_image(source: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        operations.append(("motion", source.shape))
        return source

    def compress(source: np.ndarray, quality: int) -> np.ndarray:
        operations.append(("jpeg", source.shape))
        return source

    def smooth(source: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        operations.append(("smooth", source.shape))
        return source

    module = "traiNNer.data.calibrated_degradation_dataset"
    with (
        patch(f"{module}.cv2.filter2D", side_effect=filter_image),
        patch(f"{module}.jpeg_roundtrip", side_effect=compress),
        patch(f"{module}.cv2.bilateralFilter", side_effect=smooth),
    ):
        result = degrade_image(image, scale, options)
    lr_shape = (64 // scale, 64 // scale, 3)
    assert operations == [
        ("motion", image.shape),
        ("jpeg", image.shape),
        ("smooth", lr_shape),
        ("jpeg", lr_shape),
    ]
    assert result.shape == lr_shape


@pytest.mark.parametrize("scale", [2, 4])
def test_active_configs_generate_real_degraded_pairs(scale: int) -> None:
    options, _ = yaml_load(
        f"configs/train_esrganplus_x{scale}_calibrated_degradation_full_epoch.yml"
    )
    recipe = dict(options.datasets["train"].degradation)
    assert recipe["pipeline_version"] == 2
    assert "kernel_probabilities" not in recipe
    # Exercise the actual OpenCV operations, including noise and both JPEGs.
    for key in (
        "degradation_probability",
        "motion_blur_probability",
        "pre_resize_jpeg_probability",
        "noise_probability",
        "texture_smoothing_probability",
        "jpeg_probability",
    ):
        recipe[key] = 1.0
    image = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with patch(
        "traiNNer.data.calibrated_degradation_dataset.RNG.get_rng",
        return_value=np.random.default_rng(3),
    ):
        result = degrade_image(image, scale, recipe)
    assert result.shape == (64 // scale, 64 // scale, 3)
    assert result.dtype == np.uint8
