"""Paired SR dataset using the calibrated camera/web degradation prior."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.utils import RNG, img2tensor
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply a real 4:2:0 JPEG encode/decode round trip."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def sample_jpeg_quality(options: dict[str, Any]) -> int:
    mixture = options.get("jpeg_quality_mixture")
    if not mixture:
        return random.randint(*options["jpeg_quality"])
    weights = [float(component["probability"]) for component in mixture]
    component = random.choices(mixture, weights=weights, k=1)[0]
    return random.randint(*component["quality"])


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def _kaiser_sinc_kernel(size: int, cutoff: float, beta: float) -> np.ndarray:
    size = _odd(size)
    x = np.arange(size, dtype=np.float32) - size // 2
    one_d = cutoff * np.sinc(cutoff * x) * np.kaiser(size, beta)
    one_d /= one_d.sum()
    kernel = np.outer(one_d, one_d)
    return (kernel / kernel.sum()).astype(np.float32)


def _degrade_legacy_image(
    hr: np.ndarray, scale: int, options: dict[str, Any]
) -> np.ndarray:
    """Apply PSF, sampling, sensor noise, and calibrated JPEG compression."""
    height, width = hr.shape[:2]
    if random.random() >= float(options.get("degradation_probability", 1.0)):
        return cv2.resize(
            hr, (width // scale, height // scale), interpolation=cv2.INTER_AREA
        )

    probabilities = options["kernel_probabilities"]
    families = list(probabilities)
    family = random.choices(
        families, weights=[probabilities[name] for name in families], k=1
    )[0]
    filtered = hr
    if family == "gaussian":
        sigma_x = random.uniform(*options["gaussian_sigma"])
        sigma_y = sigma_x * random.uniform(*options["gaussian_anisotropy"])
        size = _odd(max(3, int(2 * math.ceil(3 * max(sigma_x, sigma_y)) + 1)))
        gaussian = (
            cv2.getGaussianKernel(size, sigma_x)
            @ cv2.getGaussianKernel(size, sigma_y).T
        )
        angle = random.uniform(0, 180)
        center = ((size - 1) / 2, (size - 1) / 2)
        gaussian = cv2.warpAffine(
            gaussian, cv2.getRotationMatrix2D(center, angle, 1), (size, size)
        )
        gaussian /= gaussian.sum()
        filtered = cv2.filter2D(hr, -1, gaussian, borderType=cv2.BORDER_REFLECT_101)
    elif family == "area":
        size = _odd(random.randint(*options["area_kernel_size"]))
        if size > 1:
            filtered = cv2.blur(hr, (size, size), borderType=cv2.BORDER_REFLECT_101)
    elif family in {"kaiser", "ringing"}:
        size = random.randrange(
            options["sinc_kernel_size"][0], options["sinc_kernel_size"][1] + 1, 2
        )
        beta_range = (
            options["ringing_kaiser_beta"]
            if family == "ringing"
            else options["kaiser_beta"]
        )
        kernel = _kaiser_sinc_kernel(
            size,
            random.uniform(*options["sinc_cutoff"]),
            random.uniform(*beta_range),
        )
        filtered = cv2.filter2D(hr, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    else:
        raise ValueError(f"Unknown degradation kernel family: {family}")

    interpolation = cv2.INTER_LANCZOS4 if family == "ringing" else cv2.INTER_AREA
    lr = cv2.resize(
        filtered, (width // scale, height // scale), interpolation=interpolation
    )
    if random.random() < float(options["noise_probability"]):
        sigma = random.uniform(*options["noise_sigma"])
        noise = RNG.get_rng().normal(0.0, sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < float(options["jpeg_probability"]):
        if random.random() < float(options.get("double_jpeg_probability", 0.0)):
            lr = jpeg_roundtrip(lr, random.randint(*options["double_jpeg_quality"]))
        lr = jpeg_roundtrip(lr, sample_jpeg_quality(options))
    return lr


def motion_blur_kernel(length: float, angle: float) -> np.ndarray:
    """Rasterize a centered line PSF with fractional-pixel endpoint support."""
    radius = math.ceil(length / 2) + 1
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    positions = np.linspace(-length / 2, length / 2, max(2, math.ceil(length * 16)))
    radians = math.radians(angle)
    x = radius + positions * math.cos(radians)
    y = radius + positions * math.sin(radians)
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - left, y - top
    for offset_x, offset_y, weights in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        np.add.at(kernel, (top + offset_y, left + offset_x), weights)
    return kernel / kernel.sum()


def degrade_image(hr: np.ndarray, scale: int, options: dict[str, Any]) -> np.ndarray:
    """Synthesize camera/web degradation; v1 preserves historical run recipes."""
    version = options.get("pipeline_version", 1)
    if version == 1:
        return _degrade_legacy_image(hr, scale, options)
    if version != 2:
        raise ValueError(f"Unknown degradation pipeline version: {version}")

    height, width = hr.shape[:2]
    target_size = (width // scale, height // scale)
    if random.random() >= float(options.get("degradation_probability", 1.0)):
        return cv2.resize(hr, target_size, interpolation=cv2.INTER_AREA)

    image = hr
    if random.random() < float(options["motion_blur_probability"]):
        # Express strength in output pixels so x2/x4 have comparable softness.
        length = scale * random.uniform(*options["motion_blur_length"])
        kernel = motion_blur_kernel(length, random.uniform(0, 180))
        image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    if random.random() < float(options["pre_resize_jpeg_probability"]):
        image = jpeg_roundtrip(
            image, random.randint(*options["pre_resize_jpeg_quality"])
        )

    modes = {
        "area": cv2.INTER_AREA,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    probabilities = options["resize_probabilities"]
    mode = random.choices(
        list(probabilities), weights=list(probabilities.values()), k=1
    )[0]
    lr = cv2.resize(image, target_size, interpolation=modes[mode])
    if random.random() < float(options["noise_probability"]):
        sigma = random.uniform(*options["noise_sigma"])
        noise = RNG.get_rng().normal(0.0, sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < float(options["texture_smoothing_probability"]):
        lr = cv2.bilateralFilter(
            lr,
            d=int(options["texture_smoothing_diameter"]),
            sigmaColor=random.uniform(*options["texture_smoothing_sigma_color"]),
            sigmaSpace=random.uniform(*options["texture_smoothing_sigma_space"]),
            borderType=cv2.BORDER_REFLECT_101,
        )

    if random.random() < float(options["jpeg_probability"]):
        lr = jpeg_roundtrip(lr, sample_jpeg_quality(options))
    return lr


@DATASET_REGISTRY.register()
class CalibratedDegradationDataset(BaseDataset):
    """Generate calibrated degraded LQ/clean GT pairs from reusable HR crops."""

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        if not isinstance(opt.dataroot_gt, list):
            raise TypeError(f"dataroot_gt must be a list for dataset {opt.name}")
        if opt.degradation is None:
            raise ValueError(f"degradation must be defined for dataset {opt.name}")
        self.degradation = opt.degradation
        self.paths = sorted(
            path
            for folder in opt.dataroot_gt
            for path in Path(folder).rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in EXTENSIONS
        )
        if opt.max_dataset_size > 0:
            self.paths = self.paths[: opt.max_dataset_size]

    def __getitem__(self, index: int) -> DataFeed:
        path = self.paths[index]
        with Image.open(path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        scale = self.opt.scale
        gt_size = self.opt.gt_size
        if scale is None or gt_size is None:
            raise ValueError("scale and gt_size must be set")
        height, width = rgb.shape[:2]
        if height < gt_size or width < gt_size:
            raise ValueError(f"Image is smaller than the requested GT crop: {path}")
        top = random.randint(0, height - gt_size)
        left = random.randint(0, width - gt_size)
        gt = rgb[top : top + gt_size, left : left + gt_size]
        if self.opt.phase == "train":
            if self.opt.use_hflip and random.random() < 0.5:
                gt = np.flip(gt, axis=1)
            if self.opt.use_rot and random.random() < 0.5:
                gt = np.flip(gt, axis=0)
            if self.opt.use_rot and random.random() < 0.5:
                gt = np.rot90(gt)
        gt = np.ascontiguousarray(gt)
        lq = degrade_image(gt, scale, self.degradation)
        return {
            "lq": img2tensor(np.ascontiguousarray(lq), from_bgr=False, float32=True),
            "gt": img2tensor(gt, from_bgr=False, float32=True),
            "lq_path": str(path),
            "gt_path": str(path),
        }

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "calibrated degraded image pairs"
