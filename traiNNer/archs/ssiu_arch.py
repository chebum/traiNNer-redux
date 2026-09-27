"""Structural Similarity-Inspired Unfolding (SSIU) for image SR.

The deterministic module names match the locally deployed SSIU checkpoints.
The optional stochastic path is a zero-initialized, high-pass texture residual,
so enabling it does not change the pretrained model's initial output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY

NoiseMode = Literal["train", "always", "disabled"]


class StochasticTextureHead(nn.Module):
    """Predict a zero-mean HR texture residual from features and spatial noise."""

    def __init__(
        self,
        *,
        channels: int,
        out_channels: int,
        scale: int,
        noise_channels: int,
        noise_mode: NoiseMode,
        highpass_kernel: int,
        init_std: float,
    ) -> None:
        super().__init__()
        if noise_mode not in ("train", "always", "disabled"):
            raise ValueError(f"Unsupported noise_mode: {noise_mode}")
        if highpass_kernel < 3 or highpass_kernel % 2 == 0:
            raise ValueError("highpass_kernel must be an odd integer of at least 3")
        self.noise_channels = noise_channels
        self.noise_mode = noise_mode
        self.highpass_kernel = highpass_kernel
        self.out_channels = out_channels
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(channels + noise_channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * scale * scale, 1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        # A tiny non-zero initialization gives the diversity objective a
        # gradient immediately while remaining visually indistinguishable from
        # the deterministic checkpoint at step zero.
        final_conv = self.body[-1]
        assert isinstance(final_conv, nn.Conv2d)
        nn.init.normal_(final_conv.weight, std=init_std)
        assert final_conv.bias is not None
        nn.init.zeros_(final_conv.bias)

    def forward(self, features: Tensor) -> Tensor:
        enabled = self.noise_mode == "always" or (
            self.noise_mode == "train" and self.training
        )
        if not enabled:
            height, width = features.shape[-2:]
            return features.new_zeros(
                features.shape[0],
                self.out_channels,
                height * self.scale,
                width * self.scale,
            )
        noise = torch.randn(
            features.shape[0],
            self.noise_channels,
            *features.shape[-2:],
            device=features.device,
            dtype=features.dtype,
        )
        residual = self.body(torch.cat((features, noise), dim=1))
        radius = self.highpass_kernel // 2
        low_frequency = F.avg_pool2d(
            F.pad(residual, (radius, radius, radius, radius), mode="reflect"),
            self.highpass_kernel,
            stride=1,
        )
        return residual - low_frequency


@ARCH_REGISTRY.register()
class SSIU(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        scale: int = 2,
        colors: int = 3,
        n_feats: int = 64,
        n_blocks: int = 9,
        stochastic: bool = False,
        noise_channels: int = 8,
        noise_mode: NoiseMode = "train",
        highpass_kernel: int = 5,
        texture_init_std: float = 1e-5,
        train_texture_only: bool = False,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError("scale must be 2, 3, or 4")
        if n_blocks < 3:
            raise ValueError("n_blocks must be at least 3")
        self.scale = scale
        self.n_blocks = n_blocks
        self.train_texture_only = train_texture_only
        self.expert_indices = {
            max(1, round(n_blocks * fraction / 3)) for fraction in (1, 2, 3)
        }
        self.window_sizes = (8, 16)
        self.head = nn.Conv2d(
            colors, n_feats, 3, padding=1, padding_mode="reflect"
        )
        self.body = nn.ModuleList(
            SSIURecurrentModule(
                n_feats,
                num_heads=4,
                block_size=8,
                kernel_size=3,
                reduction=2,
            )
            for _ in range(n_blocks)
        )
        self.moe = MixtureOfExperts(n_feats)
        if scale == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, n_feats * 4, 1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(
                    n_feats, colors, 3, padding=1, padding_mode="reflect"
                ),
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * scale * scale, 1),
                nn.PixelShuffle(scale),
                nn.GELU(),
                nn.Conv2d(
                    n_feats, colors, 3, padding=1, padding_mode="reflect"
                ),
            )
        self.texture_head = (
            StochasticTextureHead(
                channels=n_feats,
                out_channels=colors,
                scale=scale,
                noise_channels=noise_channels,
                noise_mode=noise_mode,
                highpass_kernel=highpass_kernel,
                init_std=texture_init_std,
            )
            if stochastic
            else None
        )
        if train_texture_only:
            if self.texture_head is None:
                raise ValueError("train_texture_only requires stochastic: true")
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.startswith("texture_head.")

    def is_parameter_intentionally_frozen(self, name: str) -> bool:
        return self.train_texture_only and not name.startswith("texture_head.")

    def forward(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        padded = self._pad(image)
        shallow = self.head(padded)
        features = shallow
        expert_features: list[Tensor] = []
        for index, block in enumerate(self.body, start=1):
            features = block(features, shallow)
            if index in self.expert_indices:
                expert_features.append(features)
        features = self.moe(expert_features) + shallow
        result = self.tail(features) + F.interpolate(
            padded,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )
        if self.texture_head is not None:
            result = result + self.texture_head(features)
        return result[..., : height * self.scale, : width * self.scale]

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        incompatible = super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        if self.texture_head is not None and not strict:
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("texture_head.")
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Stochastic SSIU checkpoint mismatch outside texture_head: "
                    f"missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        return incompatible

    def _pad(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        multiple = math.lcm(*self.window_sizes)
        pad_height = (-height) % multiple
        pad_width = (-width) % multiple
        if not pad_height and not pad_width:
            return image
        return F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")


class MixtureOfExperts(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight_a = nn.Conv2d(channels, channels, 1)
        self.weight_b = nn.Conv2d(channels, channels, 1)
        self.weight_c = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) == 4:
            a, b, c = features[1:]
        elif len(features) == 3:
            a, b, c = features
        else:
            raise ValueError(
                f"SSIU mixture expects 3 or 4 features, received {len(features)}"
            )
        weights = torch.stack(
            (self.weight_a(a), self.weight_b(b), self.weight_c(c))
        ).softmax(dim=0)
        result = a * weights[0] + b * weights[1] + c * weights[2]
        return self.fuse(result) + result


class SSIURecurrentModule(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        block_size: int,
        kernel_size: int,
        reduction: int,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.sparse_constraint = MixedScaleGating(channels, kernel_size=kernel_size)
        self.similarity_constraint = MixedScaleGating(
            channels, kernel_size=kernel_size
        )
        self.attention = EfficientSparseAttention(
            channels,
            block_size=block_size,
            halo_size=1,
            num_heads=num_heads,
            kernel_size=3,
            reduction=reduction,
        )
        self.aggregate = MixedScaleGating(channels, kernel_size=kernel_size)
        self.feed_forward = MixedScaleGating(channels, kernel_size=kernel_size)

    def forward(self, features: Tensor, shallow: Tensor) -> Tensor:
        normalized = self.norm(features)
        sparse = self.sparse_constraint(normalized)
        similar = self.similarity_constraint(normalized)
        attended_input = normalized + sparse + shallow
        attended = self.attention(attended_input) + attended_input
        features = self.aggregate(attended - similar) + similar
        return self.feed_forward(features) + features


class EfficientSparseAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        block_size: int,
        halo_size: int,
        num_heads: int,
        kernel_size: int,
        reduction: int,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.reduction = reduction
        if reduction > 1:
            self.sampler = nn.MaxPool2d(2, reduction)
            self.local_propagation = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                    padding_mode="reflect",
                ),
                Interpolate(
                    scale_factor=reduction, mode="bilinear", align_corners=True
                ),
            )
        neighborhood = block_size + 2 * halo_size
        half_head = self.head_channels // 2
        self.relative_height = nn.Parameter(
            torch.randn(1, neighborhood, 1, half_head)
        )
        self.relative_width = nn.Parameter(
            torch.randn(1, 1, neighborhood, half_head)
        )
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        original_height, original_width = features.shape[-2:]
        if self.reduction > 1:
            features = self.sampler(features)
        batch, channels, height, width = features.shape
        block, halo = self.block_size, self.halo_size
        pad_right, pad_bottom = (-width) % block, (-height) % block
        if pad_right or pad_bottom:
            features = F.pad(
                features, (0, pad_right, 0, pad_bottom), mode="reflect"
            )
        height, width = features.shape[-2:]
        height_blocks, width_blocks = height // block, width // block

        query, key, value = self.qkv(features).chunk(3, dim=1)
        query = query.reshape(
            batch, channels, height_blocks, block, width_blocks, block
        )
        query = query.permute(0, 2, 4, 3, 5, 1).reshape(
            -1, block * block, channels
        )
        query = query * self.head_channels**-0.5
        neighborhood = block + 2 * halo
        key = self._unfold_windows(key, neighborhood, block, halo, channels)
        value = self._unfold_windows(value, neighborhood, block, halo, channels)
        windows = key.shape[0]

        query = query.reshape(
            windows, block * block, self.num_heads, self.head_channels
        )
        query = query.permute(0, 2, 1, 3).reshape(
            -1, block * block, self.head_channels
        )
        value = value.reshape(
            windows,
            neighborhood * neighborhood,
            self.num_heads,
            self.head_channels,
        )
        value = value.permute(0, 2, 1, 3).reshape(
            -1, neighborhood * neighborhood, self.head_channels
        )
        key = key.reshape(
            windows,
            neighborhood,
            neighborhood,
            self.num_heads,
            self.head_channels,
        )
        key = key.permute(0, 3, 1, 2, 4).reshape(
            -1, neighborhood, neighborhood, self.head_channels
        )
        key_height, key_width = key.split(self.head_channels // 2, dim=-1)
        key = torch.cat(
            (
                key_height + self.relative_height,
                key_width + self.relative_width,
            ),
            dim=-1,
        )
        key = key.reshape(-1, neighborhood * neighborhood, self.head_channels)

        attention = torch.einsum(
            "b i d, b j d -> b i j", query, key
        ).softmax(dim=-1)
        result = torch.einsum("b i j, b j d -> b i d", attention, value)
        result = result.reshape(
            batch,
            height_blocks,
            width_blocks,
            self.num_heads,
            block,
            block,
            self.head_channels,
        )
        result = result.permute(0, 3, 6, 1, 4, 2, 5).reshape(
            batch, channels, height, width
        )
        if self.reduction > 1:
            result = self.local_propagation(result)
        return result[..., :original_height, :original_width]

    @staticmethod
    def _unfold_windows(
        tensor: Tensor,
        neighborhood: int,
        block: int,
        halo: int,
        channels: int,
    ) -> Tensor:
        tensor = F.unfold(
            tensor, kernel_size=neighborhood, stride=block, padding=halo
        )
        batch, _, windows = tensor.shape
        return (
            tensor.reshape(
                batch, channels, neighborhood * neighborhood, windows
            )
            .permute(0, 3, 2, 1)
            .reshape(batch * windows, neighborhood * neighborhood, channels)
        )


class MixedScaleGating(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        expansion: float = 1.0,
    ) -> None:
        super().__init__()
        expanded = int(channels * expansion)
        self.direct = nn.Conv2d(channels, expanded, 1)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                padding_mode="reflect",
            ),
            nn.Conv2d(channels, expanded, 1),
        )
        self.output = nn.Conv2d(expanded, channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(F.gelu(self.direct(features)) * self.local(features))


class Interpolate(nn.Module):
    def __init__(
        self, *, scale_factor: int, mode: str, align_corners: bool
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: Tensor) -> Tensor:
        return F.interpolate(
            features,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=True,
        )


class WithBiasLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = 1e-6

    def forward(self, features: Tensor) -> Tensor:
        mean = features.mean(dim=-1, keepdim=True)
        variance = features.var(dim=-1, keepdim=True, unbiased=False)
        return (
            (features - mean)
            / torch.sqrt(variance + self.eps)
            * self.weight
            + self.bias
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = WithBiasLayerNorm(channels)

    def forward(self, features: Tensor) -> Tensor:
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
