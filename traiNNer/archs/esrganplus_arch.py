"""ESRGAN+ / nESRGAN+ generator.

The module layout intentionally matches the public ESRGAN+ implementation so
that checkpoints such as 4x_Valar_v1 can be loaded without rewriting keys.
"""

import math
from typing import Literal

import torch
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY

NoiseMode = Literal["train", "always", "disabled"]
NoiseStyle = Literal["multiplicative", "learned_additive"]


class GaussianNoise(nn.Module):
    """Multiplicative Gaussian noise used by nESRGAN+.

    ``train`` reproduces the reference implementation. ``always`` is useful
    for stochastic inference, while ``disabled`` permits deterministic tests
    and fidelity pretraining without changing the checkpoint structure.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        noise_mode: NoiseMode = "train",
        relative_detach: bool = False,
        *,
        channels: int | None = None,
        noise_style: NoiseStyle = "multiplicative",
        init_gain: float = 0.001,
    ) -> None:
        super().__init__()
        if noise_mode not in ("train", "always", "disabled"):
            raise ValueError(f"Unsupported noise_mode: {noise_mode}")
        if noise_style not in ("multiplicative", "learned_additive"):
            raise ValueError(f"Unsupported noise_style: {noise_style}")
        if noise_style == "learned_additive" and channels is None:
            raise ValueError("channels are required for learned additive noise")
        if init_gain < 0:
            raise ValueError("noise_init_gain must be non-negative")
        self.sigma = sigma
        self.noise_mode = noise_mode
        self.relative_detach = relative_detach
        self.noise_style = noise_style
        self.gain = (
            nn.Parameter(torch.full((1, channels, 1, 1), float(init_gain)))
            if noise_style == "learned_additive" and channels is not None
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        enabled = self.noise_mode == "always" or (
            self.noise_mode == "train" and self.training
        )
        if not enabled:
            return x
        if self.gain is not None:
            return x + torch.randn_like(x) * self.gain
        if self.sigma == 0:
            return x
        scale = self.sigma * (x.detach() if self.relative_detach else x)
        return x + torch.randn_like(x) * scale


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
    ]
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ResidualDenseBlock5C(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        num_feat: int,
        num_grow_ch: int,
        noise_sigma: float,
        noise_mode: NoiseMode,
        noise_style: NoiseStyle,
        noise_init_gain: float,
        residual_path_init_gain: float | None,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(
            noise_sigma,
            noise_mode,
            channels=num_feat,
            noise_style=noise_style,
            init_gain=noise_init_gain,
        )
        # Bias=False is required for compatibility with the reference model.
        self.conv1x1 = nn.Conv2d(num_feat, num_grow_ch, 1, bias=False)
        self.residual_path_gain = (
            nn.Parameter(
                torch.full((1, num_grow_ch, 1, 1), residual_path_init_gain)
            )
            if residual_path_init_gain is not None
            else None
        )
        self.conv1 = _conv(num_feat, num_grow_ch)
        self.conv2 = _conv(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = _conv(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = _conv(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = _conv(num_feat + 4 * num_grow_ch, num_feat, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        extra = self.conv1x1(x)
        if self.residual_path_gain is not None:
            extra = extra * self.residual_path_gain
        x2 = self.conv2(torch.cat((x, x1), 1)) + extra
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        extra = x2
        if self.residual_path_gain is not None:
            extra = extra * self.residual_path_gain
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1)) + extra
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.noise(x + x5 * 0.2)


class RRDBPlus(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        num_feat: int,
        num_grow_ch: int,
        noise_sigma: float,
        noise_mode: NoiseMode,
        noise_style: NoiseStyle,
        noise_init_gain: float,
        noise_after_rrdb: bool,
        residual_path_init_gain: float | None,
    ) -> None:
        super().__init__()
        args = (
            num_feat,
            num_grow_ch,
            noise_sigma,
            noise_mode,
            noise_style,
            noise_init_gain,
            residual_path_init_gain,
        )
        # Attribute names preserve ESRGAN+ checkpoint compatibility.
        self.RDB1 = ResidualDenseBlock5C(*args)
        self.RDB2 = ResidualDenseBlock5C(*args)
        self.RDB3 = ResidualDenseBlock5C(*args)
        self.noise = (
            GaussianNoise(
                noise_sigma,
                noise_mode,
                channels=num_feat,
                noise_style=noise_style,
                init_gain=noise_init_gain,
            )
            if noise_after_rrdb
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.RDB3(self.RDB2(self.RDB1(x)))
        return self.noise(x + out * 0.2)


class _ShortcutBlock(nn.Module):
    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()
        self.sub = submodule

    def forward(self, x: Tensor) -> Tensor:
        return x + self.sub(x)


class ESRGANPlus(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        scale: int = 4,
        in_nc: int = 3,
        out_nc: int = 3,
        num_filters: int = 64,
        num_blocks: int = 23,
        growth_channels: int = 32,
        noise_sigma: float = 0.1,
        noise_mode: NoiseMode = "train",
        noise_style: NoiseStyle = "multiplicative",
        noise_init_gain: float = 0.001,
        noise_after_rrdb: bool = False,
        rrdrb_residual_path_init_gain: float | None = None,
    ) -> None:
        super().__init__()
        if scale not in (1, 2, 3, 4, 8):
            raise ValueError("ESRGANPlus supports scale 1, 2, 3, 4, or 8")

        body: list[nn.Module] = [
            RRDBPlus(
                num_filters,
                growth_channels,
                noise_sigma,
                noise_mode,
                noise_style,
                noise_init_gain,
                noise_after_rrdb,
                rrdrb_residual_path_init_gain,
            )
            for _ in range(num_blocks)
        ]
        body.append(nn.Conv2d(num_filters, num_filters, 3, 1, 1))

        model: list[nn.Module] = [
            nn.Conv2d(in_nc, num_filters, 3, 1, 1),
            _ShortcutBlock(nn.Sequential(*body)),
        ]
        factors = [3] if scale == 3 else [2] * int(math.log2(scale))
        for factor in factors:
            model.extend(
                [
                    nn.Upsample(scale_factor=factor, mode="nearest"),
                    *_conv(num_filters, num_filters),
                ]
            )
        model.extend(
            [
                nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_filters, out_nc, 3, 1, 1),
            ]
        )
        self.model = nn.Sequential(*model)
        self.hyperparameters = {
            "scale": scale,
            "in_nc": in_nc,
            "out_nc": out_nc,
            "num_filters": num_filters,
            "num_blocks": num_blocks,
            "growth_channels": growth_channels,
            "noise_sigma": noise_sigma,
            "noise_mode": noise_mode,
            "noise_style": noise_style,
            "noise_init_gain": noise_init_gain,
            "noise_after_rrdb": noise_after_rrdb,
            "rrdrb_residual_path_init_gain": rrdrb_residual_path_init_gain,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def set_noise_mode(self, noise_mode: NoiseMode) -> None:
        for module in self.modules():
            if isinstance(module, GaussianNoise):
                module.noise_mode = noise_mode

    def feature_noise_gain_statistics(self) -> tuple[Tensor, Tensor] | None:
        """Return mean and maximum absolute learned noise gains."""
        gains = [
            module.gain.reshape(-1)
            for module in self.modules()
            if isinstance(module, GaussianNoise) and module.gain is not None
        ]
        if not gains:
            return None
        absolute_gains = torch.cat(gains).abs()
        return absolute_gains.mean(), absolute_gains.max()


@ARCH_REGISTRY.register()
def esrganplus(**kwargs: object) -> ESRGANPlus:
    return ESRGANPlus(**kwargs)  # type: ignore[arg-type]
