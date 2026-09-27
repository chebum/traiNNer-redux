"""Losses for useful seed-dependent stochastic texture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY


def split_frequency(value: Tensor, kernel_size: int) -> tuple[Tensor, Tensor]:
    padding = kernel_size // 2
    low = F.avg_pool2d(
        F.pad(value.float(), (padding,) * 4, mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )
    return low, value.float() - low


@LOSS_REGISTRY.register()
class StochasticTextureLoss(nn.Module):
    """Reward bounded high-frequency diversity while penalizing structural drift."""

    def __init__(
        self,
        loss_weight: float,
        diversity_margin: float = 0.015,
        low_frequency_weight: float = 1.0,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        self.loss_weight = loss_weight
        self.diversity_margin = diversity_margin
        self.low_frequency_weight = low_frequency_weight
        self.kernel_size = kernel_size

    def forward(self, first: Tensor, second: Tensor) -> dict[str, Tensor]:
        first_low, first_high = split_frequency(first, self.kernel_size)
        second_low, second_high = split_frequency(second, self.kernel_size)
        low_diversity = F.l1_loss(first_low, second_low)
        high_diversity = F.l1_loss(first_high, second_high)
        return {
            "high_margin": torch.relu(
                high_diversity.new_tensor(self.diversity_margin) - high_diversity
            ),
            "low_stability": low_diversity * self.low_frequency_weight,
        }
