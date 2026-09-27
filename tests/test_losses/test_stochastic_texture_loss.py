import pytest
import torch
from traiNNer.losses.stochastic_texture_loss import StochasticTextureLoss


def test_stochastic_texture_loss_rewards_high_frequency_not_drift() -> None:
    loss = StochasticTextureLoss(loss_weight=1, diversity_margin=0.02)
    base = torch.full((1, 3, 32, 32), 0.5)
    checker = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2
    checker = (checker.float() * 2 - 1)[None, None].expand_as(base) * 0.02

    texture = loss(base, base + checker)
    drift = loss(base, base + 0.02)

    assert texture["high_margin"] < drift["high_margin"]
    assert texture["low_stability"] < drift["low_stability"]


def test_stochastic_texture_loss_validates_kernel() -> None:
    with pytest.raises(ValueError):
        StochasticTextureLoss(loss_weight=1, kernel_size=4)
