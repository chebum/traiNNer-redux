import pytest
import torch
from traiNNer.metrics.stochastic import (
    stochastic_output_diagnostics,
    summarize_seed_metrics,
)


def test_seed_metric_summary_respects_metric_direction() -> None:
    values = [30.0, 31.0, 29.0, 30.0]
    summary = summarize_seed_metrics(values, "higher")
    assert summary["mean"] == 30.0
    assert summary["std"] == pytest.approx(2**-0.5)
    assert summary["worst"] == 29.0
    assert summarize_seed_metrics(values, "lower")["worst"] == 31.0


def test_high_frequency_seed_variation_is_distinguished_from_drift() -> None:
    base = torch.full((1, 3, 32, 32), 0.5)
    checker = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2
    checker = (checker.float() * 2 - 1)[None, None].expand_as(base) * 0.05
    high_frequency = stochastic_output_diagnostics([base, base + checker], base)
    low_frequency = stochastic_output_diagnostics([base, base + 0.05], base)

    assert high_frequency["high_to_low_diversity"] > 5
    assert low_frequency["high_to_low_diversity"] < 0.1


def test_detail_localization_distinguishes_textured_and_smooth_regions() -> None:
    base = torch.full((1, 3, 32, 32), 0.5)
    target = base.clone()
    target[:, :, :, :16] += (
        ((torch.arange(32)[:, None] + torch.arange(16)[None, :]) % 2)
        .float()
        .mul(0.1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    variation = torch.zeros_like(base)
    variation[:, :, :, :16] = target[:, :, :, :16] - base[:, :, :, :16]
    diagnostics = stochastic_output_diagnostics(
        [base, base + variation], base, target
    )
    assert diagnostics["detail_to_smooth_diversity"] > 1
