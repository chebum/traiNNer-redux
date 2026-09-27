"""Diagnostics for seeded stochastic super-resolution outputs."""

from __future__ import annotations

from itertools import combinations
from statistics import fmean, pstdev
from typing import Literal

from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def summarize_seed_metrics(
    values: list[float], better: Literal["higher", "lower"]
) -> dict[str, float]:
    """Summarize samples without choosing an oracle output as the benchmark."""
    if not values:
        raise ValueError("At least one seeded metric value is required")
    return {
        "mean": fmean(values),
        "std": pstdev(values),
        "worst": min(values) if better == "higher" else max(values),
    }


def stochastic_output_diagnostics(
    outputs: list[Tensor], lq: Tensor
) -> dict[str, float]:
    """Measure LR consistency and whether seed variation is high-frequency."""
    if len(outputs) < 2:
        raise ValueError("At least two stochastic outputs are required")
    target_size = lq.shape[-2:]
    consistency = fmean(
        F.l1_loss(
            F.interpolate(
                output.float(), size=target_size, mode="bicubic", antialias=True
            ),
            lq.float(),
        ).item()
        for output in outputs
    )
    # These are model-agnostic proxies rather than a reconstruction guarantee:
    # bicubic may differ from the training degradation kernel. Reflect padding
    # keeps artificial zero-valued borders out of the frequency comparison.
    low_frequency = [
        F.avg_pool2d(
            F.pad(output.float(), (2, 2, 2, 2), mode="reflect"),
            kernel_size=5,
            stride=1,
        )
        for output in outputs
    ]
    high_frequency = [
        output.float() - low for output, low in zip(outputs, low_frequency, strict=True)
    ]
    low_diversity = fmean(
        F.l1_loss(left, right).item() for left, right in combinations(low_frequency, 2)
    )
    high_diversity = fmean(
        F.l1_loss(left, right).item() for left, right in combinations(high_frequency, 2)
    )
    return {
        "lr_consistency_proxy_l1": consistency,
        "lowfreq_diversity_l1": low_diversity,
        "highfreq_diversity_l1": high_diversity,
        "high_to_low_diversity": high_diversity / max(low_diversity, 1e-12),
    }
