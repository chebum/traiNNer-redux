from types import SimpleNamespace

import torch
from torch import nn
from traiNNer.models.sr_model import SRModel, _stochastic_frequency_deltas


class RandomGenerator(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.randn_like(value) * 0.1


def test_validation_generates_four_seeds_and_selects_worst() -> None:
    model = object.__new__(SRModel)
    model.device = torch.device("cpu")
    model.amp_dtype = torch.float32
    model.use_amp = False
    model.optimizers_schedule_free = []
    model.net_g = RandomGenerator()
    model.net_g_ema = None
    model.lq = torch.ones(1, 3, 8, 8)
    model.gt = torch.ones_like(model.lq)
    model.is_train = True
    model.opt = SimpleNamespace(
        input_pixel_format="rgb",
        output_pixel_format="rgb",
        val=SimpleNamespace(
            stochastic_samples=4,
            stochastic_seed=10,
            stochastic_selection="worst",
            tile_size=0,
        ),
    )

    torch.manual_seed(99)
    expected_next_random = torch.rand(1)
    torch.manual_seed(99)
    SRModel.test(model)

    assert len(model.validation_outputs) == 4
    assert all(
        not torch.equal(model.validation_outputs[0], output)
        for output in model.validation_outputs[1:]
    )
    errors = [
        torch.mean(torch.abs(output - model.gt)).item()
        for output in model.validation_outputs
    ]
    assert torch.equal(model.output, model.validation_outputs[errors.index(max(errors))])
    # Validation must not disturb the training RNG stream.
    assert torch.equal(torch.rand(1), expected_next_random)


def test_detail_weighted_diversity_ignores_flat_targets_and_has_gradients() -> None:
    first = torch.zeros(1, 3, 8, 8, requires_grad=True)
    second = torch.randn(1, 3, 8, 8, requires_grad=True) * 0.01
    flat_target = torch.zeros_like(first)
    _, global_delta, flat_detail_delta = _stochastic_frequency_deltas(
        first, second, flat_target, 5, 0.01
    )
    assert global_delta > 0
    assert flat_detail_delta == 0

    textured_target = torch.zeros_like(first)
    textured_target[:, :, ::2, ::2] = 1
    _, _, detail_delta = _stochastic_frequency_deltas(
        first, second, textured_target, 5, 0.01
    )
    assert detail_delta > 0
    detail_delta.backward()
    assert first.grad is not None
    assert torch.count_nonzero(first.grad) > 0
