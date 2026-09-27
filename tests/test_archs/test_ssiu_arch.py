import pytest
import torch
from traiNNer.archs.ssiu_arch import SSIU, StochasticTextureHead


def test_ssiu_x2_shape_and_deterministic_eval() -> None:
    model = SSIU(scale=2, n_feats=8, n_blocks=3)
    value = torch.randn(1, 3, 17, 19)
    first = model(value)
    second = model(value)
    assert first.shape == (1, 3, 34, 38)
    assert torch.equal(first, second)


def test_stochastic_texture_head_is_near_zero_but_seed_dependent() -> None:
    model = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic=True,
        noise_channels=2,
        noise_mode="always",
        texture_init_std=1e-5,
    )
    value = torch.randn(1, 3, 16, 16)
    torch.manual_seed(1)
    first = model(value)
    torch.manual_seed(2)
    second = model(value)
    assert not torch.equal(first, second)
    assert torch.mean(torch.abs(first - second)) < 1e-3


def test_deterministic_checkpoint_only_misses_texture_head() -> None:
    deterministic = SSIU(scale=2, n_feats=8, n_blocks=3)
    stochastic = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic=True,
        noise_channels=2,
    )
    incompatible = stochastic.load_state_dict(deterministic.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("texture_head.") for key in incompatible.missing_keys)

    broken_state = dict(deterministic.state_dict())
    broken_state.pop("head.weight")
    with pytest.raises(RuntimeError, match="outside texture_head"):
        stochastic.load_state_dict(broken_state, strict=False)


def test_stochastic_diversity_objective_reaches_texture_head() -> None:
    model = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic=True,
        noise_channels=2,
        noise_mode="always",
        texture_init_std=1e-5,
    )
    value = torch.randn(1, 3, 16, 16)
    first = model(value)
    second = model(value)
    first_low = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(first, (2, 2, 2, 2), mode="reflect"), 5, stride=1
    )
    second_low = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(second, (2, 2, 2, 2), mode="reflect"), 5, stride=1
    )
    high_delta = torch.nn.functional.l1_loss(
        first - first_low, second - second_low
    )
    torch.nn.functional.relu(high_delta.new_tensor(0.0019607843) - high_delta).backward()
    assert model.texture_head is not None
    gradients = [
        parameter.grad
        for parameter in model.texture_head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_texture_head_disabled_mode_is_shape_safe_for_x3() -> None:
    head = StochasticTextureHead(
        channels=8,
        out_channels=1,
        scale=3,
        noise_channels=2,
        noise_mode="disabled",
        highpass_kernel=5,
        init_std=1e-5,
    )
    output = head(torch.randn(2, 8, 7, 9))
    assert output.shape == (2, 1, 21, 27)
    assert torch.count_nonzero(output) == 0


def test_train_texture_only_freezes_deterministic_parameters() -> None:
    model = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic=True,
        noise_channels=2,
        train_texture_only=True,
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("texture_head.") for name in trainable)
    assert not model.head.weight.requires_grad


def test_train_texture_only_requires_stochastic_head() -> None:
    with pytest.raises(ValueError, match="requires stochastic"):
        SSIU(scale=2, n_feats=8, n_blocks=3, train_texture_only=True)


def test_nonstandard_block_count_still_selects_three_experts() -> None:
    model = SSIU(scale=2, n_feats=8, n_blocks=5)
    assert len(model.expert_indices) == 3
    assert model(torch.randn(1, 3, 16, 16)).shape == (1, 3, 32, 32)
