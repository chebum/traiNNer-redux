import pytest
import torch
from traiNNer.archs.esrganplus_arch import ESRGANPlus, GaussianNoise


def test_esrganplus_x2_shape_and_noise_layer_count() -> None:
    model = ESRGANPlus(scale=2, num_filters=8, num_blocks=2, growth_channels=4)
    output = model(torch.randn(1, 3, 8, 9))
    assert output.shape == (1, 3, 16, 18)
    # Reference nESRGAN+ injects after each of the three RDBs, not after RRDBs.
    assert sum(isinstance(module, GaussianNoise) for module in model.modules()) == 6


def test_legacy_rrdb_noise_can_reproduce_earlier_experiments() -> None:
    model = ESRGANPlus(
        scale=2,
        num_filters=8,
        num_blocks=2,
        growth_channels=4,
        noise_after_rrdb=True,
    )
    assert sum(isinstance(module, GaussianNoise) for module in model.modules()) == 8


def test_noise_modes_are_explicit_and_reproducible() -> None:
    x = torch.ones(1, 3, 4, 4)
    noise = GaussianNoise(sigma=0.1, noise_mode="train")
    noise.eval()
    assert torch.equal(noise(x), x)

    noise.noise_mode = "always"
    torch.manual_seed(12)
    first = noise(x)
    torch.manual_seed(12)
    second = noise(x)
    assert torch.equal(first, second)
    assert not torch.equal(first, x)

    noise.noise_mode = "disabled"
    assert torch.equal(noise(x), x)


def test_gradients_cross_multiplicative_noise() -> None:
    x = torch.ones(1, 2, 3, 3, requires_grad=True)
    GaussianNoise(noise_mode="always")(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_learned_additive_noise_has_channel_gains_and_gradients() -> None:
    noise = GaussianNoise(
        noise_mode="always",
        channels=4,
        noise_style="learned_additive",
        init_gain=0.001,
    )
    x = torch.randn(2, 4, 5, 6, requires_grad=True)
    torch.manual_seed(1)
    first = noise(x)
    torch.manual_seed(2)
    second = noise(x)
    assert noise.gain is not None
    assert noise.gain.shape == (1, 4, 1, 1)
    assert not torch.equal(first, second)
    first.square().mean().backward()
    assert noise.gain.grad is not None
    assert torch.count_nonzero(noise.gain.grad) > 0


def test_fidelity_checkpoint_only_misses_learned_noise_gains() -> None:
    fidelity = ESRGANPlus(scale=2, num_filters=8, num_blocks=2, growth_channels=4)
    stochastic = ESRGANPlus(
        scale=2,
        num_filters=8,
        num_blocks=2,
        growth_channels=4,
        noise_style="learned_additive",
    )
    incompatible = stochastic.load_state_dict(fidelity.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert len(incompatible.missing_keys) == 6
    assert all(key.endswith("noise.gain") for key in incompatible.missing_keys)


def test_invalid_noise_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported noise_mode"):
        GaussianNoise(noise_mode="sometimes")  # type: ignore[arg-type]


def test_valar_checkpoint_keys_and_strict_load() -> None:
    checkpoint = pytest.importorskip("os").environ.get("VALAR_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set VALAR_CHECKPOINT to run the external compatibility test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ESRGANPlus(scale=4, noise_mode="always")
    model.load_state_dict(state, strict=True)
