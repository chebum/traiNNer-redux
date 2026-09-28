import pytest
import torch
from traiNNer.archs.ssiu_arch import SSIU, LearnedFeatureNoise, StochasticTextureHead


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
    with pytest.raises(RuntimeError, match="outside optional paths"):
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
        noise_scale=1,
        noise_mode="disabled",
        highpass_kernel=5,
        init_std=1e-5,
    )
    output = head(torch.randn(2, 8, 7, 9))
    assert output.shape == (2, 1, 21, 27)
    assert torch.count_nonzero(output) == 0


def test_stochastic_texture_head_supports_correlated_noise() -> None:
    head = StochasticTextureHead(
        channels=8,
        out_channels=3,
        scale=2,
        noise_channels=2,
        noise_scale=4,
        noise_mode="always",
        highpass_kernel=15,
        init_std=1e-5,
    )
    features = torch.randn(2, 8, 17, 19)
    torch.manual_seed(1)
    first = head(features)
    torch.manual_seed(2)
    second = head(features)
    assert first.shape == (2, 3, 34, 38)
    assert not torch.equal(first, second)


def test_stochastic_texture_head_rejects_invalid_noise_scale() -> None:
    with pytest.raises(ValueError, match="noise_scale"):
        StochasticTextureHead(
            channels=8,
            out_channels=3,
            scale=2,
            noise_channels=2,
            noise_scale=0,
            noise_mode="always",
            highpass_kernel=5,
            init_std=1e-5,
        )


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


def test_zero_gain_deep_noise_preserves_deterministic_checkpoint() -> None:
    deterministic = SSIU(scale=2, n_feats=8, n_blocks=3)
    stochastic = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic_feature_blocks=2,
        noise_mode="always",
        feature_noise_init_gain=0,
    )
    stochastic.load_state_dict(deterministic.state_dict(), strict=False)
    value = torch.randn(1, 3, 16, 16)
    assert torch.equal(deterministic(value), stochastic(value))


def test_zero_initialized_plus_residuals_preserve_checkpoint_and_rng() -> None:
    torch.manual_seed(7)
    deterministic = SSIU(scale=2, n_feats=8, n_blocks=3)
    state_after_deterministic = torch.random.get_rng_state()

    torch.random.set_rng_state(state_after_deterministic)
    plus = SSIU(scale=2, n_feats=8, n_blocks=3, plus_residuals=True)
    state_after_plus = torch.random.get_rng_state()

    # Compare with constructing the same deterministic base model: the added
    # zero branches must not perturb later discriminator/data RNG state.
    torch.random.set_rng_state(state_after_deterministic)
    SSIU(scale=2, n_feats=8, n_blocks=3)
    assert torch.equal(state_after_plus, torch.random.get_rng_state())

    plus.load_state_dict(deterministic.state_dict(), strict=False)
    value = torch.randn(1, 3, 16, 16)
    assert torch.equal(deterministic(value), plus(value))


def test_plus_residuals_receive_gradients() -> None:
    model = SSIU(scale=2, n_feats=8, n_blocks=3, plus_residuals=True)
    model(torch.randn(1, 3, 16, 16)).square().mean().backward()

    for block in model.body:
        assert block.plus_input_projection is not None
        assert block.plus_input_projection.weight.grad is not None
        assert torch.count_nonzero(block.plus_input_projection.weight.grad) > 0
        assert block.plus_long_skip_gain is not None
        assert block.plus_long_skip_gain.grad is not None
        assert torch.count_nonzero(block.plus_long_skip_gain.grad) > 0


def test_feature_noise_is_training_only_and_uses_one_spatial_field() -> None:
    noise = LearnedFeatureNoise(channels=3, noise_mode="train", init_gain=0.5)
    value = torch.zeros(1, 3, 8, 8)

    noise.eval()
    assert torch.equal(noise(value), value)

    noise.train()
    output = noise(value)
    assert not torch.equal(output, value)
    assert torch.equal(output[:, 0], output[:, 1])
    assert torch.equal(output[:, 1], output[:, 2])


def test_internal_residual_noise_is_training_only() -> None:
    deterministic = SSIU(scale=2, n_feats=8, n_blocks=3)
    stochastic = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic_internal_residuals=True,
        noise_mode="train",
        feature_noise_init_gain=1e-3,
    )
    stochastic.load_state_dict(deterministic.state_dict(), strict=False)
    value = torch.randn(1, 3, 16, 16)

    stochastic.eval()
    first_eval = stochastic(value)
    second_eval = stochastic(value)
    assert torch.equal(first_eval, second_eval)
    assert torch.equal(first_eval, deterministic.eval()(value))

    stochastic.train()
    first_train = stochastic(value)
    second_train = stochastic(value)
    assert not torch.equal(first_train, second_train)
    assert sum(len(block.residual_noise) for block in stochastic.body) == 9


def test_original_training_checkpoint_names_are_mapped() -> None:
    model = SSIU(scale=2, n_feats=8, n_blocks=3)
    original_names = {}
    reverse_renames = (
        (".norm.norm.", ".norm.body."),
        (".sparse_constraint.", ".s1."),
        (".similarity_constraint.", ".s2."),
        (".attention.", ".s3."),
        (".aggregate.", ".s4."),
        (".feed_forward.", ".ffn."),
        (".direct.", ".project_in1."),
        (".local.", ".project_in2."),
        (".output.", ".project_out."),
        (".local_propagation.", ".LocalProp."),
        (".relative_height", ".rel_h"),
        (".relative_width", ".rel_w"),
        (".qkv.", ".qkv_conv."),
        ("moe.weight_a.", "moe.fc_a."),
        ("moe.weight_b.", "moe.fc_b."),
        ("moe.weight_c.", "moe.fc_c."),
    )
    for runtime_key, value in model.state_dict().items():
        training_key = runtime_key
        for runtime_name, training_name in reverse_renames:
            training_key = training_key.replace(runtime_name, training_name)
        original_names[f"module.{training_key}"] = value

    mapped = model.map_state_dict(original_names)
    assert mapped.keys() == model.state_dict().keys()
    model.load_state_dict(original_names, strict=True)


def test_deep_noise_is_seed_dependent_and_gain_receives_gradient() -> None:
    model = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=3,
        stochastic_feature_blocks=2,
        noise_mode="always",
        feature_noise_init_gain=1e-3,
    )
    value = torch.randn(1, 3, 16, 16)
    torch.manual_seed(1)
    first = model(value)
    torch.manual_seed(2)
    second = model(value)
    assert not torch.equal(first, second)
    first.mean().backward()
    gains = [
        module.gain.grad
        for module in model.feature_noise.values()
        if isinstance(module, LearnedFeatureNoise)
    ]
    assert all(gain is not None for gain in gains)
    assert any(torch.count_nonzero(gain) > 0 for gain in gains if gain is not None)


def test_zero_deep_noise_gain_receives_learning_signal() -> None:
    noise = LearnedFeatureNoise(channels=4, noise_mode="always", init_gain=0)
    output = noise(torch.randn(1, 4, 8, 8))
    output.square().mean().backward()
    assert noise.gain.grad is not None
    assert torch.count_nonzero(noise.gain.grad) > 0


def test_train_late_blocks_freezes_early_structure() -> None:
    model = SSIU(
        scale=2,
        n_feats=8,
        n_blocks=5,
        stochastic_feature_blocks=2,
        train_late_blocks=2,
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "feature_noise.3.gain" in trainable
    assert any(name.startswith("body.3.") for name in trainable)
    assert any(name.startswith("body.4.") for name in trainable)
    assert any(name.startswith("moe.") for name in trainable)
    assert any(name.startswith("tail.") for name in trainable)
    assert not any(name.startswith("head.") for name in trainable)
    assert not any(name.startswith("body.2.") for name in trainable)


def test_late_block_training_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SSIU(
            scale=2,
            n_feats=8,
            n_blocks=3,
            stochastic=True,
            train_texture_only=True,
            train_late_blocks=1,
        )


def test_nonstandard_block_count_still_selects_three_experts() -> None:
    model = SSIU(scale=2, n_feats=8, n_blocks=5)
    assert len(model.expert_indices) == 3
    assert model(torch.randn(1, 3, 16, 16)).shape == (1, 3, 32, 32)
