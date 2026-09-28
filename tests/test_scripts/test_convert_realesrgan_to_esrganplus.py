from __future__ import annotations

import torch
from scripts.convert_realesrgan_to_esrganplus import convert_state, target_key
from traiNNer.archs.esrganplus_arch import ESRGANPlus


def test_rrdb_keys_map_to_esrganplus_layout() -> None:
    assert (
        target_key("body.7.rdb2.conv4.weight")
        == "model.1.sub.7.RDB2.conv4.0.weight"
    )
    assert target_key("conv_up2.bias") == "model.6.bias"


def test_conversion_preserves_shared_weights_and_neutralizes_new_paths() -> None:
    model = ESRGANPlus(
        scale=4,
        noise_mode="train",
        noise_style="learned_additive",
        noise_init_gain=0.001,
        noise_after_rrdb=False,
        rrdrb_residual_path_init_gain=0.0,
    )
    state = model.state_dict()
    source = {
        "conv_first.weight": torch.randn_like(state["model.0.weight"]),
        "conv_first.bias": torch.randn_like(state["model.0.bias"]),
        "body.0.rdb1.conv1.weight": torch.randn_like(
            state["model.1.sub.0.RDB1.conv1.0.weight"]
        ),
    }

    converted = convert_state(source, noise_init_gain=0.001)

    assert torch.equal(converted["model.0.weight"], source["conv_first.weight"])
    assert torch.equal(
        converted["model.1.sub.0.RDB1.conv1.0.weight"],
        source["body.0.rdb1.conv1.weight"],
    )
    assert all(
        torch.count_nonzero(value) == 0
        for key, value in converted.items()
        if key.endswith("residual_path_gain")
    )
    assert all(
        torch.allclose(value, torch.full_like(value, 0.001))
        for key, value in converted.items()
        if key.endswith("noise.gain")
    )
