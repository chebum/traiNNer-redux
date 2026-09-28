"""Initialize ESRGAN+ from a RealESRGAN RRDBNet checkpoint.

The additional ESRGAN+ intra-RDB residual paths use trainable per-feature gates
initialized to zero, so deterministic inference initially reproduces the source
RRDBNet. Clean training can then open those paths gradually. Learned noise gains
retain their small constructor initialization.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from traiNNer.archs.esrganplus_arch import ESRGANPlus

BODY_PATTERN = re.compile(
    r"^body\.(?P<block>\d+)\.rdb(?P<rdb>[123])\.conv(?P<conv>[1-5])\."
    r"(?P<parameter>weight|bias)$"
)
DIRECT_KEYS = {
    "conv_first.weight": "model.0.weight",
    "conv_first.bias": "model.0.bias",
    "conv_body.weight": "model.1.sub.23.weight",
    "conv_body.bias": "model.1.sub.23.bias",
    "conv_up1.weight": "model.3.weight",
    "conv_up1.bias": "model.3.bias",
    "conv_up2.weight": "model.6.weight",
    "conv_up2.bias": "model.6.bias",
    "conv_hr.weight": "model.8.weight",
    "conv_hr.bias": "model.8.bias",
    "conv_last.weight": "model.10.weight",
    "conv_last.bias": "model.10.bias",
}


def target_key(source_key: str) -> str:
    if source_key in DIRECT_KEYS:
        return DIRECT_KEYS[source_key]
    match = BODY_PATTERN.fullmatch(source_key)
    if match is None:
        raise KeyError(f"Unsupported RRDBNet parameter: {source_key}")
    fields = match.groupdict()
    return (
        f"model.1.sub.{fields['block']}.RDB{fields['rdb']}."
        f"conv{fields['conv']}.0.{fields['parameter']}"
    )


def extract_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary")
    for key in ("params_ema", "params"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint has no params_ema or params state dictionary")


def convert_state(
    source: dict[str, torch.Tensor], *, noise_init_gain: float
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = ESRGANPlus(
        scale=4,
        noise_mode="train",
        noise_style="learned_additive",
        noise_init_gain=noise_init_gain,
        noise_after_rrdb=False,
        rrdrb_residual_path_init_gain=0.0,
    )
    converted = model.state_dict()

    mapped_targets: set[str] = set()
    for source_name, tensor in source.items():
        destination_name = target_key(source_name)
        if destination_name not in converted:
            raise KeyError(f"Mapped target does not exist: {destination_name}")
        if converted[destination_name].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {source_name} -> {destination_name}: "
                f"{tuple(tensor.shape)} != {tuple(converted[destination_name].shape)}"
            )
        converted[destination_name] = tensor.detach().cpu().contiguous()
        mapped_targets.add(destination_name)

    if len(mapped_targets) != len(source):
        raise RuntimeError("Not every source tensor was mapped exactly once")
    return {key: value.cpu().contiguous() for key, value in converted.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-init-gain", type=float, default=0.001)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify deterministic output equivalence with the source model.",
    )
    return parser.parse_args()


def verify_equivalence(
    source_path: Path, converted: dict[str, torch.Tensor]
) -> None:
    from spandrel import ModelLoader

    source = ModelLoader().load_from_file(source_path).eval()
    target = ESRGANPlus(
        scale=4,
        noise_mode="disabled",
        noise_style="learned_additive",
        noise_after_rrdb=False,
        rrdrb_residual_path_init_gain=0.0,
    ).eval()
    target.load_state_dict(converted, strict=True)
    torch.manual_seed(7)
    sample = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        source_output = source(sample)
        target_output = target(sample)
    maximum_error = float((source_output - target_output).abs().max())
    if maximum_error > 1e-6:
        raise RuntimeError(f"Conversion verification failed: max error={maximum_error}")
    print(f"verified deterministic equivalence; max error={maximum_error}")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    source = extract_state(checkpoint)
    converted = convert_state(source, noise_init_gain=args.noise_init_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, args.output)
    projection_count = sum(key.endswith("residual_path_gain") for key in converted)
    gain_count = sum(key.endswith("noise.gain") for key in converted)
    print(
        f"converted {len(source)} source tensors to {args.output}; "
        f"gated {projection_count} new residual paths at zero; "
        f"initialized {gain_count} gains"
    )
    if args.verify:
        verify_equivalence(args.input, converted)


if __name__ == "__main__":
    main()
