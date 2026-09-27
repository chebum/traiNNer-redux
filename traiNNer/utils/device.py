from typing import Literal

import torch

DeviceName = Literal["auto", "cuda", "mps", "cpu"]


def mps_is_available() -> bool:
    """Return whether the current PyTorch build can use Apple's MPS backend."""
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(device: DeviceName, num_gpu: int) -> torch.device:
    """Resolve and validate the accelerator selected by the training config."""
    if device == "auto":
        if num_gpu > 0:
            device = "cuda"
        else:
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device is set to 'cuda', but CUDA is not available in this PyTorch "
            "installation. Select device: mps on Apple Silicon or device: cpu."
        )
    if device == "mps" and not mps_is_available():
        raise RuntimeError(
            "device is set to 'mps', but the MPS backend is not available. MPS "
            "requires Apple Silicon (or a supported AMD GPU) and an MPS-enabled "
            "PyTorch installation."
        )
    if device == "cuda" and num_gpu == 0:
        raise ValueError("device: cuda requires num_gpu to be at least 1")
    if device == "mps" and num_gpu != 0:
        raise ValueError("device: mps requires num_gpu: 0 or num_gpu: auto")

    return torch.device(device)


def empty_device_cache(device: torch.device) -> None:
    """Release unused allocator cache for the selected accelerator."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_device_oom(error: RuntimeError, device: torch.device) -> bool:
    """Recognize allocator out-of-memory errors for CUDA and MPS."""
    message = str(error).lower()
    if device.type == "cuda":
        return "cuda out of memory" in message
    if device.type == "mps":
        return "mps backend out of memory" in message or (
            "out of memory" in message and "mps" in message
        )
    return False
