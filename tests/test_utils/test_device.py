import pytest
import torch
from traiNNer.utils.device import is_device_oom, resolve_device


def test_resolve_explicit_cpu() -> None:
    assert resolve_device("cpu", 0) == torch.device("cpu")


def test_auto_preserves_num_gpu_cpu_selection() -> None:
    assert resolve_device("auto", 0) == torch.device("cpu")


def test_resolve_explicit_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("traiNNer.utils.device.mps_is_available", lambda: True)
    assert resolve_device("mps", 0) == torch.device("mps")


def test_mps_rejects_cuda_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("traiNNer.utils.device.mps_is_available", lambda: True)
    with pytest.raises(ValueError, match="requires num_gpu"):
        resolve_device("mps", 1)


def test_mps_oom_detection() -> None:
    error = RuntimeError("MPS backend out of memory (MPS allocated: 1.00 GiB)")
    assert is_device_oom(error, torch.device("mps"))
    assert not is_device_oom(error, torch.device("cpu"))
