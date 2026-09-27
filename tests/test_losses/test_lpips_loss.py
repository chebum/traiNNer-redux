import torch
from traiNNer.losses.lpips_loss import LPIPSLoss


def test_lpips_loss_supports_training_batch_and_backpropagates() -> None:
    loss = LPIPSLoss(loss_weight=1.0, net="alex", pnet_rand=True)
    prediction = torch.rand(8, 3, 32, 32, requires_grad=True)
    target = torch.rand_like(prediction)

    value = loss(prediction, target)
    value.backward()

    assert value.ndim == 0
    assert value >= 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert all(parameter.grad is None for parameter in loss.parameters())
