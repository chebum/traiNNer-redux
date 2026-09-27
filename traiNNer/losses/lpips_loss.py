from typing import Literal

from torch import Tensor, nn

from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """Learned perceptual loss for RGB tensors in the [0, 1] range."""

    def __init__(
        self,
        loss_weight: float,
        net: Literal["alex", "vgg", "squeeze"] = "vgg",
        pnet_rand: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.metric = LPIPS(
            net=net,
            pnet_rand=pnet_rand,
            # Bundled LPIPS checkpoints use ``model.1.weight`` keys because
            # the original calibration layers include dropout. Eval mode
            # disables dropout while preserving checkpoint key compatibility.
            use_dropout=True,
            eval_mode=True,
        )
        for parameter in self.metric.parameters():
            parameter.requires_grad = False

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.metric(pred, target.detach(), normalize=True).mean()
