import torch
import torch.nn as nn
from torch import Tensor


class CurveLoss2d(nn.Module):
    """
    Loss module for parameter curve represented by gaussian-blurred 2-D probability bins.
    """

    def __init__(self, vmin, vmax, num_bins, deviation):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.interval = (vmax - vmin) / (num_bins - 1)  # align with centers of bins
        self.sigma = deviation / self.interval
        self.register_buffer('x', torch.arange(num_bins).float().reshape(1, 1, -1))  # [1, 1, N]
        self.loss = nn.BCEWithLogitsLoss()

    def values_to_bins(self, values: Tensor) -> Tensor:
        return (values - self.vmin) / self.interval

    def curve_to_probs(self, curve: Tensor) -> Tensor:
        miu = self.values_to_bins(curve)[:, :, None]  # [B, T, 1]
        probs = (((self.x - miu) / self.sigma) ** 2 / -2).exp()  # gaussian blur, [B, T, N]
        return probs

    def forward(self, y_pred: Tensor, c_gt: Tensor, mask: Tensor = None) -> Tensor:
        """
        Calculate BCE with logits loss between predicted probs and gaussian-blurred bins representing gt curve.
        :param y_pred: predicted probs [B, T, N]
        :param c_gt: ground truth curve [B, T]
        :param mask: (bool) mask of valid parts in ground truth curve [B, T]
        """
        y_gt = self.curve_to_probs(c_gt)
        return self.loss(y_pred, y_gt * mask[:, :, None])
