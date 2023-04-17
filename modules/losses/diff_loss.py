import torch.nn as nn
from torch import Tensor

from modules.losses.common_losses import TVLoss


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_nonpadding(x_recon, noise, nonpadding=None):
        if nonpadding is not None:
            nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)
            return x_recon * nonpadding, noise * nonpadding
        else:
            return x_recon, noise

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, nonpadding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise).mean()


class DiffusionNoiseWithSmoothnessLoss(DiffusionNoiseLoss):
    def __init__(self, loss_type, lambda_tv=0.5):
        super().__init__(loss_type)
        self.lambda_tv = lambda_tv
        self.tv_loss = TVLoss()

    def forward(self, x_recon, noise, nonpadding=None):
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise).mean() + self.lambda_tv * self.tv_loss(x_recon - noise)


class DiffusionNoiseWithSensitivityLoss(DiffusionNoiseLoss):
    def __init__(self, loss_type, alpha=1):
        super().__init__(loss_type)
        self.alpha = alpha

    def forward(self, x_recon, noise, nonpadding=None, reference=None):
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        loss = self._forward(x_recon, noise)
        if reference is not None:
            difference = reference.diff(dim=1, prepend=reference[:, 0]).abs()
            sensitivity = 1 / (1 + self.alpha * difference)
            loss = loss * sensitivity.transpose(1, 2).unsqueeze(1)
        return loss.mean()
