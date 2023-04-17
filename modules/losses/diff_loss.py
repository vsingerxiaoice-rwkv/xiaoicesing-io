import torch.nn as nn
from torch import Tensor

from modules.losses.common_losses import TVLoss


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type, lambda_tv=0):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.main_loss = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.main_loss = nn.MSELoss()
        else:
            raise NotImplementedError()
        self.lambda_tv = lambda_tv
        if lambda_tv > 0:
            self.tv_loss = TVLoss()

    def forward(self, x_recon: Tensor, noise: Tensor, nonpadding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        if nonpadding is not None:
            nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)
            x_recon = x_recon * nonpadding
            noise *= nonpadding
        main_loss = self.main_loss(x_recon, noise)
        if self.lambda_tv > 0:
            return main_loss + self.lambda_tv * self.tv_loss(x_recon)
        else:
            return main_loss
