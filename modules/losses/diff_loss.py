import torch.nn as nn
from torch import Tensor

from modules.losses.ssim import SSIMLoss


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'ssim':
            self.loss = SSIMLoss()
        else:
            raise NotImplementedError()

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
        return self.loss(x_recon, noise)
