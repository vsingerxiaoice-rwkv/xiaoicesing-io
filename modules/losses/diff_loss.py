import torch.nn as nn
from torch import Tensor


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

    def forward(self, x_recon: Tensor, noise: Tensor, nonpadding: Tensor = None) -> Tensor:
        if nonpadding is not None:
            nonpadding = nonpadding.unsqueeze(1)
            x_recon *= nonpadding
            noise *= nonpadding
        return self.loss(x_recon, noise)
