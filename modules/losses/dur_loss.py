import torch
import torch.nn as nn
from torch import Tensor


class DurationLoss(nn.Module):
    def __init__(self, loss_type, offset=1.0):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset

    def forward(self, xs_pred: Tensor, xs_gt: Tensor) -> Tensor:
        xs_gt_log = torch.log(xs_gt + self.offset)  # calculate in log domain
        return self.loss(xs_pred, xs_gt_log)
