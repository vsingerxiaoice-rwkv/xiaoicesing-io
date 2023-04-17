import torch.nn as nn


class TVLoss(nn.Module):
    """
    Adapted from https://github.com/jxgu1016/Total_Variation_Loss.pytorch
    """

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        """
        b, c, h_x, w_x, *_ = x.shape
        count_h = c * (h_x - 1) * w_x
        count_w = c * h_x * (w_x - 1)
        h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
        return self.weight * 2 * (
            (h_tv / count_h if count_h > 0 else 0) + (w_tv / count_w if count_w > 0 else 0)
        ) / b
