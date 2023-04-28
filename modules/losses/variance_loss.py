import torch
import torch.nn as nn
from torch import Tensor


class DurationLoss(nn.Module):
    """
    Loss module as combination of phone duration loss, word duration loss and sentence duration loss.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset

        self.lambda_pdur = lambda_pdur
        self.lambda_wdur = lambda_wdur
        self.lambda_sdur = lambda_sdur

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor) -> Tensor:
        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

        # wdur loss
        shape = dur_pred.shape[0], ph2word.max() + 1
        wdur_pred = dur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, dur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_gt = dur_gt.new_zeros(*shape).scatter_add(
            1, ph2word, dur_gt
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_loss = self.lambda_wdur * self.loss(self.linear2log(wdur_pred), self.linear2log(wdur_gt))

        # sdur loss
        sdur_pred = dur_pred.sum(dim=1)
        sdur_gt = dur_gt.sum(dim=1)
        sdur_loss = self.lambda_sdur * self.loss(self.linear2log(sdur_pred), self.linear2log(sdur_gt))

        # combine
        dur_loss = pdur_loss + wdur_loss + sdur_loss

        return dur_loss


class CurveLoss1d(nn.Module):
    """
    Loss module for 1d parameter curve with non-padding masks.
    """
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss(reduction='none')
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss(reduction='none')
        else:
            raise NotImplementedError()

    def forward(self, c_pred, c_gt, mask=None):
        loss = self.loss(c_pred, c_gt)
        if mask is not None:
            loss *= mask
        return loss.mean()


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
