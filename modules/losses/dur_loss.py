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

    # noinspection PyMethodMayBeStatic
    def pdur2wdur(self, ph_dur, ph2word):
        b = ph_dur.shape[0]
        word_dur = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
            1, ph2word, ph_dur
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        return word_dur

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor) -> Tensor:
        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        # wdur loss
        wdur_pred = self.pdur2wdur(dur_pred, ph2word)
        wdur_gt = self.pdur2wdur(dur_gt, ph2word)
        wdur_loss = self.lambda_wdur * self.loss(self.linear2log(wdur_pred), self.linear2log(wdur_gt))

        # sdur loss
        sdur_pred = dur_pred.sum(dim=1)
        sdur_gt = dur_gt.sum(dim=1)
        sdur_loss = self.lambda_sdur * self.loss(self.linear2log(sdur_pred), self.linear2log(sdur_gt))

        # combine
        dur_loss = pdur_loss + wdur_loss + sdur_loss

        return dur_loss
