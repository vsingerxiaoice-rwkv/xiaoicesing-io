import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.losses.curve_loss import CurveLoss2d
from modules.losses.dur_loss import DurationLoss
from modules.toplevel import DiffSingerVariance
from utils.hparams import hparams

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def collater(self, samples):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        midi = utils.collate_nd([s['midi'] for s in samples], 0)
        ph2word = utils.collate_nd([s['ph2word'] for s in samples], 0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        base_pitch = utils.collate_nd([s['base_pitch'] for s in samples], 0)
        delta_pitch = utils.collate_nd([s['delta_pitch'] for s in samples], 0)
        uv = utils.collate_nd([s['uv'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur,
            'midi': midi,
            'ph2word': ph2word,
            'mel2ph': mel2ph,
            'base_pitch': base_pitch,
            'delta_pitch': delta_pitch,
            'uv': uv
        })
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        return batch


class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VarianceDataset

    def build_model(self):
        return DiffSingerVariance(
            vocab_size=len(self.phone_encoder),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses(self):
        if hparams['predict_dur']:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                loss_type=dur_hparams['loss_type'],
                offset=dur_hparams['log_offset']
            )
        if hparams['predict_pitch']:
            pitch_hparams = hparams['pitch_prediction_args']
            self.pitch_loss = CurveLoss2d(
                vmin=pitch_hparams['pitch_delta_vmin'],
                vmax=pitch_hparams['pitch_delta_vmax'],
                num_bins=pitch_hparams['num_pitch_bins'],
                deviation=pitch_hparams['deviation']
            )

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        midi = sample['midi']  # [B, T_ph]
        ph2word = sample['ph2word']  # [B, T_ph]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        mel2ph = sample['mel2ph']  # [B, T_t]
        base_pitch = sample['base_pitch']  # [B, T_t]

        output = self.model(txt_tokens, midi=midi, ph2word=ph2word, ph_dur=ph_dur,
                            mel2ph=mel2ph, base_pitch=base_pitch, infer=infer)

        if infer:
            dur_pred, pitch_pred = output
            return dur_pred, pitch_pred
        else:
            dur_pred_xs, pitch_prob = output
            losses = {}
            if dur_pred_xs is not None:
                losses['dur_loss'] = self.dur_loss.forward(dur_pred_xs, ph_dur)
            if pitch_prob is not None:
                pitch_delta = sample['pitch_delta']
                uv = sample['uv']
                losses['pitch_loss'] = self.pitch_loss.forward(pitch_prob, pitch_delta, ~uv)
            return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        total_loss = sum(losses.values())
        outputs = {
            'total_loss': total_loss
        }

        if batch_idx < hparams['num_valid_plots'] \
                and (self.trainer.distributed_sampler_kwargs or {}).get('rank', 0) == 0:
            dur_pred, pitch_pred = self.run_model(sample, infer=True)
            self.plot_dur(batch_idx, sample['ph_dur'], dur_pred, ph2word=sample['ph2word'])
            self.plot_curve(batch_idx, sample['base_pitch'] + sample['pitch_delta'], pitch_pred, curve_name='pitch')

        return outputs, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, batch_idx, gt_dur, pred_dur, ph2word=None):
        # TODO: plot dur to TensorBoard
        pass

    def plot_curve(self, batch_idx, gt_curve, pred_curve, curve_name='curve'):
        # TODO: plot curve to TensorBoard
        pass
