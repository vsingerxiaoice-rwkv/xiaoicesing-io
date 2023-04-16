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
from modules.losses.diff_loss import DiffusionNoiseLoss
from modules.losses.dur_loss import DurationLoss
from modules.toplevel import DiffSingerVariance
from utils.hparams import hparams
from utils.plot import dur_to_figure, f0_to_figure

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
        uv = utils.collate_nd([s['uv'] for s in samples], True)
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
        if hparams['predict_dur']:
            self.lambda_dur_loss = hparams['lambda_dur_loss']
        if hparams['predict_pitch']:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']

    def build_model(self):
        return DiffSingerVariance(
            vocab_size=len(self.phone_encoder),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses(self):
        if hparams['predict_dur']:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
        if hparams['predict_pitch']:
            self.pitch_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type']
            )
            # pitch_hparams = hparams['pitch_prediction_args']
            # self.pitch_loss = CurveLoss2d(
            #     vmin=pitch_hparams['pitch_delta_vmin'],
            #     vmax=pitch_hparams['pitch_delta_vmax'],
            #     num_bins=pitch_hparams['num_pitch_bins'],
            #     deviation=pitch_hparams['deviation']
            # )

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        midi = sample['midi']  # [B, T_ph]
        ph2word = sample['ph2word']  # [B, T_ph]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        mel2ph = sample['mel2ph']  # [B, T_t]
        base_pitch = sample['base_pitch']  # [B, T_t]
        delta_pitch = sample['delta_pitch']  # [B, T_t]

        output = self.model(txt_tokens, midi=midi, ph2word=ph2word, ph_dur=ph_dur,
                            mel2ph=mel2ph, base_pitch=base_pitch, delta_pitch=delta_pitch,
                            infer=infer)

        if infer:
            dur_pred, pitch_pred = output
            return dur_pred, pitch_pred
        else:
            dur_pred, pitch_pred_out = output
            losses = {}
            if dur_pred is not None:
                losses['dur_loss'] = self.lambda_dur_loss * self.dur_loss(dur_pred, ph_dur, ph2word=ph2word)
            if pitch_pred_out is not None:
                pitch_x_recon, pitch_noise = pitch_pred_out
                losses['pitch_loss'] = self.pitch_loss.forward(pitch_x_recon, pitch_noise)
                # delta_pitch = sample['delta_pitch']
                # uv = sample['uv']
                # losses['pitch_loss'] = self.lambda_pitch_loss * self.pitch_loss(pitch_prob, delta_pitch, ~uv)
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
            if dur_pred is not None:
                self.plot_dur(batch_idx, sample['ph_dur'], dur_pred, txt=sample['tokens'])
            if pitch_pred is not None:
                base_pitch = sample['base_pitch']
                self.plot_curve(
                    batch_idx, base_pitch + sample['delta_pitch'], base_pitch + pitch_pred, curve_name='pitch'
                )

        return outputs, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, batch_idx, gt_dur, pred_dur, txt=None):
        name = f'dur_{batch_idx}'
        gt_dur = gt_dur[0].cpu().numpy()
        pred_dur = pred_dur[0].cpu().numpy()
        txt = self.phone_encoder.decode(txt[0].cpu().numpy()).split()
        self.logger.experiment.add_figure(name, dur_to_figure(gt_dur, pred_dur, txt), self.global_step)

    def plot_curve(self, batch_idx, gt_curve, pred_curve, curve_name='curve'):
        name = f'{curve_name}_{batch_idx}'
        gt_curve = gt_curve[0].cpu().numpy()
        pred_curve = pred_curve[0].cpu().numpy()
        # self.logger.experiment.add_figure(name, spec_to_figure(pred_curve, vmin=0, vmax=1), self.global_step)
        self.logger.experiment.add_figure(name, f0_to_figure(gt_curve, pred_curve), self.global_step)
