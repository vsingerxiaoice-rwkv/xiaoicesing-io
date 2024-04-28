import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.losses import DurationLoss, DiffusionLoss, RectifiedFlowLoss
from modules.losses.dur_vae_loss import VAEDurationLoss
from modules.metrics.curve import RawCurveAccuracy
from modules.metrics.duration import RhythmCorrectness, PhonemeDurationAccuracy
from modules.toplevel import DiffSingerVariance, DURVAE
from utils.hparams import hparams
from utils.plot import dur_to_figure, pitch_note_to_figure, curve_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(VarianceDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur
        })



        batch['ph2word'] = utils.collate_nd([s['ph2word'] for s in samples], 0)
        batch['midi'] = utils.collate_nd([s['midi'] for s in samples], 0)

        return batch





class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VarianceDataset

        self.diffusion_type = hparams['diffusion_type']

        self.use_spk_id = hparams['use_spk_id']



        self.predict_pitch = hparams['predict_pitch']


        super()._finish_init()

    def _build_model(self):
        return DURVAE(

        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        # if self.predict_dur:
        dur_hparams = hparams['dur_prediction_args']
        self.log_offset=dur_hparams['log_offset']
        self.dur_loss = VAEDurationLoss(
            offset=dur_hparams['log_offset'],
            loss_type=dur_hparams['loss_type'],
            lambda_pdur=dur_hparams['lambda_pdur_loss'],
            lambda_wdur=dur_hparams['lambda_wdur_loss'],
            lambda_sdur=dur_hparams['lambda_sdur_loss'],
            lambda_KL=dur_hparams['lambda_KL_loss'],
        )
        self.register_validation_loss('dur_loss')
        self.register_validation_loss('KL_loss')
        self.register_validation_metric('rhythm_corr', RhythmCorrectness(tolerance=0.05))
        self.register_validation_metric('ph_dur_acc', PhonemeDurationAccuracy(tolerance=0.00002))


    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_txt]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        ph2word = sample.get('ph2word')  # [B, T_ph]
        mask=(txt_tokens!=0).float().unsqueeze(1)


        dur_pred,mean,log=self.model(torch.log(ph_dur + self.log_offset)/6,mask)
        if infer:
            dur_pred=(dur_pred*6).exp()-self.log_offset
            if dur_pred is not None:
                dur_pred = dur_pred.round().long()
            return dur_pred  # Tensor, Tensor, Dict[str, Tensor]
        else:
            losses = {}
            if dur_pred is not None:
                dur_pred = (dur_pred*6).exp()-self.log_offset
                losses['dur_loss'],losses['KL_loss'] = self.dur_loss(dur_pred, ph_dur, ph2word=ph2word,mean=mean,log=log)


            return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if min(sample['indices']) < hparams['num_valid_plots']:
            def sample_get(key, idx, abs_idx):
                return sample[key][idx][:self.valid_dataset.metadata[key][abs_idx]].unsqueeze(0)

            dur_preds = self.run_model(sample, infer=True)
            for i in range(len(sample['indices'])):
                data_idx = sample['indices'][i]
                if data_idx < hparams['num_valid_plots']:
                    if dur_preds is not None:
                        dur_len = self.valid_dataset.metadata['ph_dur'][data_idx]
                        tokens = sample_get('tokens', i, data_idx)
                        gt_dur = sample_get('ph_dur', i, data_idx)
                        pred_dur = dur_preds[i][:dur_len].unsqueeze(0)
                        ph2word = sample_get('ph2word', i, data_idx)
                        mask = tokens != 0
                        self.valid_metrics['rhythm_corr'].update(
                            pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                        )
                        self.valid_metrics['ph_dur_acc'].update(
                            pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                        )
                        self.plot_dur(data_idx, gt_dur, pred_dur, tokens)

        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, data_idx, gt_dur, pred_dur, txt=None):
        gt_dur = gt_dur[0].cpu().numpy()
        pred_dur = pred_dur[0].cpu().numpy()
        txt = self.phone_encoder.decode(txt[0].cpu().numpy()).split()
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'dur_{data_idx}', dur_to_figure(
            gt_dur, pred_dur, txt, title_text
        ), self.global_step)


