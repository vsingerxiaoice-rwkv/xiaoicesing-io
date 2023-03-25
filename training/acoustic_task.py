import os
from multiprocessing.pool import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
import lightning.pytorch as pl
from tqdm import tqdm

import utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.fastspeech.tts_modules import mel2ph_to_dur
from modules.toplevel.acoustic_model import DiffSingerAcoustic
from modules.vocoders.registry import get_vocoder_cls
from utils import audio
from utils.binarizer_utils import get_pitch_parselmouth
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.phoneme_utils import build_phoneme_list
from utils.plot import spec_to_figure
from utils.text_encoder import TokenTextEncoder
from utils.training_utils import WarmupCosineSchedule

matplotlib.use('Agg')


class AcousticDataset(BaseDataset):
    def __init__(self, prefix):
        super().__init__()
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.sizes = np.load(os.path.join(self.data_dir, f'{self.prefix}.lengths'))
        self.indexed_ds = IndexedDataset(self.data_dir, self.prefix)

    def __getitem__(self, index):
        return self.indexed_ds[index]

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        mel = utils.collate_nd([s['mel'] for s in samples], 0.0)
        batch = {
            'size': len(samples),
            'tokens': tokens,
            'mel2ph': mel2ph,
            'mel': mel,
            'f0': f0,
        }
        # if hparams['use_energy_embed']:
        #     batch['energy'] = utils.collate_nd([s['energy'] for s in samples], 0.0)
        if hparams.get('use_key_shift_embed', False):
            batch['key_shift'] = torch.FloatTensor([float(s['key_shift']) for s in samples])[:, None]
        if hparams.get('use_speed_embed', False):
            batch['speed'] = torch.FloatTensor([float(s['speed']) for s in samples])[:, None]
        # if hparams['use_spk_embed']:
        #     spk_embed = torch.stack([s['spk_embed'] for s in samples])
        #     batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([int(s['spk_id']) for s in samples])
            batch['spk_ids'] = spk_ids
        return batch

class AcousticTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = AcousticDataset
        self.use_vocoder = hparams['infer'] or hparams.get('val_with_vocoder', True)
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        self.logged_gt_wav = set()
        
    def setup(self, stage):
        self.phone_encoder = self.build_phone_encoder()
        self.model = self.build_model()
        self.train_dataset = self.dataset_cls(hparams['train_set_name'])
        self.valid_dataset = self.dataset_cls(hparams['valid_set_name'])

    @staticmethod
    def build_phone_encoder():
        phone_list = build_phoneme_list()
        return TokenTextEncoder(vocab_list=phone_list)

    def build_model(self):
        model = DiffSingerAcoustic(
            vocab_size=len(self.phone_encoder),
            out_dims=hparams['audio_num_mel_bins']
        )
        if self.trainer.local_rank == 0:
            utils.print_arch(model)
        return model

    def build_optimizer(self, model):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def build_scheduler(self, optimizer):
        # return WarmupCosineSchedule(optimizer,
        #                             warmup_steps=hparams['warmup_updates'],
        #                             t_total=hparams['max_updates'] // hparams['accumulate_grad_batches'],
        #                             eta_min=0)
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=hparams.get('gamma', 0.5))
    
    def train_dataloader(self):
        self.training_sampler = self.build_batch_sampler(self.train_dataset,
                                                         max_tokens=self.max_tokens,
                                                         max_sentences=self.max_sentences,
                                                         shuffle=True)
        return torch.utils.data.DataLoader(self.train_dataset,
                                           collate_fn=self.train_dataset.collater,
                                           batch_sampler=self.training_sampler,
                                           num_workers=int(hparams.get('ds_workers', os.getenv('NUM_WORKERS', 1))),
                                           prefetch_factor=hparams.get('dataloader_prefetch_factor', 2),
                                           pin_memory=True,
                                           persistent_workers=True)

    def val_dataloader(self):
        sampler = self.build_batch_sampler(self.valid_dataset,
                                           max_tokens=self.max_tokens,
                                           max_sentences=self.max_sentences)
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           collate_fn=self.valid_dataset.collater,
                                           batch_sampler=sampler,
                                           num_workers=int(hparams.get('ds_workers', os.getenv('NUM_WORKERS', 1))),
                                           prefetch_factor=hparams.get('dataloader_prefetch_factor', 2),
                                           shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()
    
    def run_model(self, sample, return_output=False, infer=False):
        """
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        """
        txt_tokens = sample['tokens']  # [B, T_t]
        target = sample['mel']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')

        if hparams['use_spk_id']:
            spk_embed_id = sample['spk_ids']
        # elif hparams['use_spk_embed']:
        #     spk_embed = sample['spk_embed']
        else:
            spk_embed_id = None
        output = self.model(txt_tokens, mel2ph=mel2ph, f0=f0,
                            key_shift=key_shift, speed=speed,
                            spk_embed_id=spk_embed_id,
                            gt_mel=target, infer=infer)

        losses = {}
        if not infer:
            losses['mel'] = output
        if not return_output:
            return losses
        else:
            return losses, output

    def _training_step(self, sample, batch_idx, _):
        log_outputs = self.run_model(sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample['tokens'].size()[0]
        log_outputs['lr'] = self.lr_schedulers().get_lr()[0]
        return total_loss, log_outputs

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, return_output=False, infer=False)
        total_loss = sum(losses.values())
        outputs = {
            'losses': losses,
            'total_loss': total_loss,
            'size': sample['size']
        }
        outputs = utils.tensors_to_scalars(outputs)

        if batch_idx < hparams['num_valid_plots']:
            _, mel_pred = self.run_model(sample, return_output=True, infer=True)
            if self.use_vocoder:
                self.plot_wav(batch_idx, sample['mel'], mel_pred, f0=sample['f0'])
            self.plot_mel(batch_idx, sample['mel'], mel_pred, name=f'diffmel_{batch_idx}')

        return outputs

    def _on_validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['size']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_mel, pred_mel, f0=None):
        gt_mel = gt_mel[0].cpu().numpy()
        pred_mel = pred_mel[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if batch_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav(gt_mel, f0=f0)
            self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                             global_step=self.global_step)
            self.logged_gt_wav.add(batch_idx)
        pred_wav = self.vocoder.spec2wav(pred_mel, f0=f0)
        self.logger.experiment.add_audio(f'pred_{batch_idx}', pred_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    ############
    # infer
    ############
    def on_test_start(self):
        self.saving_result_pool = Pool(8)
        self.saving_results_futures = []
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def test_step(self, sample, batch_idx):
        _, mel_pred = self.run_model(sample, return_output=True, infer=True)
        sample['outputs'] = mel_pred
        return self.after_infer(sample)

    def on_test_end(self):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    def after_infer(self, predictions):
        if self.saving_result_pool is None:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')
            text = prediction.get('text').replace(':', '%3A')[:80]

            # remove paddings
            mel_gt = prediction['mel']
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel2ph_gt = prediction.get('mel2ph')
            mel2ph_gt = mel2ph_gt[mel_gt_mask] if mel2ph_gt is not None else None
            mel_pred = prediction['outputs']
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            mel2ph_pred = prediction.get('mel2ph_pred')
            if mel2ph_pred is not None:
                if len(mel2ph_pred) > len(mel_pred_mask):
                    mel2ph_pred = mel2ph_pred[:len(mel_pred_mask)]
                mel2ph_pred = mel2ph_pred[mel_pred_mask]

            f0_gt = prediction.get('f0')
            f0_pred = prediction.get('f0_pred')
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]

            str_phs = None
            if self.phone_encoder is not None and 'tokens' in prediction:
                str_phs = self.phone_encoder.decode(prediction['tokens'], strip_padding=True)
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            os.makedirs(os.path.join(hparams['work_dir'], 'P_mels_npy'), exist_ok=True)
            os.makedirs(os.path.join(hparams['work_dir'], 'G_mels_npy'), exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, 'P', item_name, text, gen_dir, str_phs, mel2ph_pred, f0_gt, f0_pred]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, 'G', item_name, text, gen_dir, str_phs, mel2ph_gt, f0_gt, f0_pred]))
                if hparams['save_f0']:
                    import matplotlib.pyplot as plt
                    # f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_pred_ = f0_pred
                    f0_gt_, _, _ = get_pitch_parselmouth(wav_gt, len(mel_gt), hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label=r'$f0_P$')
                    plt.plot(f0_gt_, label=r'$f0_G$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)

            t.set_description(
                f'Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}')

        return {}

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, str_phs=None, mel2ph=None, gt_f0=None,
                    pred_f0=None):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'

        if text is not None:
            base_fn += text
        base_fn += ('-' + hparams['exp_name'])
        np.save(os.path.join(hparams['work_dir'], f'{prefix}_mels_npy', item_name), mel)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _, _ = get_pitch_parselmouth(wav_out, len(mel), hparams)
        f0 = (f0 - 100) / (800 - 100) * 80 * (f0 > 0)
        plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(' ')
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)
