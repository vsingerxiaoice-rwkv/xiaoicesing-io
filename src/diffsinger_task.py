import glob
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F

import utils
import matplotlib
from basics.base_dataset import BaseDataset
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.tts_modules import mel2ph_to_dur
from src.vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from utils.cwt import get_lf0_cwt
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import denorm_f0, norm_interp_f0
from .diff.candidate_decoder import FFT
from .diff.diffusion import GaussianDiffusion
from .diff.net import DiffNet
from .diffspeech_task import DiffSpeechTask

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'fft': lambda hp: FFT(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}

matplotlib.use('Agg')


class DiffSingerTask(DiffSpeechTask):
    def __init__(self):
        super(DiffSingerTask, self).__init__()
        self.dataset_cls = AcousticDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().cuda()
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()

    def build_tts_model(self):
        # import torch
        # from tqdm import tqdm
        # v_min = torch.ones([80]) * 100
        # v_max = torch.ones([80]) * -100
        # for i, ds in enumerate(tqdm(self.dataset_cls('train'))):
        #     v_max = torch.max(torch.max(ds['mel'].reshape(-1, 80), 0)[0], v_max)
        #     v_min = torch.min(torch.min(ds['mel'].reshape(-1, 80), 0)[0], v_min)
        #     if i % 100 == 0:
        #         print(i, v_min, v_max)
        # print('final', v_min, v_max)
        mel_bins = hparams['audio_num_mel_bins']
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        if hparams.get('fs2_ckpt', '') != '':
            utils.load_ckpt(self.model.fs2, hparams['fs2_ckpt'], 'model', strict=True)
            # self.model.fs2.decoder = None
            for k, v in self.model.fs2.named_parameters():
                v.requires_grad = False

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)


        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=None, infer=True)

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = model_out.get('f0_denorm')
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], gt_f0=gt_f0, pred_f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            self.plot_mel(batch_idx, sample['mels'], model_out['fs2_mel'], name=f'fs2mel_{batch_idx}')
        return outputs


class AcousticDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if hparams['pitch_type'] == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        # energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "pitch": pitch,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if self.hparams['use_energy_embed']:
            sample['energy'] = item['energy']
        if self.hparams.get('use_key_shift_embed', False):
            sample['key_shift'] = item['key_shift']
        if self.hparams.get('use_speed_embed', False):
            sample['speed'] = item['speed']
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if self.hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
        if self.hparams['pitch_type'] == 'cwt':
            cwt_spec = torch.Tensor(item['cwt_spec'])[:max_frames]
            f0_mean = item.get('f0_mean', item.get('cwt_mean'))
            f0_std = item.get('f0_std', item.get('cwt_std'))
            sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        elif self.hparams['pitch_type'] == 'ph':
            f0_phlevel_sum = torch.zeros_like(phone).float().scatter_add(0, mel2ph - 1, f0)
            f0_phlevel_num = torch.zeros_like(phone).float().scatter_add(
                0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
            sample["f0_ph"] = f0_phlevel_sum / f0_phlevel_num
        item = self._get_item(index)
        sample['pitch_midi'] = torch.LongTensor(item['pitch_midi'])[:hparams['max_frames']]
        sample['midi_dur'] = torch.FloatTensor(item['midi_dur'])[:hparams['max_frames']]
        sample['is_slur'] = torch.LongTensor(item['is_slur'])[:hparams['max_frames']]
        sample['word_boundary'] = torch.LongTensor(item['word_boundary'])[:hparams['max_frames']]
        return sample

    def collater(self, samples):
        from preprocessing.opencpop import File2Batch
        return File2Batch.processed_input2batch(samples)

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'basics.base_binarizer.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes


class DiffSingerMIDITask(DiffSingerTask):
    def __init__(self):
        super(DiffSingerMIDITask, self).__init__()
        self.dataset_cls = AcousticDataset

    def run_model(self, model, sample, return_output=False, infer=False):
        '''
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        '''
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph'] # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample.get('energy')

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            # NOTE: this part of script is *isolated* from other scripts, which means
            #       it may not be compatible with the current version.    
            pass
            # cwt_spec = sample[f'cwt_spec']
            # f0_mean = sample['f0_mean']
            # f0_std = sample['f0_std']
            # sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        # output == ret
        # model == src.diff.diffusion.GaussianDiffusion
        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer, pitch_midi=sample['pitch_midi'],
                       midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, sample['word_boundary'], losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample.get('energy')
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=None, uv=None, energy=energy, ref_mels=None, infer=True,
                pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = model_out.get('f0_denorm')
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], gt_f0=gt_f0, pred_f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            #self.plot_mel(batch_idx, sample['mels'], model_out['fs2_mel'], name=f'fs2mel_{batch_idx}')
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
            the effect of each loss component:
                hparams['dur_loss'] : align each phoneme
                hparams['lambda_word_dur']: align each word
                hparams['lambda_sent_dur']: align each sentence
            
            :param dur_pred: [B, T], float, log scale
            :param mel2ph: [B, T]
            :param txt_tokens: [B, T]
            :param losses:
            :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            #idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            idx = wdb.cumsum(axis=1)
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_pred)
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_gt)
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']
