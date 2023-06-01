from collections import OrderedDict

import tqdm
import json
import pathlib

import numpy as np
import torch

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import DiffSingerAcoustic
from modules.vocoders.registry import VOCODERS
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import cross_fade, resample_align_curve, save_wav
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class DiffSingerAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None):
        super().__init__(device=device)
        if load_model:
            self.variance_checklist = []

            self.variances_to_embed = set()

            if hparams.get('use_energy_embed', False):
                self.variances_to_embed.add('energy')
            if hparams.get('use_breathiness_embed', False):
                self.variances_to_embed.add('breathiness')

            self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
            if hparams['use_spk_id']:
                with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                    self.spk_map = json.load(f)
                assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
                assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
            self.model = self.build_model(ckpt_steps=ckpt_steps)
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.ph_encoder),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        vocoder.to_device(self.device)
        return vocoder

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'])]).to(self.device)  # => [B, T_txt]
        batch['tokens'] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param['ph_dur'].split(), np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        summary['tokens'] = txt_tokens.size(1)
        summary['frames'] = length
        summary['seconds'] = '%.2f' % (length * self.timestep)

        if hparams['use_spk_id']:
            spk_mix_map = param.get('spk_mix')  # { spk_name: value } or { spk_name: "value value value ..." }
            dynamic = False
            if spk_mix_map is None:
                # Get the first speaker
                for name in self.spk_map.keys():
                    spk_mix_map = {name: 1.0}
                    break
            else:
                for name in spk_mix_map:
                    assert name in self.spk_map, f'Speaker \'{name}\' not found.'

            if len(spk_mix_map) == 1:
                summary['spk'] = list(spk_mix_map.keys())[0]
            elif any([isinstance(val, str) for val in spk_mix_map.values()]):
                print_mix = '|'.join(spk_mix_map.keys())
                summary['spk_mix'] = f'dynamic({print_mix})'
                dynamic = True
            else:
                print_mix = '|'.join([f'{n}:{"%.3f" % spk_mix_map[n]}' for n in spk_mix_map])
                summary['spk_mix'] = f'static({print_mix})'

            spk_mix_id_list = []
            spk_mix_value_list = []
            if dynamic:
                for name, values in spk_mix_map.items():
                    spk_mix_id_list.append(self.spk_map[name])
                    if isinstance(values, str):
                        # this speaker has a variable proportion
                        cur_spk_mix_value = torch.from_numpy(resample_align_curve(
                            np.array(values.split(), 'float32'),
                            original_timestep=float(param['spk_mix_timestep']),
                            target_timestep=self.timestep,
                            align_length=length
                        )).to(self.device)[None]  # => [B=1, T]
                        assert torch.all(cur_spk_mix_value >= 0.), \
                            f'Speaker mix checks failed.\n' \
                            f'Proportions of speaker \'{name}\' on some frames are negative.'
                    else:
                        # this speaker has a constant proportion
                        assert values >= 0., f'Speaker mix checks failed.\n' \
                                             f'Proportion of speaker \'{name}\' is negative.'
                        cur_spk_mix_value = torch.full(
                            (1, length), fill_value=values,
                            dtype=torch.float32, device=self.device
                        )
                    spk_mix_value_list.append(cur_spk_mix_value)
                spk_mix_id = torch.LongTensor(spk_mix_id_list).to(self.device)[None, None]  # => [B=1, 1, N]
                spk_mix_value = torch.stack(spk_mix_value_list, dim=2)  # [B=1, T] => [B=1, T, N]
                spk_mix_value_sum = torch.sum(spk_mix_value, dim=2, keepdim=True)  # => [B=1, T, 1]
                assert torch.all(spk_mix_value_sum > 0.), \
                    f'Speaker mix checks failed.\n' \
                    f'Proportions of speaker mix on some frames sum to zero.'
                spk_mix_value /= spk_mix_value_sum  # normalize
            else:
                for name, value in spk_mix_map.items():
                    spk_mix_id_list.append(self.spk_map[name])
                    assert value >= 0., f'Speaker mix checks failed.\n' \
                                        f'Proportion of speaker \'{name}\' is negative.'
                    spk_mix_value_list.append(value)
                spk_mix_id = torch.LongTensor(spk_mix_id_list).to(self.device)[None, None]  # => [B=1, 1, N]
                spk_mix_value = torch.FloatTensor(spk_mix_value_list).to(self.device)[None, None]  # => [B=1, 1, N]
                spk_mix_value_sum = spk_mix_value.sum()
                assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                               f'Proportions of speaker mix sum to zero.'
                spk_mix_value /= spk_mix_value_sum  # normalize

            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value

        batch['f0'] = torch.from_numpy(resample_align_curve(
            np.array(param['f0_seq'].split(), np.float32),
            original_timestep=float(param['f0_timestep']),
            target_timestep=self.timestep,
            align_length=length
        )).to(self.device)[None]

        for v_name in VARIANCE_CHECKLIST:
            if v_name in self.variances_to_embed:
                batch[v_name] = torch.from_numpy(resample_align_curve(
                    np.array(param[v_name].split(), np.float32),
                    original_timestep=float(param[f'{v_name}_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )).to(self.device)[None]
                summary[v_name] = 'manual'

        if hparams.get('use_key_shift_embed', False):
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            gender = param.get('gender', 0.)
            if isinstance(gender, (int, float, bool)):  # static gender value
                summary['gender'] = f'static({gender:.3f})'
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch['key_shift'] = torch.FloatTensor([key_shift_value]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['gender'] = 'dynamic'
                gender_seq = resample_align_curve(
                    np.array(gender.split(), np.float32),
                    original_timestep=float(param['gender_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                batch['key_shift'] = torch.clip(
                    torch.from_numpy(key_shift_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=shift_min, max=shift_max
                )

        if hparams.get('use_speed_embed', False):
            if param.get('velocity') is None:
                summary['velocity'] = 'default'
                batch['speed'] = torch.FloatTensor([1.]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['velocity'] = 'manual'
                speed_min, speed_max = hparams['augmentation_args']['random_time_stretching']['range']
                speed_seq = resample_align_curve(
                    np.array(param['velocity'].split(), np.float32),
                    original_timestep=float(param['velocity_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                batch['speed'] = torch.clip(
                    torch.from_numpy(speed_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=speed_min, max=speed_max
                )

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))

        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        txt_tokens = sample['tokens']
        variances = {
            v_name: sample.get(v_name)
            for v_name in self.variances_to_embed
        }
        if hparams['use_spk_id']:
            spk_mix_id = sample['spk_mix_id']
            spk_mix_value = sample['spk_mix_value']
            # perform mixing on spk embed
            spk_mix_embed = torch.sum(
                self.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),  # => [B, T, N, H]
                dim=2, keepdim=False
            )  # => [B, T, H]
        else:
            spk_mix_embed = None
        mel_pred = self.model(
            txt_tokens, mel2ph=sample['mel2ph'], f0=sample['f0'], **variances,
            key_shift=sample.get('key_shift'), speed=sample.get('speed'),
            spk_mix_embed=spk_mix_embed, infer=True
        )
        return mel_pred

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            spk_mix: dict[str, float] = None,
            seed: int = -1,
            save_mel: bool = False
    ):
        batches = [self.preprocess_input(param, idx=i) for i, param in enumerate(params)]

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = '.wav' if not save_mel else '.mel.pt'
        for i in range(num_runs):
            if save_mel:
                result = []
            else:
                result = np.zeros(0)
            current_length = 0

            for param, batch in tqdm.tqdm(
                    zip(params, batches), desc='infer segments', total=len(params)
            ):
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                mel_pred = self.forward_model(batch)
                if save_mel:
                    result.append({
                        'offset': param.get('offset', 0.),
                        'mel': mel_pred.cpu(),
                        'f0': batch['f0'].cpu()
                    })
                else:
                    waveform_pred = self.run_vocoder(mel_pred, f0=batch['f0'])[0].cpu().numpy()
                    silent_length = round(param.get('offset', 0) * hparams['audio_sample_rate']) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(result, waveform_pred, current_length + silent_length)
                    current_length = current_length + silent_length + waveform_pred.shape[0]

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}{suffix}'
            else:
                filename = title + suffix
            save_path = out_dir / filename
            if save_mel:
                print(f'| save mel: {save_path}')
                torch.save(result, save_path)
            else:
                print(f'| save audio: {save_path}')
                save_wav(result, save_path, hparams['audio_sample_rate'])
