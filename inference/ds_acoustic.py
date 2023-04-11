import json
import pathlib

import numpy as np
import torch

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import DiffSingerAcoustic
from modules.vocoders.registry import VOCODERS
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class DiffSingerAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None):
        super().__init__(device=device)
        if load_model:
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
            self.vocoder.model.eval()
            self.vocoder.model.to(self.device)

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.ph_encoder),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps, required_category='acoustic',
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        vocoder.model.eval()
        vocoder.model.to(self.device)
        return vocoder

    def preprocess_input(self, param):
        """
        :param param: one segment in the .ds file
        :return: batch of the model inputs
        """
        batch = {}
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'])]).to(self.device)  # => [B, T_txt]
        batch['tokens'] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param['ph_dur'].split(), np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        print(f'Length: {txt_tokens.size(1)} token(s), {length} frame(s), {length * self.timestep:.2f} second(s)')

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
                print(f'Using speaker \'{list(spk_mix_map.keys())[0]}\'')
            elif any([isinstance(val, str) for val in spk_mix_map.values()]):
                print_mix = '|'.join(spk_mix_map.keys())
                print(f'Using dynamic speaker mix \'{print_mix}\'')
                dynamic = True
            else:
                print_mix = '|'.join([f'{n}:{"%.3f" % spk_mix_map[n]}' for n in spk_mix_map])
                print(f'Using static speaker mix \'{print_mix}\'')

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

        if hparams.get('use_key_shift_embed', False):
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            gender = param.get('gender', 0.)
            if isinstance(gender, (int, float, bool)):  # static gender value
                print(f'Using static gender value: {gender:.3f}')
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch['key_shift'] = torch.FloatTensor([key_shift_value]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                print('Using dynamic gender curve')
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
                print('Using default velocity value')
                batch['speed'] = torch.FloatTensor([1.]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                print('Using manual velocity curve')
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

        return batch

    @torch.no_grad()
    def run_model(self, sample, return_mel=False):
        txt_tokens = sample['tokens']
        if hparams['use_spk_id']:
            # perform mixing on spk embed
            spk_mix_embed = torch.sum(
                self.model.fs2.spk_embed(sample['spk_mix_id']) * sample['spk_mix_value'].unsqueeze(3),  # => [B, T, N, H]
                dim=2, keepdim=False
            )  # => [B, T, H]
        else:
            spk_mix_embed = None
        mel_pred = self.model(txt_tokens, mel2ph=sample['mel2ph'], f0=sample['f0'],
                              key_shift=sample.get('key_shift'), speed=sample.get('speed'),
                              spk_mix_embed=spk_mix_embed, infer=True)
        return mel_pred

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def infer_once(self, param, return_mel=False):
        batch = self.preprocess_input(param)
        mel = self.run_model(batch, return_mel=True)
        if return_mel:
            return mel.cpu(), batch['f0'].cpu()
        else:
            waveform = self.run_vocoder(mel, f0=batch['f0'])
        return waveform.view(-1).cpu().numpy()
