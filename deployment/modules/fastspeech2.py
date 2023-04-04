import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from utils.hparams import hparams
from utils.pitch_utils import (
    f0_bin, f0_mel_min, f0_mel_max
)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    return f0_coarse


class LengthRegulator(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur):
        token_idx = torch.arange(1, dur.shape[1] + 1, device=dur.device)[None, :, None]
        dur_cumsum = torch.cumsum(dur, dim=1)
        dur_cumsum_prev = F.pad(dur_cumsum, (1, -1), mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(dim=1).max(), device=dur.device)[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask).sum(dim=1)
        return mel2ph


class FastSpeech2AcousticOnnx(FastSpeech2Acoustic):
    def __init__(self, vocab_size, frozen_gender=None, frozen_spk_embed=None):
        super().__init__(vocab_size=vocab_size)
        self.lr = LengthRegulator()
        if hparams.get('use_key_shift_embed', False):
            self.shift_min, self.shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
        if hparams.get('use_speed_embed', False):
            self.speed_min, self.speed_max = hparams['augmentation_args']['random_time_stretching']['range']
        self.frozen_gender = frozen_gender
        self.frozen_spk_embed = frozen_spk_embed

    # noinspection PyMethodOverriding
    def forward(self, tokens, durations, f0, gender=None, velocity=None, spk_embed=None):
        durations = durations * (tokens > 0)
        mel2ph = self.lr(durations)
        f0 = f0 * (mel2ph > 0)
        mel2ph = mel2ph[..., None].repeat((1, 1, hparams['hidden_size']))
        dur_embed = self.dur_embed(durations.float()[:, :, None])
        encoded = self.encoder(tokens, dur_embed)
        encoded = F.pad(encoded, (0, 0, 1, 0))
        condition = torch.gather(encoded, 1, mel2ph)

        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition += pitch_embed

        if hparams.get('use_key_shift_embed', False):
            if self.frozen_gender is not None:
                # noinspection PyUnresolvedReferences, PyTypeChecker
                key_shift = frozen_gender * self.shift_max \
                    if frozen_gender >= 0. else frozen_gender * abs(self.shift_min)
                key_shift_embed = self.key_shift_embed(key_shift[:, None, None])
            else:
                gender = torch.clip(gender, min=-1., max=1.)
                gender_mask = (gender < 0.).float()
                key_shift = gender * ((1. - gender_mask) * self.shift_max + gender_mask * abs(self.shift_min))
                key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition += key_shift_embed

        if hparams.get('use_speed_embed', False):
            if velocity is not None:
                velocity = torch.clip(velocity, min=self.speed_min, max=self.speed_max)
                speed_embed = self.speed_embed(velocity[:, :, None])
            else:
                speed_embed = self.speed_embed(torch.FloatTensor([1.]).to(condition.device)[:, None, None])
            condition += speed_embed

        if hparams['use_spk_id']:
            if self.frozen_spk_embed is not None:
                condition += self.frozen_spk_embed
            else:
                condition += spk_embed
        return condition
