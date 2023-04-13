import warnings

import torch

warnings.filterwarnings("ignore")

import parselmouth
from utils.pitch_utils import interp_f0
import numpy as np


def get_pitch_parselmouth(wav_data, length, hparams, speed=1, interp_uv=False):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, f0_coarse, uv
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))

    time_step = hop_size / hparams['audio_sample_rate']
    f0_min = 65
    f0_max = 800

    # noinspection PyArgumentList
    f0 = parselmouth.Sound(wav_data, sampling_frequency=hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    len_f0 = f0.shape[0]
    lpad = (int(len(wav_data) // hop_size) - len_f0 + 1) // 2
    rpad = length - len_f0 - lpad
    if lpad < 0:
        f0 = f0[-lpad:]
        lpad = 0
    if rpad < 0:
        f0 = f0[:rpad]
        rpad = 0
    if lpad > 0 or rpad > 0:
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


@torch.no_grad()
def get_mel2ph_torch(lr, durs, length, timestep, device='cpu'):
    ph_acc = torch.round(torch.cumsum(durs.to(device), dim=0) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(device))
    mel2ph = lr(ph_dur[None])[0]
    num_frames = mel2ph.shape[0]
    if num_frames < length:
        mel2ph = torch.cat((mel2ph, torch.full((length - num_frames,), fill_value=mel2ph[-1], device=device)), dim=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph
