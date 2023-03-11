import warnings

warnings.filterwarnings("ignore")

import parselmouth
from utils.pitch_utils import f0_to_coarse
import numpy as np


def get_pitch_parselmouth(wav_data, mel, hparams, speed=1):
    """

    :param wav_data: [T]
    :param mel: [T, mel_bins]
    :param hparams:
    :return:
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))

    time_step = hop_size / hparams['audio_sample_rate'] * 1000
    f0_min = 65
    f0_max = 800

    f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (int(len(wav_data) // hop_size) - len(f0) + 1) // 2
    f0 = np.pad(f0, [[pad_size, len(mel) - len(f0) - pad_size]], mode='constant')
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse
