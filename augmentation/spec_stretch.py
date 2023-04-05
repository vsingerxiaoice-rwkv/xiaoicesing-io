from copy import deepcopy

import numpy as np
import torch

from basics.base_augmentation import BaseAugmentation, require_same_keys
from modules.fastspeech.tts_modules import LengthRegulator
from modules.vocoders.registry import VOCODERS
from utils.binarizer_utils import get_pitch_parselmouth, get_mel2ph_torch
from utils.hparams import hparams


class SpectrogramStretchAugmentation(BaseAugmentation):
    """
    This class contains methods for frequency-domain and time-domain stretching augmentation.
    """
    def __init__(self, data_dirs: list, augmentation_args: dict):
        super().__init__(data_dirs, augmentation_args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = LengthRegulator().to(self.device)

    @require_same_keys
    def process_item(self, item: dict, key_shift=0., speed=1., replace_spk_id=None) -> dict:
        aug_item = deepcopy(item)
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(
                aug_item['wav_fn'], keyshift=key_shift, speed=speed
            )
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(
                aug_item['wav_fn'], keyshift=key_shift, speed=speed
            )

        aug_item['mel'] = mel

        if speed != 1. or hparams.get('use_speed_embed', False):
            aug_item['length'] = mel.shape[0]
            aug_item['speed'] = int(np.round(hparams['hop_size'] * speed)) / hparams['hop_size'] # real speed
            aug_item['seconds'] /= aug_item['speed']
            aug_item['ph_dur'] /= aug_item['speed']
            aug_item['mel2ph'] = get_mel2ph_torch(
                self.lr, torch.from_numpy(aug_item['ph_dur']), aug_item['length'], hparams, device=self.device
            ).cpu().numpy()
            f0, _, _ = get_pitch_parselmouth(
                wav, aug_item['length'], hparams, speed=speed, interp_uv=hparams['interp_uv']
            )
            aug_item['f0'] = f0.astype(np.float32)

        if key_shift != 0. or hparams.get('use_key_shift_embed', False):
            if replace_spk_id is None:
                aug_item['key_shift'] = key_shift
            else:
                aug_item['spk_id'] = replace_spk_id
            aug_item['f0'] *= 2 ** (key_shift / 12)

        return aug_item
