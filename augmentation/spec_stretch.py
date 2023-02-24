from copy import deepcopy

import numpy as np
import torch

from basics.base_augmentation import BaseAugmentation
from data_gen.data_gen_utils import get_pitch_parselmouth
from modules.fastspeech.tts_modules import LengthRegulator
from src.vocoders.base_vocoder import VOCODERS
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse


class SpectrogramStretchAugmentation(BaseAugmentation):
    """
    This class contains methods for frequency-domain and time-domain stretching augmentation.
    """
    def __init__(self, data_dirs: list, augmentation_args: dict):
        super().__init__(data_dirs, augmentation_args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = LengthRegulator().to(self.device)

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
            aug_item['len'] = len(mel)
            aug_item['speed'] = int(np.round(hparams['hop_size'] * speed)) / hparams['hop_size'] # real speed
            aug_item['sec'] /= aug_item['speed']
            aug_item['ph_durs'] /= aug_item['speed']
            aug_item['mel2ph'] = self.get_mel2ph(aug_item['ph_durs'], aug_item['len'])
            aug_item['f0'], aug_item['pitch'] = get_pitch_parselmouth(wav, mel, hparams, speed=speed)

        if key_shift != 0. or hparams.get('use_key_shift_embed', False):
            aug_item['key_shift'] = key_shift
            aug_item['f0'] *= 2 ** (key_shift / 12)
            aug_item['pitch'] = f0_to_coarse(aug_item['f0'])

        if replace_spk_id is not None:
            aug_item['spk_id'] = replace_spk_id

        return aug_item

    @torch.no_grad()
    def get_mel2ph(self, durs, length):
        ph_acc = np.around(
            np.add.accumulate(durs) * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5
        ).astype('int')
        ph_dur = np.diff(ph_acc, prepend=0)
        ph_dur = torch.LongTensor(ph_dur)[None].to(self.device)
        mel2ph = self.lr(ph_dur).cpu().numpy()[0]
        num_frames = len(mel2ph)
        if num_frames < length:
            mel2ph = np.concatenate((mel2ph, np.full((length - num_frames, mel2ph[-1]))), axis=0)
        elif num_frames > length:
            mel2ph = mel2ph[:length]
        return mel2ph
