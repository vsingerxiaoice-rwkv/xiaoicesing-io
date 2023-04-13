import os
from multiprocessing.pool import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
from tqdm import tqdm

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.fastspeech.tts_modules import mel2ph_to_dur
from modules.losses.dur_loss import DurationLoss
from modules.toplevel import DiffSingerVariance
from modules.vocoders.registry import get_vocoder_cls
from utils.binarizer_utils import get_pitch_parselmouth
from utils.hparams import hparams
from utils.plot import spec_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def collater(self, samples):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        ph_midi = utils.collate_nd([s['ph_midi'] for s in samples], 0)
        midi_dur = utils.collate_nd([s['word_dur'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur,
            'midi': ph_midi,
            'midi_dur': midi_dur
        })
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        return batch


class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VarianceDataset

    def build_model(self):
        # return DiffSingerVariance()
        raise NotImplementedError()

    # noinspection PyAttributeOutsideInit
    def build_losses(self):
        self.dur_loss = DurationLoss(
            loss_type=hparams['dur_loss_type'],
            offset=hparams['dur_log_offset']
        )
