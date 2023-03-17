# coding=utf8

import torch

from utils.hparams import hparams


class BaseSVSInfer:
    """
        Base class for SVS inference models.
        Subclasses should define:
        1. *build_model*:
            how to build the model;
        2. *run_model*:
            how to run the model (typically, generate a mel-spectrogram and
            pass it to the pre-built vocoder);
        3. *preprocess_input*:
            how to preprocess user input.
        4. *infer_once*
            infer from raw inputs to the final outputs
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        self.spk_map = {}

    def build_model(self, ckpt_steps=None):
        raise NotImplementedError

    def preprocess_input(self, inp):
        raise NotImplementedError

    def run_model(self, param, return_mel):
        raise NotImplementedError

    def infer_once(self, param):
        raise NotImplementedError()
