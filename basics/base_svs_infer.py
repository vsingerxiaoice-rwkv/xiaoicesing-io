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
        self.model: torch.nn.Module = None

    def build_model(self, ckpt_steps=None) -> torch.nn.Module:
        raise NotImplementedError

    def preprocess_input(self, param: dict, idx=0) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_model(self, sample: dict[str, torch.Tensor]):
        raise NotImplementedError

    def run_inference(self, params, **kwargs):
        raise NotImplementedError()
