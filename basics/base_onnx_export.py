import json
import os
import pathlib

import torch
import torch.nn as nn

from utils.hparams import hparams


class BaseOnnxExport:
    def __init__(self, device=None, cache_dir=None, **kwargs):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir: pathlib.Path = cache_dir if cache_dir is not None \
            else pathlib.Path(__file__).parent.parent / 'deployment' / 'cache'
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def build_spk_map(self) -> dict:
        if hparams['use_spk_id']:
            with open(os.path.join(hparams['work_dir'], 'spk_map.json'), 'r', encoding='utf8') as f:
                spk_map = json.load(f)
            assert isinstance(spk_map, dict) and len(spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(spk_map) == len(set(spk_map.values())), 'Duplicate speaker id in speaker map!'
            return spk_map
        else:
            return {}

    def build_model(self) -> nn.Module:
        raise NotImplementedError()

    def export_model(self, path: pathlib.Path):
        raise NotImplementedError()
