import os

import numpy as np
from torch.utils.data import Dataset

from utils.hparams import hparams


class BaseDataset(Dataset):
    '''
        Base class for datasets.
        1. *sizes*:
            clipped length if "max_frames" is set;
        2. *num_tokens*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    '''
    def __init__(self):
        super().__init__()
        self.hparams = hparams
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self._sizes[index]
