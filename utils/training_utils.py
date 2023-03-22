import utils
from utils.hparams import hparams

import math
import numpy as np
from torch.utils.data.distributed import Sampler, DistributedSampler

class RSQRTSchedule(object):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.warmup_updates = hparams['warmup_updates']
        self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, indices=None, max_tokens=None, max_sentences=None, required_batch_size_multiple=-1, batch_by_size=True, shuffle=True):
        self.shuffle = shuffle
            
        if batch_by_size:
            self.batches = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple
            )
        else:
            self.batches = [indices[i:i + max_sentences] for i in range(0, len(indices), max_sentences)]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(self, dataset, num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_sampler_cls=None) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_sampler_cls = batch_sampler_cls
        self.batch_sampler = None

    def __iter__(self):
        if self.shuffle:
            indices = np.random.RandomState(seed=self.seed).permutation(len(self.dataset))
            if self.dataset.sort_by_len:
                indices = indices[np.argsort(np.array(self.dataset._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self.dataset))
        indices = indices.tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        self.batch_sampler = self.batch_sampler_cls(self.dataset, indices=indices, shuffle=self.shuffle)
        return iter(self.batch_sampler)

    def __len__(self) -> int:
        if self.batch_sampler is None:
            raise ValueError("BatchSampler is not initialized. Call __iter__ first.")
        return len(self.batch_sampler)
