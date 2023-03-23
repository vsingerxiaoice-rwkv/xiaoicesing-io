import math

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import Sampler, DistributedSampler

import utils
from utils.hparams import hparams

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

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
        `eta_min` (default=0.0) corresponds to the minimum learning rate reached by the scheduler.
    """
    def __init__(self, optimizer, warmup_steps, t_total, eta_min=0.0, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.eta_min = eta_min
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1.0, self.warmup_steps)
        # progress after warmup
        progress = (step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        return max(self.eta_min, 0.5 * (1. + math.cos(math.pi * self.cycles * 2.0 * progress)))

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, max_tokens, max_sentences, indices=None, batch_by_size=True, seed=0, shuffle=True):
        self.dataset = dataset
        self.sub_indices = indices
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.batch_by_size = batch_by_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = None

    def __iter__(self):
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            if self.sub_indices is not None:
                rng.shuffle(self.sub_indices)
                indices = np.array(self.sub_indices)
            else:
                indices = rng.permutation(len(self.dataset))
            if self.dataset.sort_by_len:
                grid = hparams.get('sampler_frame_count_grid', 100)
                sizes = (np.round(np.array(self.dataset._sizes)[indices] / grid) * grid).clip(grid, None).astype(np.int64)
                indices = indices[np.argsort(sizes, kind='mergesort')]
            indices = indices.tolist()
        else:
            indices = self.sub_indices if self.sub_indices is not None else list(range(len(self.dataset)))
        
        if self.batch_by_size:
            self.batches = utils.batch_by_size(indices, self.dataset.num_tokens, max_tokens=self.max_tokens, max_sentences=self.max_sentences)
        else:
            self.batches = [indices[i:i + self.max_sentences] for i in range(0, len(indices), self.max_sentences)]
            
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        if self.batches is None:
            raise RuntimeError("Batches are not initialized. Call __iter__ first.")
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(self, dataset, num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_sampler_cls=None) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_sampler_cls = batch_sampler_cls
        self.batch_sampler = None

    def __iter__(self):
        indices = list(super().__iter__())
        self.batch_sampler = self.batch_sampler_cls(self.dataset, indices=indices, seed=self.seed, shuffle=self.shuffle)
        self.batch_sampler.set_epoch(self.epoch)
        return iter(self.batch_sampler)

    def __len__(self) -> int:
        if self.batch_sampler is None:
            raise RuntimeError("BatchSampler is not initialized. Call __iter__ first.")
        return len(self.batch_sampler)
