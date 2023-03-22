import os.path
import pathlib
import pickle
import multiprocessing
from copy import deepcopy
import h5py
import torch

import numpy as np


class IndexedDataset:
    def __init__(self, path, prefix, num_cache=0):
        super().__init__()
        self.path = pathlib.Path(path)
        # self.data_file = None
        # self.data_offsets = np.load(self.path / f'{prefix}.idx'))
        # self.data_file = open(self.path / f'{prefix}.data', 'rb', buffering=-1)
        self.dset = h5py.File(self.path / f'{prefix}.hdf5', 'r')
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        # if i < 0 or i >= len(self.data_offsets) - 1:
        #     raise IndexError('index out of range')
        if i < 0 or i >= len(self.dset):
            raise IndexError('index out of range')

    def __del__(self):
        # if self.data_file:
        #     self.data_file.close()
        if self.dset:
            del self.dset

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        # self.data_file.seek(self.data_offsets[i])
        # b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        # item = pickle.loads(b)
        item = {k: v[()] if v.shape == () else torch.from_numpy(v[()]) for k, v in self.dset[str(i)].items()}
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        # return len(self.data_offsets) - 1
        return len(self.dset)

class IndexedDatasetBuilder:
    def __init__(self, path, prefix, allowed_attr=None):
        self.path = pathlib.Path(path)
        self.prefix = prefix
        # self.out_file = open(os.path.join(path, f'{prefix}.data'), 'wb')
        self.dset = h5py.File(self.path / f'{prefix}.hdf5', 'w')
        self.counter = 0
        self.lock = multiprocessing.Lock()
        # self.byte_offsets = [0]
        if allowed_attr is not None:
            self.allowed_attr = set(allowed_attr)
        else:
            self.allowed_attr = None

    def add_item(self, item):
        if self.allowed_attr is not None:
            item = {
                k: item.get(k)
                for k in self.allowed_attr
            }
        with self.lock:
            item_no = self.counter
            self.counter += 1
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                self.dset.create_dataset(f'{item_no}/{k}', data=v, compression="gzip", compression_opts=4)
            else:
                self.dset.create_dataset(f'{item_no}/{k}', data=v)
        # s = pickle.dumps(item)
        # n_bytes = self.out_file.write(s)
        # self.byte_offsets.append(self.byte_offsets[-1] + n_bytes)

    def finalize(self):
        del self.dset
        # self.out_file.close()
        # with open(os.path.join(self.path, f'{self.prefix}.idx'), 'wb') as f:
        #     # noinspection PyTypeChecker
        #     np.save(f, self.byte_offsets)


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    ds_path = './checkpoints/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path, 'example')
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path, 'example')
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
