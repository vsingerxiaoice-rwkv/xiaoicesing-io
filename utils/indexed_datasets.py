import os.path
import pickle
from copy import deepcopy

import numpy as np


class IndexedDataset:
    def __init__(self, path, num_cache=0):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx")
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets):
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets)

class IndexedDatasetBuilder:
    def __init__(self, path, name, allowed_attr=None):
        self.path = path
        self.name = name
        self.out_file = open(os.path.join(path, f'{name}.data'), 'wb')
        self.byte_offsets = [0]
        if allowed_attr is not None:
            self.allowed_attr = set(allowed_attr)

    def add_item(self, item):
        if self.allowed_attr is not None:
            item = {
                k: item.get(k)
                for k in self.allowed_attr
            }
        s = pickle.dumps(item)
        n_bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + n_bytes)

    def finalize(self):
        self.out_file.close()
        with open(os.path.join(self.path, f'{self.name}.idx'), 'wb') as f:
            # noinspection PyTypeChecker
            np.save(f, self.byte_offsets[:-1])


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path, 'example')
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path)
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
