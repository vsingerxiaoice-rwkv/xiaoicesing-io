import json
import logging
import os
import random
from copy import deepcopy

from utils.hparams import set_hparams, hparams
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class BinarizationError(Exception):
    pass

class BaseBinarizer:
    """
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    """
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = hparams['raw_data_dir']

        speakers = hparams['speakers']
        assert isinstance(speakers, list), 'Speakers must be a list'
        assert len(speakers) == len(set(speakers)), 'Speakers cannot contain duplicate names'

        self.raw_data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        assert len(speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'

        self.binarization_args = hparams['binarization_args']
        self.augmentation_args = hparams.get('augmentation_args', {})

        self.spk_map = None
        self.items = {}
        self.phone_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())

        # load each dataset
        for ds_id, data_dir in enumerate(self.raw_data_dirs):
            self.load_meta_data(data_dir, ds_id)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._test_item_names = self.split_train_test_set()

        if self.binarization_args['shuffle']:
            random.seed(hparams['seed'])
            random.shuffle(self.item_names)

    def load_meta_data(self, raw_data_dir, ds_id):
        raise NotImplementedError()

    def split_train_test_set(self):
        item_names = set(deepcopy(self.item_names))
        prefixes = set([str(pr) for pr in hparams['test_prefixes']])
        test_item_names = set()
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in deepcopy(prefixes):
            if prefix in item_names:
                test_item_names.add(prefix)
                prefixes.remove(prefix)
        # Add prefixes that exactly matches item name without speaker id to test set
        for prefix in deepcopy(prefixes):
            for name in item_names:
                if name.split(':')[-1] == prefix:
                    test_item_names.add(name)
                    prefixes.remove(prefix)
        # Add names with one of the remaining prefixes to test set
        for prefix in deepcopy(prefixes):
            for name in item_names:
                if name.startswith(prefix):
                    test_item_names.add(name)
                    prefixes.remove(prefix)
        for prefix in prefixes:
            for name in item_names:
                if name.split(':')[-1].startswith(prefix):
                    test_item_names.add(name)
        test_item_names = sorted(list(test_item_names))
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def build_spk_map(self):
        spk_map = {x: i for i, x in enumerate(hparams['speakers'])}
        assert len(spk_map) <= hparams['num_spk'], 'Actual number of speakers should be smaller than num_spk!'
        self.spk_map = spk_map

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))
        self.check_coverage()

    def check_coverage(self):
        raise NotImplementedError()

    def process_data_split(self, prefix, multiprocess=False, apply_augmentation=False):
        raise NotImplementedError()

    def arrange_data_augmentation(self, prefix):
        """
        Code for all types of data augmentation should be added here.
        """
        raise NotImplementedError()

    def process_item(self, item_name, meta_data, binarization_args):
        raise NotImplementedError()


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
