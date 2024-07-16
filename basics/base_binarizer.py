import json
import pathlib
import pickle
import random
import shutil
import warnings
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.phoneme_utils import load_phoneme_dictionary
from utils.plot import distribution_to_figure


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

    def __init__(self, data_dir=None, data_attrs=None):
        if data_dir is None:
            data_dir = hparams['raw_data_dir']
        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        self.raw_data_dirs = [pathlib.Path(d) for d in data_dir]
        self.binary_data_dir = pathlib.Path(hparams['binary_data_dir'])
        self.data_attrs = [] if data_attrs is None else data_attrs

        self.binarization_args = hparams['binarization_args']
        self.augmentation_args = hparams.get('augmentation_args', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.spk_map = {}
        self.spk_ids = hparams['spk_ids']
        self.speakers = hparams['speakers']
        self.build_spk_map()

        self.lang_map = {}
        self.dictionaries = hparams['dictionaries']
        self.languages = hparams['languages']
        self.build_lang_map()

        self.items = {}
        self.item_names: list = None
        self._train_item_names: list = None
        self._valid_item_names: list = None

        self.phoneme_dictionary = load_phoneme_dictionary()
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']

    def build_spk_map(self):
        assert isinstance(self.speakers, list), 'Speakers must be a list'
        assert len(self.speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'
        if len(self.spk_ids) == 0:
            self.spk_ids = list(range(len(self.raw_data_dirs)))
        else:
            assert len(self.spk_ids) == len(self.raw_data_dirs), \
                'Length of explicitly given spk_ids must equal the number of raw datasets.'
        assert max(self.spk_ids) < hparams['num_spk'], \
            f'Index in spk_id sequence {self.spk_ids} is out of range. All values should be smaller than num_spk.'

        for spk_name, spk_id in zip(self.speakers, self.spk_ids):
            if spk_name in self.spk_map and self.spk_map[spk_name] != spk_id:
                raise ValueError(f'Invalid speaker ID assignment. Name \'{spk_name}\' is assigned '
                                 f'with different speaker IDs: {self.spk_map[spk_name]} and {spk_id}.')
            self.spk_map[spk_name] = spk_id

        print("| spk_map: ", self.spk_map)

    def build_lang_map(self):
        assert isinstance(self.languages, list), 'Languages must be a list'
        assert len(self.languages) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of language names!'
        for lang in self.languages:
            assert lang in self.dictionaries, f'Unrecognized language name: {lang}'
        assert len(self.dictionaries.keys()) <= hparams['num_lang'], \
            'Number of languages must not be greater than num_lang!'

        for lang_id, lang_name in enumerate(sorted(self.dictionaries.keys()), start=1):
            self.lang_map[lang_name] = lang_id

        print("| lang_map: ", self.lang_map)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id, spk, lang):
        raise NotImplementedError()

    def split_train_valid_set(self, item_names):
        """
        Split the dataset into training set and validation set.
        :return: train_item_names, valid_item_names
        """
        prefixes = {str(pr): 1 for pr in hparams['test_prefixes']}
        valid_item_names = {}
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in deepcopy(prefixes):
            if prefix in item_names:
                valid_item_names[prefix] = 1
                prefixes.pop(prefix)
        # Add prefixes that exactly matches item name without speaker id to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in item_names:
                if name.split(':')[-1] == prefix:
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)
        # Add names with one of the remaining prefixes to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in item_names:
                if name.startswith(prefix):
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in item_names:
                if name.split(':')[-1].startswith(prefix):
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)

        if len(prefixes) != 0:
            warnings.warn(
                f'The following rules in test_prefixes have no matching names in the dataset: {", ".join(prefixes.keys())}',
                category=UserWarning
            )
            warnings.filterwarnings('default')

        valid_item_names = list(valid_item_names.keys())
        assert len(valid_item_names) > 0, 'Validation set is empty!'
        train_item_names = [x for x in item_names if x not in set(valid_item_names)]
        assert len(train_item_names) > 0, 'Training set is empty!'

        return train_item_names, valid_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    def meta_data_iterator(self, prefix):
        if prefix == 'train':
            item_names = self.train_item_names
        else:
            item_names = self.valid_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        # load each dataset
        for ds_id, (data_dir, spk, lang) in enumerate(zip(self.raw_data_dirs, self.speakers, self.languages)):
            self.load_meta_data(pathlib.Path(data_dir), ds_id=ds_id, spk=spk, lang=lang)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._valid_item_names = self.split_train_valid_set(self.item_names)

        if self.binarization_args['shuffle']:
            random.shuffle(self.item_names)

        self.binary_data_dir.mkdir(parents=True, exist_ok=True)

        # Copy spk_map, lang_map and dictionary to binary data dir
        spk_map_fn = self.binary_data_dir / 'spk_map.json'
        with open(spk_map_fn, 'w', encoding='utf-8') as f:
            json.dump(self.spk_map, f, ensure_ascii=False)
        lang_map_fn = self.binary_data_dir / 'lang_map.json'
        with open(lang_map_fn, 'w', encoding='utf-8') as f:
            json.dump(self.lang_map, f, ensure_ascii=False)
        for lang, dict_path in hparams['dictionaries'].items():
            shutil.copy(dict_path, self.binary_data_dir / f'dictionary-{lang}.txt')
        self.check_coverage()

        # Process valid set and train set
        try:
            self.process_dataset('valid')
            self.process_dataset(
                'train',
                num_workers=int(self.binarization_args['num_workers']),
                apply_augmentation=any(args['enabled'] for args in self.augmentation_args.values())
            )
        except KeyboardInterrupt:
            exit(-1)

    def check_coverage(self):
        # Group by phonemes in the dictionary.
        ph_idx_required = set(range(1, len(self.phoneme_dictionary)))
        ph_idx_occurred = set()
        ph_idx_count_map = {
            idx: 0
            for idx in ph_idx_required
        }

        # Load and count those phones that appear in the actual data
        for item_name in self.items:
            ph_idx_occurred.update(self.items[item_name]['ph_seq'])
            for idx in self.items[item_name]['ph_seq']:
                ph_idx_count_map[idx] += 1
        ph_count_map = {
            self.phoneme_dictionary.decode_one(idx, scalar=False): count
            for idx, count in ph_idx_count_map.items()
        }

        print('===== Phoneme Distribution Summary =====')
        keys = sorted(ph_count_map.keys(), key=lambda v: v[0] if isinstance(v, tuple) else v)
        for i, key in enumerate(keys):
            if i == len(ph_count_map) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            if isinstance(key, tuple):
                key_disp = '(' + ', '.join(key) + ')'
            else:
                key_disp = key
            print(f'{key_disp}: {ph_count_map[key]}', end=end)

        # Draw graph.
        xs = [str(k) for k in keys]
        ys = [ph_count_map[k] for k in keys]
        plt = distribution_to_figure(
            title='Phoneme Distribution Summary',
            x_label='Phoneme', y_label='Number of occurrences',
            items=xs, values=ys
        )
        filename = self.binary_data_dir / 'phoneme_distribution.jpg'
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')

        # Check unrecognizable or missing phonemes
        if ph_idx_occurred != ph_idx_required:
            missing_phones = sorted({
                self.phoneme_dictionary.decode_one(idx, scalar=False)
                for idx in ph_idx_required.difference(ph_idx_occurred)
            }, key=lambda v: v[0] if isinstance(v, tuple) else v)
            raise BinarizationError(
                f'The following phonemes are not covered in transcriptions: {sorted(missing_phones)}'
            )

    def process_dataset(self, prefix, num_workers=0, apply_augmentation=False):
        args = []
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=self.data_attrs)
        total_sec = {k: 0.0 for k in self.spk_map}
        total_raw_sec = {k: 0.0 for k in self.spk_map}
        extra_info = {'names': {}, 'spk_ids': {}, 'spk_names': {}, 'lengths': {}}
        max_no = -1

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        aug_map = self.arrange_data_augmentation(self.meta_data_iterator(prefix)) if apply_augmentation else {}

        def postprocess(_item):
            nonlocal total_sec, total_raw_sec, extra_info, max_no
            if _item is None:
                return
            item_no = builder.add_item(_item)
            max_no = max(max_no, item_no)
            for k, v in _item.items():
                if isinstance(v, np.ndarray):
                    if k not in extra_info:
                        extra_info[k] = {}
                    extra_info[k][item_no] = v.shape[0]
            extra_info['names'][item_no] = _item['name'].split(':', 1)[-1]
            extra_info['spk_ids'][item_no] = _item['spk_id']
            extra_info['spk_names'][item_no] = _item['spk_name']
            extra_info['lengths'][item_no] = _item['length']
            total_raw_sec[_item['spk_name']] += _item['seconds']
            total_sec[_item['spk_name']] += _item['seconds']

            for task in aug_map.get(_item['name'], []):
                aug_item = task['func'](_item, **task['kwargs'])
                aug_item_no = builder.add_item(aug_item)
                max_no = max(max_no, aug_item_no)
                for k, v in aug_item.items():
                    if isinstance(v, np.ndarray):
                        if k not in extra_info:
                            extra_info[k] = {}
                        extra_info[k][aug_item_no] = v.shape[0]
                extra_info['names'][aug_item_no] = aug_item['name'].split(':', 1)[-1]
                extra_info['spk_ids'][aug_item_no] = aug_item['spk_id']
                extra_info['spk_names'][aug_item_no] = aug_item['spk_name']
                extra_info['lengths'][aug_item_no] = aug_item['length']
                total_sec[aug_item['spk_name']] += aug_item['seconds']

        try:
            if num_workers > 0:
                # code for parallel processing
                for item in tqdm(
                        chunked_multiprocess_run(self.process_item, args, num_workers=num_workers),
                        total=len(list(self.meta_data_iterator(prefix)))
                ):
                    postprocess(item)
            else:
                # code for single cpu processing
                for a in tqdm(args):
                    item = self.process_item(*a)
                    postprocess(item)
            for k in extra_info:
                assert set(extra_info[k]) == set(range(max_no + 1)), f'Item numbering is not consecutive.'
                extra_info[k] = list(map(lambda x: x[1], sorted(extra_info[k].items(), key=lambda x: x[0])))
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()
        if prefix == "train":
            extra_info.pop("names")
            extra_info.pop("spk_names")
        with open(self.binary_data_dir / f"{prefix}.meta", "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(extra_info, f)
        if apply_augmentation:
            print(f"| {prefix} total duration (before augmentation): {sum(total_raw_sec.values()):.2f}s")
            print(
                f"| {prefix} respective duration (before augmentation): "
                + ', '.join(f'{k}={v:.2f}s' for k, v in total_raw_sec.items())
            )
            print(
                f"| {prefix} total duration (after augmentation): "
                f"{sum(total_sec.values()):.2f}s ({sum(total_sec.values()) / sum(total_raw_sec.values()):.2f}x)"
            )
            print(
                f"| {prefix} respective duration (after augmentation): "
                + ', '.join(f'{k}={v:.2f}s' for k, v in total_sec.items())
            )
        else:
            print(f"| {prefix} total duration: {sum(total_raw_sec.values()):.2f}s")
            print(f"| {prefix} respective duration: " + ', '.join(f'{k}={v:.2f}s' for k, v in total_raw_sec.items()))

    def arrange_data_augmentation(self, data_iterator):
        """
        Code for all types of data augmentation should be added here.
        """
        raise NotImplementedError()

    def process_item(self, item_name, meta_data, binarization_args):
        raise NotImplementedError()
