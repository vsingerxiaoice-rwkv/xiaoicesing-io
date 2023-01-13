"""

    item: one piece of data
    item_name: data id
    wavfn: wave file path
    txt: lyrics
    ph: phoneme
    tgfn: text grid file path (unused)
    spk: dataset name
    wdb: word boundary
    ph_durs: phoneme durations
    midi: pitch as midi notes
    midi_dur: midi duration
    is_slur: keep singing upon note changes
"""

import logging
from copy import deepcopy

from data_gen.midisinging import MidiSingingBinarizer
from utils.hparams import hparams


class OpencpopBinarizer(MidiSingingBinarizer):
    def split_train_test_set(self, item_names):
        item_names = set(deepcopy(item_names))
        prefixes = set(deepcopy(hparams['test_prefixes']))
        test_item_names = set()
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in hparams['test_prefixes']:
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

    def load_meta_data(self, raw_data_dir, ds_id):
        from preprocessing.opencpop import File2Batch
        self.items.update(File2Batch.file2temporary_dict(raw_data_dir, ds_id))
