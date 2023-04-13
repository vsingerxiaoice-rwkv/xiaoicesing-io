"""
    item: one piece of data
    item_name: data id
    wav_fn: wave file path
    spk: dataset name
    ph_seq: phoneme sequence
    ph_dur: phoneme durations
    midi_seq: midi note sequence
    midi_dur: midi note durations
"""
import csv
import os
import pathlib
import shutil

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from basics.base_binarizer import BaseBinarizer
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import get_mel2ph_torch
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.phoneme_utils import locate_dictionary

os.environ["OMP_NUM_THREADS"] = "1"
VARIANCE_ITEM_ATTRIBUTES = [
    'spk_id',  # index number of dataset/speaker
    'tokens',  # index numbers of phonemes
    'ph_dur',  # durations of phonemes, in seconds
    'ph_midi',  # phoneme-level mean MIDI pitch
    'word_dur',  # durations of words/syllables (vowel-consonant pattern)
    'mel2ph',  # mel2ph format representing gt ph_dur
    # 'base_pitch',
    # 'f0'
]


class VarianceBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__()
        self.lr = LengthRegulator()

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        for utterance_label in csv.DictReader(
                open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8')
        ):
            item_name = utterance_label['name']
            temp_dict = {
                'spk_id': ds_id,
                'wav_fn': str(raw_data_dir / 'wav' / f'{item_name}.wav'),
                'ph_seq': utterance_label['ph_seq'].split(),
                'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()],
                'word_dur': [float(x) for x in utterance_label['word_dur'].split()],
                'note_seq': utterance_label['note_seq'].split(),
                'note_dur': [float(x) for x in utterance_label['note_dur'].split()],
            }
            assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']) == len(temp_dict['word_dur']), \
                f'Lengths of ph_seq, ph_dur and word_dur mismatch in \'{item_name}\'.'
            assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
            meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        self.items.update(meta_data_dict)

    def process(self):
        super().process()
        self.process_data_split('valid')
        self.process_data_split('train', num_workers=self.binarization_args['num_workers'])

    def check_coverage(self):
        shutil.copy(locate_dictionary(), self.binary_data_dir / 'dictionary.txt')
        print('Coverage checks are temporarily skipped.')
        pass

    def process_data_split(self, prefix, num_workers=0, apply_augmentation=False):
        args = []
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=VARIANCE_ITEM_ATTRIBUTES)
        lengths = []
        total_raw_sec = 0

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        def postprocess(_item):
            nonlocal total_raw_sec
            if _item is None:
                return
            builder.add_item(_item)
            lengths.append(_item['length'])
            total_raw_sec += _item['seconds']

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

        builder.finalize()
        with open(self.binary_data_dir / f'{prefix}.lengths', 'wb') as f:
            # noinspection PyTypeChecker
            np.save(f, lengths)

        print(f'| {prefix} total duration: {total_raw_sec:.2f}s')

    def process_item(self, item_name, meta_data, binarization_args):
        length = len(meta_data['ph_dur'])  # temporarily use number of tokens as sample length
        seconds = sum(meta_data['ph_dur'])
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'seconds': seconds,
            'length': length,
            'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
            'word_dur': np.array(meta_data['word_dur']).astype(np.float32),
        }

        # Below: calculate phoneme-level mean pitch for MIDI input
        ph_dur = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
        mel2ph = get_mel2ph_torch(
            self.lr, ph_dur, round(seconds / self.timestep), self.timestep, device=self.device
        )
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        ph_dur_long = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))

        mel2dur = torch.gather(F.pad(ph_dur_long, [1, 0], value=1), 0, mel2ph)  # frame-level phone duration
        note_dur = torch.FloatTensor(meta_data['note_dur']).to(self.device)
        mel2note = get_mel2ph_torch(
            self.lr, note_dur, mel2ph.shape[0], self.timestep, device=self.device
        )
        note_pitch = torch.FloatTensor(
            [(librosa.note_to_midi(n) if n != 'rest' else 0) for n in meta_data['note_seq']]
        ).to(self.device)
        frame_step_pitch = torch.gather(F.pad(note_pitch, [1, 0], value=0), 0, mel2note)  # => frame-level MIDI pitch
        # Below: handle rest parts where pitch == 0 in frame_step_pitch
        ph_dur_rest = mel2ph.new_zeros(len(ph_dur) + 1).scatter_add(
            0, mel2ph, (frame_step_pitch == 0).long()
        )[1:]
        mel2dur_rest = torch.gather(F.pad(ph_dur_rest, [1, 0], value=1), 0, mel2ph)  # frame-level rest phone duration

        ph_midi = mel2ph.new_zeros(ph_dur.shape[0] + 1).float().scatter_add(
            0, mel2ph, frame_step_pitch / ((mel2dur - mel2dur_rest) + (mel2dur == mel2dur_rest))  # avoid div by zero
        )[1:]

        processed_input['ph_dur'] = ph_dur_long.cpu().numpy()  # number of frames of each phone
        processed_input['ph_midi'] = ph_midi.long().cpu().numpy()
        processed_input['mel2ph'] = mel2ph.cpu().numpy()

        return processed_input
