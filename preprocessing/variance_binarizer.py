import csv
import os
import pathlib

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate

from basics.base_binarizer import BaseBinarizer
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import (
    get_mel2ph_torch,
    get_pitch_parselmouth,
    get_energy_librosa,
    get_breathiness_pyworld
)
from utils.hparams import hparams

os.environ["OMP_NUM_THREADS"] = "1"
VARIANCE_ITEM_ATTRIBUTES = [
    'spk_id',  # index number of dataset/speaker, int64
    'tokens',  # index numbers of phonemes, int64[T_ph,]
    'ph_dur',  # durations of phonemes, in number of frames, int64[T_ph,]
    'midi',  # phoneme-level mean MIDI pitch, int64[T_ph,]
    'ph2word',  # similar to mel2ph format, representing number of phones within each note, int64[T_ph,]
    'mel2ph',  # mel2ph format representing number of frames within each phone, int64[T_t,]
    'base_pitch',  # interpolated and smoothed frame-level MIDI pitch, float32[T_t,]
    'delta_pitch',  # delta_pitch = actual_pitch - base_pitch, in semitones, float32[T_t,]
    'energy',  # float32[T_t,]
    'breathiness',  # float32[T_t,]
]


class VarianceBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=VARIANCE_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator().to(self.device)
        smooth_kernel_size = round(hparams['midi_smooth_width'] / self.timestep)
        self.smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval().to(self.device)
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ).to(self.device))
        smooth_kernel /= smooth_kernel.sum()
        self.smooth.weight.data = smooth_kernel[None, None]

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        for utterance_label in csv.DictReader(
                open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8')
        ):
            item_name = utterance_label['name']
            temp_dict = {
                'spk_id': ds_id,
                'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                'ph_seq': utterance_label['ph_seq'].split(),
                'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()],
                'ph_num': [int(x) for x in utterance_label['ph_num'].split()],
                'note_seq': utterance_label['note_seq'].split(),
                'note_dur': [float(x) for x in utterance_label['note_dur'].split()],
            }
            assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
            assert len(temp_dict['ph_seq']) == sum(temp_dict['ph_num']), \
                f'Sum of ph_num does not equal length of ph_seq in \'{item_name}\'.'
            assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
            assert any([note != 'rest' for note in temp_dict['note_seq']]), \
                f'All notes are rest in \'{item_name}\'.'
            meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        self.items.update(meta_data_dict)

    def check_coverage(self):
        print('Coverage checks are temporarily skipped.')
        pass

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        seconds = sum(meta_data['ph_dur'])
        length = round(seconds / self.timestep)
        t_txt = len(meta_data['ph_seq'])
        ph_dur_sec = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=0) / self.timestep + 0.5).long()
        ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
        ph_num = torch.LongTensor(meta_data['ph_num']).to(self.device)
        ph2word = self.lr(ph_num[None])[0]
        mel2ph = get_mel2ph_torch(
            self.lr, ph_dur_sec, length, self.timestep, device=self.device
        )

        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'seconds': seconds,
            'length': length,
            'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
            'ph_dur': ph_dur.cpu().numpy(),
            'ph2word': ph2word.cpu().numpy(),
            'mel2ph': mel2ph.cpu().numpy(),
        }

        # Below: calculate and interpolate frame-level MIDI pitch, which is a step function curve
        mel2dur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 0, mel2ph)  # frame-level phone duration
        note_dur = torch.FloatTensor(meta_data['note_dur']).to(self.device)
        mel2note = get_mel2ph_torch(
            self.lr, note_dur, mel2ph.shape[0], self.timestep, device=self.device
        )
        note_pitch = torch.FloatTensor(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in meta_data['note_seq']]
        ).to(self.device)
        frame_midi_pitch = torch.gather(F.pad(note_pitch, [1, 0], value=0), 0, mel2note)  # => frame-level MIDI pitch
        rest = (frame_midi_pitch < 0).cpu().numpy()
        frame_midi_pitch = frame_midi_pitch.cpu().numpy()
        interp_func = interpolate.interp1d(
            np.where(~rest)[0], frame_midi_pitch[~rest],
            kind='nearest', fill_value='extrapolate'
        )
        frame_midi_pitch[rest] = interp_func(np.where(rest)[0])
        frame_midi_pitch = torch.from_numpy(frame_midi_pitch).to(self.device)

        # Below: calculate phoneme-level mean MIDI pitch
        ph_midi = frame_midi_pitch.new_zeros(t_txt + 1).scatter_add(
            0, mel2ph, frame_midi_pitch / mel2dur
        )[1:]

        processed_input['midi'] = ph_midi.round().long().cpu().numpy()

        # Below: smoothen the pitch step curve as the base pitch curve
        smoothed_midi_pitch = self.smooth(frame_midi_pitch[None])[0]

        processed_input['base_pitch'] = smoothed_midi_pitch.cpu().numpy()

        # Below: extract actual f0, convert to pitch and calculate delta pitch
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        f0, uv = get_pitch_parselmouth(waveform, length, hparams, interp_uv=True)
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None

        processed_input['delta_pitch'] = librosa.hz_to_midi(f0.astype(np.float32)) - processed_input['base_pitch']

        # Below: extract energy
        if hparams['predict_energy']:
            energy = get_energy_librosa(waveform, length, hparams)
            processed_input['energy'] = energy.astype(np.float32)

        # Below: extract breathiness
        if hparams['predict_breathiness']:
            breathiness = get_breathiness_pyworld(waveform, f0 * ~uv, length, hparams)
            processed_input['breathiness'] = breathiness.astype(np.float32)

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
