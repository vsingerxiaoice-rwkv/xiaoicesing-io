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

import os
import os.path
import random
import traceback
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from librosa import note_to_midi
from tqdm import tqdm

from basics.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.data_gen_utils import get_pitch_parselmouth
from preprocessing.opencpop import vowels
from src.vocoders.vocoder_utils import VOCODERS
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.phoneme_utils import build_phoneme_list

os.environ["OMP_NUM_THREADS"] = "1"


class AcousticBinarizer(BaseBinarizer):
    def load_meta_data(self, raw_data_dir, ds_id):
        utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt'), encoding='utf-8').readlines()
        all_temp_dict = {}
        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = song_info[0]
            temp_dict = {
                'wav_fn': f'{raw_data_dir}/wavs/{item_name}.wav',
                'txt': song_info[1],
                'ph': song_info[2],
                'word_boundary': np.array([1 if x in vowels + ['AP', 'SP'] else 0 for x in song_info[2].split()]),
                'ph_durs': [float(x) for x in song_info[5].split()],
                'pitch_midi': np.array([note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                        for x in song_info[3].split()]),
                'midi_dur': np.array([float(x) for x in song_info[4].split()]),
                'is_slur': np.array([int(x) for x in song_info[6].split()]),
                'spk_id': ds_id
            }

            assert temp_dict['pitch_midi'].shape == temp_dict['midi_dur'].shape == temp_dict['is_slur'].shape, \
                (temp_dict['pitch_midi'].shape, temp_dict['midi_dur'].shape, temp_dict['is_slur'].shape)

            all_temp_dict[f'{ds_id}:{item_name}'] = temp_dict
        self.items.update(all_temp_dict)

    def process(self):
        super().process()
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train', apply_augmentation=len(self.augmentation_args) > 0)

    def check_coverage(self):
        # Group by phonemes in the dictionary.
        ph_required = set(build_phoneme_list())
        phoneme_map = {}
        for ph in ph_required:
            phoneme_map[ph] = 0
        ph_occurred = []
        # Load and count those phones that appear in the actual data
        for item in self.items.values():
            ph_occurred += item['ph'].split(' ')
        for ph in ph_occurred:
            if ph not in ph_required:
                continue
            phoneme_map[ph] += 1
        ph_occurred = set(ph_occurred)

        print('===== Phoneme Distribution Summary =====')
        for i, key in enumerate(sorted(phoneme_map.keys())):
            if i == len(ph_required) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{key}\': {phoneme_map[key]}', end=end)

        # Draw graph.
        plt.figure(figsize=(int(len(ph_required) * 0.8), 10))
        x = list(phoneme_map.keys())
        values = list(phoneme_map.values())
        plt.bar(x=x, height=values)
        plt.tick_params(labelsize=15)
        plt.xlim(-1, len(ph_required))
        for a, b in zip(x, values):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
        plt.grid()
        plt.title('Phoneme Distribution Summary', fontsize=30)
        plt.xlabel('Phoneme', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        filename = os.path.join(hparams['binary_data_dir'], 'phoneme_distribution.jpg')
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')
        # Check unrecognizable or missing phonemes
        if ph_occurred != ph_required:
            unrecognizable_phones = ph_occurred.difference(ph_required)
            missing_phones = ph_required.difference(ph_occurred)
            raise AssertionError('transcriptions and dictionary mismatch.\n'
                                 f' (+) {sorted(unrecognizable_phones)}\n'
                                 f' (-) {sorted(missing_phones)}')

    def process_data_split(self, prefix, multiprocess=False, apply_augmentation=False):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0
        total_raw_sec = 0

        if self.binarization_args['with_spk_embed']:
            from resemblyzer import VoiceEncoder
            voice_encoder = VoiceEncoder().cuda()

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        aug_map = self.arrange_data_augmentation(prefix) if apply_augmentation else {}

        def postprocess(item_):
            nonlocal total_sec, total_raw_sec
            if item_ is None:
                return
            item_['spk_embed'] = voice_encoder.embed_utterance(item_['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item_:
                del item_['wav']
            builder.add_item(item_)
            lengths.append(item_['len'])
            total_sec += item_['sec']
            total_raw_sec += item_['sec']
            if item_.get('f0') is not None:
                f0s.append(item_['f0'])

            for task in aug_map.get(item_['item_name'], []):
                aug_item = task['func'](item_, **task['kwargs'])
                builder.add_item(aug_item)
                lengths.append(aug_item['len'])
                total_sec += aug_item['sec']
                if aug_item.get('f0') is not None:
                    f0s.append(aug_item['f0'])

        if multiprocess:
            # code for parallel processing
            num_workers = int(os.getenv('N_PROC', hparams.get('ds_workers', os.cpu_count() // 3)))
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
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])

        if apply_augmentation:
            print(f'| {prefix} total duration (before augmentation): {total_raw_sec:.2f}s')
            print(
                f'| {prefix} total duration (after augmentation): {total_sec:.2f}s ({total_sec / total_raw_sec:.2f}x)')
        else:
            print(f'| {prefix} total duration: {total_raw_sec:.2f}s')

    def arrange_data_augmentation(self, prefix):
        aug_map = {}
        aug_list = []
        all_item_names = [item_name for item_name, _ in self.meta_data_iterator(prefix)]
        total_scale = 0
        if self.augmentation_args.get('random_pitch_shifting') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_pitch_shifting']
            key_shift_min, key_shift_max = aug_args['range']
            assert hparams.get('use_key_shift_embed', False), \
                'Random pitch shifting augmentation requires use_key_shift_embed == True.'
            assert key_shift_min < 0 < key_shift_max, \
                'Random pitch shifting augmentation must have a range where min < 0 < max.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
            scale = aug_args['scale']
            aug_item_names = random.choices(all_item_names, k=int(scale * len(all_item_names)))

            for aug_item_name in aug_item_names:
                rand = random.uniform(-1, 1)
                if rand < 0:
                    key_shift = key_shift_min * abs(rand)
                else:
                    key_shift = key_shift_max * rand
                aug_task = {
                    'name': aug_item_name,
                    'func': aug_ins.process_item,
                    'kwargs': {'key_shift': key_shift}
                }
                if aug_item_name in aug_map:
                    aug_map[aug_item_name].append(aug_task)
                else:
                    aug_map[aug_item_name] = [aug_task]
                aug_list.append(aug_task)

            total_scale += scale

        if self.augmentation_args.get('fixed_pitch_shifting') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['fixed_pitch_shifting']
            targets = aug_args['targets']
            scale = aug_args['scale']
            assert self.augmentation_args.get('random_pitch_shifting') is None, \
                'Fixed pitch shifting augmentation is not compatible with random pitch shifting.'
            assert len(targets) == len(set(targets)), \
                'Fixed pitch shifting augmentation requires having no duplicate targets.'
            assert hparams['use_spk_id'], 'Fixed pitch shifting augmentation requires use_spk_id == True.'
            assert hparams['num_spk'] >= (1 + len(targets)) * len(self.spk_map), \
                'Fixed pitch shifting augmentation requires num_spk >= (1 + len(targets)) * len(speakers).'
            assert scale < 1, 'Fixed pitch shifting augmentation requires scale < 1.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
            for i, target in enumerate(targets):
                aug_item_names = random.choices(all_item_names, k=int(scale * len(all_item_names)))
                for aug_item_name in aug_item_names:
                    replace_spk_id = int(aug_item_name.split(':', maxsplit=1)[0]) + (i + 1) * len(self.spk_map)
                    aug_task = {
                        'name': aug_item_name,
                        'func': aug_ins.process_item,
                        'kwargs': {'key_shift': target, 'replace_spk_id': replace_spk_id}
                    }
                    if aug_item_name in aug_map:
                        aug_map[aug_item_name].append(aug_task)
                    else:
                        aug_map[aug_item_name] = [aug_task]
                    aug_list.append(aug_task)

            total_scale += scale * len(targets)

        if self.augmentation_args.get('random_time_stretching') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_time_stretching']
            speed_min, speed_max = aug_args['range']
            domain = aug_args['domain']
            assert hparams.get('use_speed_embed', False), \
                'Random time stretching augmentation requires use_speed_embed == True.'
            assert 0 < speed_min < 1 < speed_max, \
                'Random time stretching augmentation must have a range where 0 < min < 1 < max.'
            assert domain in ['log', 'linear'], 'domain must be \'log\' or \'linear\'.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
            scale = aug_args['scale']
            k_from_raw = int(scale / (1 + total_scale) * len(all_item_names))
            k_from_aug = int(total_scale * scale / (1 + total_scale) * len(all_item_names))
            k_mutate = int(total_scale * scale / (1 + scale) * len(all_item_names))
            aug_types = [0] * k_from_raw + [1] * k_from_aug + [2] * k_mutate
            aug_items = random.choices(all_item_names, k=k_from_raw) + random.choices(aug_list, k=k_from_aug + k_mutate)

            for aug_type, aug_item in zip(aug_types, aug_items):
                if domain == 'log':
                    # Uniform distribution in log domain
                    speed = speed_min * (speed_max / speed_min) ** random.random()
                else:
                    # Uniform distribution in linear domain
                    rand = random.uniform(-1, 1)
                    speed = 1 + (speed_max - 1) * rand if rand >= 0 else 1 + (1 - speed_min) * rand
                if aug_type == 0:
                    aug_task = {
                        'name': aug_item,
                        'func': aug_ins.process_item,
                        'kwargs': {'speed': speed}
                    }
                    if aug_item in aug_map:
                        aug_map[aug_item].append(aug_task)
                    else:
                        aug_map[aug_item] = [aug_task]
                    aug_list.append(aug_task)
                elif aug_type == 1:
                    aug_task = deepcopy(aug_item)
                    aug_item['kwargs']['speed'] = speed
                    if aug_item['name'] in aug_map:
                        aug_map[aug_item['name']].append(aug_task)
                    else:
                        aug_map[aug_item['name']] = [aug_task]
                    aug_list.append(aug_task)
                elif aug_type == 2:
                    aug_item['kwargs']['speed'] = speed

            total_scale += scale

        return aug_map

    def process_item(self, item_name, meta_data, binarization_args):
        mel_path = f"{meta_data['wav_fn'][:-4]}_mel.npy"
        if os.path.exists(mel_path):
            wav = None
            mel = np.load(mel_path)
            print("load mel from npy")
        else:
            if hparams['vocoder'] in VOCODERS:
                wav, mel = VOCODERS[hparams['vocoder']].wav2spec(meta_data['wav_fn'])
            else:
                wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(meta_data['wav_fn'])
        processed_input = {
            'item_name': item_name, 'mel': mel, 'wav': wav,
            'sec': len(mel) * hparams["hop_size"] / hparams["audio_sample_rate"], 'len': mel.shape[0]
        }
        processed_input = {**meta_data, **processed_input}  # merge two dicts
        try:
            if binarization_args['with_f0']:
                # get ground truth f0 by self.get_pitch_algorithm
                f0_path = f"{meta_data['wav_fn'][:-4]}_f0.npy"
                if os.path.exists(f0_path):
                    from utils.pitch_utils import f0_to_coarse
                    processed_input['f0'] = np.load(f0_path)
                    processed_input['pitch'] = f0_to_coarse(np.load(f0_path))
                else:
                    gt_f0, gt_pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
                    if sum(gt_f0) == 0:
                        raise BinarizationError("Empty **gt** f0")
                    processed_input['f0'] = gt_f0
                    processed_input['pitch'] = gt_pitch_coarse
            if binarization_args['with_txt']:
                try:
                    processed_input['phone'] = self.phone_encoder.encode(meta_data['ph'])
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    size = hparams['hop_size']
                    rate = hparams['audio_sample_rate']
                    mel2ph = np.zeros([mel.shape[0]], int)
                    startTime = 0
                    ph_durs = meta_data['ph_durs']
                    processed_input['ph_durs'] = np.asarray(ph_durs, dtype=np.float32)
                    for i_ph in range(len(ph_durs)):
                        start_frame = int(startTime * rate / size + 0.5)
                        end_frame = int((startTime + ph_durs[i_ph]) * rate / size + 0.5)
                        mel2ph[start_frame:end_frame] = i_ph + 1
                        startTime = startTime + ph_durs[i_ph]
                    processed_input['mel2ph'] = mel2ph
            if hparams.get('use_key_shift_embed', False):
                processed_input['key_shift'] = 0.
            if hparams.get('use_speed_embed', False):
                processed_input['speed'] = 1.
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {meta_data['wav_fn']}")
            return None
        return processed_input
