import torch
from basics.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel.acoustic_model import DiffSingerAcoustic
import librosa
import numpy as np


class DiffSingerCascadeInfer(BaseSVSInfer):
    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.ph_encoder),
            out_dims=hparams['audio_num_mel_bins']
        )
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model', ckpt_steps=ckpt_steps,
                  required_category='acoustic', strict=True, device=self.device)
        return model

    def preprocess_word_level_input(self, inp):
        return super().preprocess_word_level_input(inp)

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = np.array(inp['is_slur_seq'].split(), 'float')
        ph_dur = None
        f0_timestep = float(inp['f0_timestep'])
        f0_seq = None
        gender_timestep = None
        gender = 0.
        if inp['f0_seq'] is not None:
            f0_seq = np.array(inp['f0_seq'].split(), 'float')
        if inp.get('gender') is not None:
            if isinstance(inp['gender'], str):
                gender_timestep = float(inp['gender_timestep'])
                gender = np.array(inp['gender'].split(), 'float')
            else:
                gender = float(inp['gender'])
        velocity_timestep = None
        velocity = None
        if inp.get('velocity') is not None:
            velocity_timestep = float(inp['velocity_timestep'])
            velocity = np.array(inp['velocity'].split(), 'float')
        ph_seq_lst = ph_seq.split()
        if inp['ph_dur'] is not None:
            ph_dur = np.array(inp['ph_dur'].split(), 'float')
            if not len(note_lst) == len(ph_seq_lst) == len(midi_dur_lst) == len(ph_dur):
                raise RuntimeError(f'The number of notes, phones and durations mismatch:'
                                   f'{len(note_lst)} {len(ph_seq.split())} {len(midi_dur_lst)} {len(ph_dur)}')
        else:
            if not len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
                raise RuntimeError(f'The number of notes, phones and durations mismatch:'
                                   f'{len(note_lst)} {len(ph_seq.split())} {len(midi_dur_lst)}')
        print(f'Processed {len(ph_seq_lst)} tokens: {" ".join(ph_seq_lst)}')

        return ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur, \
            f0_timestep, f0_seq, gender_timestep, gender, velocity_timestep, velocity

    def preprocess_input(self, inp, input_type='word'):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        if hparams['use_spk_id']:
            spk_mix = inp.get('spk_mix')
            if spk_mix is None:
                for name in self.spk_map.keys():
                    spk_mix = {name: 1.0}
                    break
            else:
                for name in spk_mix:
                    assert name in self.spk_map, f'Speaker \'{name}\' not found.'
            if len(spk_mix) == 1:
                print(f'Using speaker \'{list(spk_mix.keys())[0]}\'')
            elif any([isinstance(val, list) for val in spk_mix.values()]):
                print_mix = '|'.join(spk_mix.keys())
                print(f'Using dynamic speaker mix \'{print_mix}\'')
            else:
                print_mix = '|'.join([f'{n}:{"%.3f" % spk_mix[n]}' for n in spk_mix])
                print(f'Using static speaker mix \'{print_mix}\'')
        else:
            spk_mix = None

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ph_seq, note_lst, midi_dur_lst, is_slur = self.preprocess_word_level_input(inp)
            ph_dur = f0_timestep = f0_seq = gender_timestep = gender = velocity_timestep = velocity = None
        elif input_type == 'phoneme':  # like transcriptions.txt in Opencpop dataset.
            ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur, \
                f0_timestep, f0_seq, gender_timestep, gender, velocity_timestep, velocity = \
                self.preprocess_phoneme_level_input(inp)
        else:
            raise ValueError('Invalid input type. Must be \'word\' or \'phoneme\'.')

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                     for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_mix': spk_mix,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'ph_dur': None, 'f0_timestep': 0., 'f0_seq': None}
        item['ph_len'] = len(item['ph_token'])
        if input_type == 'phoneme':
            item['ph_dur'] = ph_dur
            item['f0_timestep'] = f0_timestep
            item['f0_seq'] = f0_seq
            item['gender_timestep'] = gender_timestep
            item['gender'] = gender
            item['velocity_timestep'] = velocity_timestep
            item['velocity'] = velocity
            item['spk_mix_timestep'] = inp.get('spk_mix_timestep')
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        if hparams['use_spk_id']:
            spk_mix_map = item['spk_mix']
            dynamic_mix = any([isinstance(val, list) for val in spk_mix_map.values()])
            max_length = max([len(val) for val in spk_mix_map.values()]) if dynamic_mix else 0
            if dynamic_mix:
                mix_value_list = []
                for spk_name in spk_mix_map:
                    if isinstance(spk_mix_map[spk_name], list):
                        mix_seq = spk_mix_map[spk_name] + \
                                  [spk_mix_map[spk_name][-1]] * (max_length - len(spk_mix_map[spk_name]))
                    else:
                        mix_seq = [spk_mix_map[spk_name]] * max_length
                    timestep = item['spk_mix_timestep']
                    t_max = (max_length - 1) * timestep
                    dt = hparams['hop_size'] / hparams['audio_sample_rate']
                    mix_value_list.append(np.interp(np.arange(0, t_max, dt), timestep * np.arange(max_length), mix_seq))
                mix_value_ndarray = np.stack(mix_value_list, axis=0)
                assert np.all(mix_value_ndarray >= 0), 'All proportion values of speaker mix should be non-negative.'
                frame_sum = mix_value_ndarray.sum(axis=0)[None, :]
                assert np.all(frame_sum > 0), 'Proportions of speaker mix on some frames sum to zero.'
                mix_value_list = list(mix_value_ndarray / frame_sum)
                spk_mixes = {
                    torch.LongTensor([self.spk_map[n]]).to(self.device) : torch.FloatTensor(mix_value_list[i][None, :, None]).to(self.device)
                    for i, n in enumerate(spk_mix_map.keys())
                }
            else:
                assert all([val >= 0 for val in spk_mix_map.values()]), 'All proportion values of speaker mix should be non-negative.'
                proportion_sum = sum(spk_mix_map.values())
                assert proportion_sum > 0, 'Proportions of speaker mix sum to zero.'
                spk_mixes = {
                    torch.LongTensor([self.spk_map[n]]).to(self.device) : spk_mix_map[n] / proportion_sum
                    for n in spk_mix_map
                }
        else:
            spk_mixes = None
        pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)
        mel2ph = None
        f0 = None
        if item['ph_dur'] is not None:
            print('Using manual phone duration')
            ph_acc = np.around(
                np.add.accumulate(item['ph_dur']) * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5).astype(
                'int')
            ph_dur = np.diff(ph_acc, prepend=0)
            ph_dur = torch.LongTensor(ph_dur)[None, :hparams['max_frames']].to(self.device)
            lr = LengthRegulator()
            mel2ph = lr(ph_dur, txt_tokens == 0).detach()
        else:
            print('Using automatic phone duration')

        if item['f0_timestep'] > 0. and item['f0_seq'] is not None:
            print('Using manual pitch curve')
            f0_timestep = item['f0_timestep']
            f0_seq = item['f0_seq']
            t_max = (len(f0_seq) - 1) * f0_timestep
            dt = hparams['hop_size'] / hparams['audio_sample_rate']
            f0_interp = np.interp(np.arange(0, t_max, dt), f0_timestep * np.arange(len(f0_seq)), f0_seq)
            f0 = torch.FloatTensor(f0_interp)[None, :].to(self.device)
        else:
            print('Using automatic pitch curve')

        if hparams.get('use_key_shift_embed', False):
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            if isinstance(item['gender'], float):
                print(f'Using static gender value: {item["gender"]:.3f}')
                gender = item['gender']
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                key_shift = torch.FloatTensor([key_shift_value]).to(self.device)
            else:
                print('Using dynamic gender curve')
                gender_timestep = item['gender_timestep']
                gender_seq = item['gender']
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                t_max = (len(key_shift_seq) - 1) * gender_timestep
                dt = hparams['hop_size'] / hparams['audio_sample_rate']
                key_shift_interp = np.interp(np.arange(0, t_max, dt), gender_timestep * np.arange(len(key_shift_seq)), key_shift_seq)
                key_shift = torch.FloatTensor(key_shift_interp)[None, :].to(self.device)
        else:
            key_shift = None

        if hparams.get('use_speed_embed', False):
            if item['velocity'] is None:
                print('Using default velocity curve')
                speed = torch.FloatTensor([1.]).to(self.device)
            else:
                print('Using manual velocity curve')
                velocity_timestep = item['velocity_timestep']
                velocity_seq = item['velocity']
                speed_min, speed_max = hparams['augmentation_args']['random_time_stretching']['range']
                speed_seq = np.clip(velocity_seq, a_min=speed_min, a_max=speed_max)
                t_max = (len(speed_seq) - 1) * velocity_timestep
                dt = hparams['hop_size'] / hparams['audio_sample_rate']
                speed_interp = np.interp(np.arange(0, t_max, dt), velocity_timestep * np.arange(len(speed_seq)), speed_seq)
                speed = torch.FloatTensor(speed_interp)[None, :].to(self.device)
        else:
            speed = None

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_mixes': spk_mixes,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur,
            'mel2ph': mel2ph,
            'f0': f0,
            'key_shift': key_shift,
            'speed': speed
        }
        return batch

    def forward_model(self, inp, return_mel=False):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        with torch.no_grad():
            if hparams['use_spk_id']:
                spk_mixes = sample['spk_mixes']
                spk_mix_embed = [self.model.fs2.spk_embed(spk_id)[:, None, :] * spk_mixes[spk_id] for spk_id in
                                 spk_mixes]
                spk_mix_embed = torch.stack(spk_mix_embed, dim=1).sum(dim=1)
            else:
                spk_mix_embed = None
            mel2ph = sample['mel2ph']
            f0 = sample['f0']
            nframes = mel2ph.size(1)
            delta_l = nframes - f0.size(1)
            if delta_l > 0:
                f0 = torch.cat((f0,torch.FloatTensor([[x[-1]] * delta_l for x in f0]).to(f0.device)),1)
            f0 = f0[:, :nframes]
            mel = self.model(txt_tokens, mel2ph=sample['mel2ph'], f0=sample['f0'],
                                key_shift=sample['key_shift'], speed=sample['speed'],
                                spk_mix_embed=spk_mix_embed, infer=True)
            if return_mel:
                return mel.cpu(), f0.cpu()
            wav_out = self.run_vocoder(mel, f0=f0)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]
