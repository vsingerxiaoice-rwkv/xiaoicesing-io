'''
    file -> temporary_dict -> processed_input -> batch
'''

import torch

import utils
from utils.hparams import hparams
from utils.phoneme_utils import get_all_vowels

vowels = get_all_vowels()


class File2Batch:
    """
        pipeline: file -> temporary_dict -> processed_input -> batch
    """
    @staticmethod
    def processed_input2batch(samples):
        """
            Args:
                samples: one batch of processed_input
            NOTE:
                the batch size is controlled by hparams['max_sentences']
        """
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }
        if hparams['use_energy_embed']:
            batch['energy'] = utils.collate_1d([s['energy'] for s in samples], 0.0)
        if hparams.get('use_key_shift_embed', False):
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])
        if hparams.get('use_speed_embed', False):
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])
        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = utils.collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        elif hparams['pitch_type'] == 'ph':
            batch['f0'] = utils.collate_1d([s['f0_ph'] for s in samples])

        batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
        batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
        batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
        batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)

        return batch
