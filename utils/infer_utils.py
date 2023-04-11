import re

import librosa
import numpy as np
from scipy.io import wavfile

head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def merge_slurs(param):
    ph_seq = param['ph_seq'].split()
    note_seq = param['note_seq'].split()
    note_dur_seq = param['note_dur_seq'].split()
    is_slur_seq = [int(s) for s in param['is_slur_seq'].split()]
    ph_dur = [float(d) for d in param['ph_dur'].split()]
    i = 0
    while i < len(ph_seq):
        if is_slur_seq[i]:
            ph_dur[i - 1] += ph_dur[i]
            ph_seq.pop(i)
            note_seq.pop(i)
            note_dur_seq.pop(i)
            is_slur_seq.pop(i)
            ph_dur.pop(i)
        else:
            i += 1
    param['ph_seq'] = ' '.join(ph_seq)
    param['note_seq'] = ' '.join(note_seq)
    param['note_dur_seq'] = ' '.join(note_dur_seq)
    param['is_slur_seq'] = ' '.join([str(s) for s in is_slur_seq])
    param['ph_dur'] = ' '.join([str(d) for d in ph_dur])


def trans_f0_seq(feature_pit, transform):
    feature_pit = feature_pit * 2 ** (transform / 12)
    return round(feature_pit, 1)


def move_key(raw_data, mv_key):
    head = raw_data[:-1]
    body = int(raw_data[-1])
    new_head_index = head_list.index(head) + mv_key
    while new_head_index < 0:
        body -= 1
        new_head_index += 12
    while new_head_index > 11:
        body += 1
        new_head_index -= 12
    result_data = head_list[new_head_index] + str(body)
    return result_data


def trans_key(raw_data, key):
    warning_tag = False
    for i in raw_data:
        note_seq_list = i["note_seq"].split(" ")
        new_note_seq_list = []
        for note_seq in note_seq_list:
            if note_seq != "rest":
                new_note_seq = librosa.midi_to_note(librosa.note_to_midi(note_seq) + key)
                # new_note_seq = move_key(note_seq, key)
                new_note_seq_list.append(new_note_seq)
            else:
                new_note_seq_list.append(note_seq)
        i["note_seq"] = " ".join(new_note_seq_list)
        if i["f0_seq"]:
            f0_seq_list = i["f0_seq"].split(" ")
            f0_seq_list = [float(x) for x in f0_seq_list]
            new_f0_seq_list = []
            for f0_seq in f0_seq_list:
                new_f0_seq = trans_f0_seq(f0_seq, key)
                new_f0_seq_list.append(str(new_f0_seq))
            i["f0_seq"] = " ".join(new_f0_seq_list)
        else:
            warning_tag = True
    if warning_tag:
        print("Warning: parts of f0_seq do not exist, please freeze the pitch line in the editor.\r\n")
    return raw_data


def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


def parse_commandline_spk_mix(mix: str) -> dict:
    """
    Parse speaker mix info from commandline
    :param mix: Input like "opencpop" or "opencpop|qixuan" or "opencpop:0.5|qixuan:0.5"
    :return: A dict whose keys are speaker names and values are proportions
    """
    name_pattern = r'[0-9A-Za-z_-]+'
    proportion_pattern = r'\d+(\.\d+)?'
    single_pattern = rf'{name_pattern}(:{proportion_pattern})?'
    assert re.fullmatch(rf'{single_pattern}(\|{single_pattern})*', mix) is not None, f'Invalid mix pattern: {mix}'
    without_proportion = set()
    proportion_map = {}
    for component in mix.split('|'):
        # If already exists
        name_and_proportion = component.split(':')
        assert name_and_proportion[0] not in without_proportion and name_and_proportion[0] not in proportion_map, \
            f'Duplicate speaker name: {name_and_proportion[0]}'
        if ':' in component:
            proportion_map[name_and_proportion[0]] = float(name_and_proportion[1])
        else:
            without_proportion.add(name_and_proportion[0])
    sum_given_proportions = sum(proportion_map.values())
    assert sum_given_proportions < 1 or len(without_proportion) == 0, \
        'Proportion of all speakers should be specified if the sum of all given proportions are larger than 1.'
    for name in without_proportion:
        proportion_map[name] = (1 - sum_given_proportions) / len(without_proportion)
    sum_all_proportions = sum(proportion_map.values())
    assert sum_all_proportions > 0, 'Sum of all proportions should be positive.'
    for name in proportion_map:
        proportion_map[name] /= sum_all_proportions
    return proportion_map


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))
