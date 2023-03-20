import argparse
import pathlib
from collections import OrderedDict

import torch


parser = argparse.ArgumentParser(description='Migrate checkpoint files of MIDI-less acoustic models from old format')
parser.add_argument('input', type=str, help='Path to the input file')
parser.add_argument('output', type=str, help='Path to the output file')
parser.add_argument('--overwrite', required=False, default=False,
                    action='store_true', help='Overwrite the existing file')
args = parser.parse_args()

input_ckpt = pathlib.Path(args.input).resolve()
output_ckpt = pathlib.Path(args.output).resolve()
assert input_ckpt.exists(), 'The input file does not exist.'
assert args.overwrite or not output_ckpt.exists(), \
    'The output file already exists or is the same as the input file.\n' \
    'This is not recommended because migration scripts may not be stable, ' \
    'and you may be at risk of losing your model.\n' \
    'If you are sure to OVERWRITE the existing file, please re-run this script with the \'--overwrite\' argument.'

ckpt_loaded = torch.load(input_ckpt)
if 'category' in ckpt_loaded:
    print('This checkpoint file is already in the new format.')
    exit(0)
state_dict: OrderedDict = ckpt_loaded['state_dict']
ckpt_loaded['optimizer_states'][0]['state'].clear()
new_state_dict = OrderedDict()
for key in state_dict:
    if key.startswith('model.fs2'):
        # keep model.fs2.xxx
        new_state_dict[key] = state_dict[key]
    else:
        # model.xxx => model.diffusion.xxx
        path = key.split('.', maxsplit=1)[1]
        new_state_dict[f'model.diffusion.{path}'] = state_dict[key]
ckpt_loaded['category'] = 'acoustic'
ckpt_loaded['state_dict'] = new_state_dict
torch.save(ckpt_loaded, output_ckpt)
