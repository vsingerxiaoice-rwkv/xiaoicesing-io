import shutil
from pathlib import Path
from typing import List, Union, Tuple, Dict

import onnx
import onnxsim
import torch

from basics.base_exporter import BaseExporter
from deployment.modules.toplevel import DiffSingerAcousticONNX
from utils import load_ckpt, onnx_helper
from utils.hparams import hparams
from utils.phoneme_utils import locate_dictionary, build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class DiffSingerAcousticExporter(BaseExporter):
    def __init__(
            self,
            device: Union[str, torch.device] = None,
            cache_dir: Path = None,
            ckpt_steps: int = None,
            expose_gender: bool = False,
            freeze_gender: float = None,
            expose_velocity: bool = False,
            export_spk: List[Tuple[str, Dict[str, float]]] = None,
            freeze_spk: Tuple[str, Dict[str, float]] = None
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        # Basic attributes
        self.model_name: str = hparams['exp_name']
        self.ckpt_steps: int = ckpt_steps
        self.spk_map: dict = self.build_spk_map()
        self.vocab = TokenTextEncoder(vocab_list=build_phoneme_list())
        self.model = self.build_model()
        self.fs2_cache_path = self.cache_dir / 'fs2.onnx'
        self.diffusion_cache_path = self.cache_dir / 'diffusion.onnx'

        # Attributes for logging
        self.fs2_class_name = self.model.fs2.__class__.__name__.removesuffix('ONNX')
        self.denoiser_class_name = self.model.diffusion.denoise_fn.__class__.__name__.removesuffix('ONNX')
        self.diffusion_class_name = self.model.diffusion.__class__.__name__.removesuffix('ONNX')

        # Attributes for exporting
        self.expose_gender = expose_gender
        self.expose_velocity = expose_velocity
        self.freeze_spk: Tuple[str, Dict[str, float]] = freeze_spk
        self.export_spk: List[Tuple[str, Dict[str, float]]] = export_spk if export_spk is not None else []
        if hparams.get('use_key_shift_embed', False) and not self.expose_gender:
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            key_shift = freeze_gender * shift_max if freeze_gender >= 0. else freeze_gender * abs(shift_min)
            key_shift = max(min(key_shift, shift_max), shift_min)  # clip key shift
            self.model.fs2.register_buffer('frozen_key_shift', torch.FloatTensor([key_shift]).to(self.device))
        if hparams['use_spk_id'] and not self.export_spk and freeze_spk is not None:
            self.model.fs2.register_buffer('spk_mix_embed', self._perform_spk_mix(freeze_spk[1]))

    def build_model(self) -> DiffSingerAcousticONNX:
        model = DiffSingerAcousticONNX(
            vocab_size=len(self.vocab),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], 'model', ckpt_steps=self.ckpt_steps,
                  required_category='acoustic', strict=True, device=self.device)
        return model

    def export(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        self.export_model((path / model_name).with_suffix('.onnx'))
        self.export_attachments(path)

    def export_model(self, path: Path):
        self._torch_export_model()
        fs2_onnx = self._optimize_fs2_graph(onnx.load(self.fs2_cache_path))
        diffusion_onnx = self._optimize_diffusion_graph(onnx.load(self.diffusion_cache_path))
        model_onnx = self._merge_fs2_diffusion_graphs(fs2_onnx, diffusion_onnx)
        onnx.save(model_onnx, path)
        self.fs2_cache_path.unlink()
        self.diffusion_cache_path.unlink()
        print(f'| export model => {path}')

    def export_attachments(self, path: Path):
        path_model_name = path / self.model_name
        for spk in self.export_spk:
            self._export_spk_embed(path.with_suffix(f'.{spk[0]}.emb'), self._perform_spk_mix(spk[1]))
        self._export_dictionary(path / 'dictionary.txt')
        self._export_phonemes(path_model_name.with_suffix('.phonemes.txt'))

    @torch.no_grad()
    def _torch_export_model(self):
        # Prepare inputs for FastSpeech2 tracing
        n_frames = 10
        tokens = torch.LongTensor([[1]]).to(self.device)
        durations = torch.LongTensor([[n_frames]]).to(self.device)
        f0 = torch.FloatTensor([[440.] * n_frames]).to(self.device)
        kwargs: dict[str, torch.Tensor] = {}
        arguments = (tokens, durations, f0, kwargs)
        input_names = ['tokens', 'durations', 'f0']
        dynamix_axes = {
            'tokens': {
                1: 'n_tokens'
            },
            'durations': {
                1: 'n_tokens'
            },
            'f0': {
                1: 'n_frames'
            }
        }
        if hparams.get('use_key_shift_embed', False):
            if self.expose_gender:
                kwargs['gender'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('gender')
                dynamix_axes['gender'] = {
                    1: 'n_frames'
                }
        if hparams.get('use_speed_embed', False):
            if self.expose_velocity:
                kwargs['velocity'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('velocity')
                dynamix_axes['velocity'] = {
                    1: 'n_frames'
                }
        if hparams['use_spk_id']:
            kwargs['spk_embed'] = torch.rand(
                (1, n_frames, hparams['hidden_size']),
                dtype=torch.float32, device=self.device
            )
            input_names.append('spk_embed')
            dynamix_axes['spk_embed'] = {
                1: 'n_frames'
            }
        dynamix_axes['condition'] = {
            1: 'n_frames'
        }

        # PyTorch ONNX export for FastSpeech2
        print(f'Exporting {self.fs2_class_name}...')
        torch.onnx.export(
            self.model.view_as_fs2(),
            arguments,
            self.fs2_cache_path,
            input_names=input_names,
            output_names=['condition'],
            dynamic_axes=dynamix_axes,
            opset_version=15
        )

        # Prepare inputs for denoiser tracing and GaussianDiffusion scripting
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        noise = torch.randn(shape, device=self.device)
        condition = torch.rand((1, hparams['hidden_size'], n_frames), device=self.device)
        step = (torch.rand((1,), device=self.device) * hparams['K_step']).long()

        print(f'Tracing {self.denoiser_class_name} denoiser...')
        diffusion = self.model.view_as_diffusion()
        diffusion.diffusion.denoise_fn = torch.jit.trace(
            diffusion.diffusion.denoise_fn,
            (
                noise,
                step,
                condition
            )
        )

        print(f'Scripting {self.diffusion_class_name}...')
        diffusion = torch.jit.script(
            diffusion,
            example_inputs=[
                (
                    condition.transpose(1, 2),
                    1  # p_sample branch
                ),
                (
                    condition.transpose(1, 2),
                    200  # p_sample_plms branch
                )
            ]
        )

        # PyTorch ONNX export for GaussianDiffusion
        print(f'Exporting {self.diffusion_class_name}...')
        torch.onnx.export(
            diffusion,
            (
                condition.transpose(1, 2),
                200
            ),
            self.diffusion_cache_path,
            input_names=[
                'condition', 'speedup'
            ],
            output_names=[
                'mel'
            ],
            dynamic_axes={
                'condition': {
                    1: 'n_frames'
                },
                'mel': {
                    1: 'n_frames'
                }
            },
            opset_version=15
        )

    def _perform_spk_mix(self, spk_mix: Dict[str, float]):
        spk_mix_ids = []
        spk_mix_values = []
        for name, value in spk_mix.items():
            spk_mix_ids.append(self.spk_map[name])
            assert value >= 0., f'Speaker mix checks failed.\n' \
                                f'Proportion of speaker \'{name}\' is negative.'
            spk_mix_values.append(value)
        spk_mix_id_N = torch.LongTensor(spk_mix_ids).to(self.device)[None]  # => [1, N]
        spk_mix_value_N = torch.FloatTensor(spk_mix_values).to(self.device)[None]  # => [1, N]
        spk_mix_value_sum = spk_mix_value_N.sum()
        assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                       f'Proportions of speaker mix sum to zero.'
        spk_mix_value_N /= spk_mix_value_sum  # normalize
        spk_mix_embed = torch.sum(
            self.model.fs2.spk_embed(spk_mix_id_N) * spk_mix_value_N.unsqueeze(2),  # => [1, N, H]
            dim=1, keepdim=False
        )  # => [1, H]
        return spk_mix_embed

    def _optimize_fs2_graph(self, fs2: onnx.ModelProto) -> onnx.ModelProto:
        print(f'Running ONNX simplifier for {self.fs2_class_name}...')
        fs2, check = onnxsim.simplify(
            fs2,
            include_subgraph=True
        )
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.fs2_class_name}')
        return fs2

    def _optimize_diffusion_graph(self, diffusion: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(diffusion, output_shapes={
            'mel': (1, 'n_frames', hparams['audio_num_mel_bins'])
        })
        print(f'Running ONNX simplifier #1 for {self.diffusion_class_name}...')
        diffusion, check = onnxsim.simplify(
            diffusion,
            include_subgraph=True
        )
        assert check, 'Simplified ONNX model could not be validated'
        onnx_helper.graph_fold_back_to_squeeze(diffusion.graph)
        onnx_helper.graph_extract_conditioner_projections(
            graph=diffusion.graph, op_type='Conv',
            weight_pattern=r'diffusion\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
            alias_prefix='/diffusion/denoise_fn/cache'
        )
        onnx_helper.graph_remove_unused_values(diffusion.graph)
        print(f'Running ONNX simplifier #2 for {self.diffusion_class_name}...')
        diffusion, check = onnxsim.simplify(
            diffusion,
            include_subgraph=True
        )
        print(f'| optimize graph: {self.diffusion_class_name}')
        assert check, 'Simplified ONNX model could not be validated'
        return diffusion

    def _merge_fs2_diffusion_graphs(self, fs2: onnx.ModelProto, diffusion: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_add_prefixes(fs2, dim_prefix='fs2.', ignored_pattern=r'(n_tokens)|(n_frames)')
        onnx_helper.model_add_prefixes(diffusion, dim_prefix='diffusion.', ignored_pattern='n_frames')
        print(f'Merging {self.fs2_class_name} and {self.diffusion_class_name} '
              f'back into {self.model.__class__.__name__}...')
        merged = onnx.compose.merge_models(
            fs2, diffusion, io_map=[('condition', 'condition')],
            prefix1='', prefix2='', doc_string='',
            producer_name=fs2.producer_name, producer_version=fs2.producer_version,
            domain=fs2.domain, model_version=fs2.model_version
        )
        merged.graph.name = fs2.graph.name
        return merged

    # noinspection PyMethodMayBeStatic
    def _export_spk_embed(self, path: Path, spk_embed: torch.Tensor):
        with open(path, 'wb') as f:
            f.write(spk_embed.cpu().numpy().tobytes())
        print(f'| export spk embed => {path}')

    # noinspection PyMethodMayBeStatic
    def _export_dictionary(self, path: Path):
        print(f'| export dictionary => {path}')
        shutil.copy(locate_dictionary(), path)

    def _export_phonemes(self, path: Path):
        self.vocab.store_to_file(path)
        print(f'| export phonemes => {path}')
