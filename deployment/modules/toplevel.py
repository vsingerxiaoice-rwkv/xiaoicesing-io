import copy

from torch import Tensor, nn

from basics.base_module import CategorizedModule
from deployment.modules.diffusion import GaussianDiffusionONNX
from deployment.modules.fastspeech2 import FastSpeech2AcousticONNX
from utils.hparams import hparams


class DiffSingerAcousticONNX(CategorizedModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.fs2 = FastSpeech2AcousticONNX(
            vocab_size=vocab_size
        )
        self.diffusion = GaussianDiffusionONNX(
            out_dims=out_dims,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            denoiser_args=(
                hparams['residual_layers'],
                hparams['residual_channels']
            ),
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    def forward_fs2(
            self,
            tokens: Tensor,
            durations: Tensor,
            f0: Tensor,
            gender: Tensor = None,
            velocity: Tensor = None,
            spk_embed: Tensor = None
    ) -> Tensor:
        return self.fs2(
            tokens, durations, f0,
            gender=gender, velocity=velocity, spk_embed=spk_embed
        )

    def forward_diffusion(self, condition: Tensor, speedup: int) -> Tensor:
        return self.diffusion(condition, speedup)

    def view_as_fs2(self) -> nn.Module:
        model = copy.deepcopy(self)
        model.diffusion = None
        model.forward = model.forward_fs2
        return model

    def view_as_diffusion(self) -> nn.Module:
        model = copy.deepcopy(self)
        model.fs2 = None
        model.forward = model.forward_diffusion
        return model


class DiffSingerVarianceOnnx(CategorizedModule):
    @property
    def category(self):
        return 'variance'
