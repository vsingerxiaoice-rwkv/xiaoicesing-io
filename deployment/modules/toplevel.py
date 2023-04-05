from torch import Tensor

from basics.base_module import CategorizedModule
from deployment.modules.diffusion import GaussianDiffusionOnnx
from deployment.modules.fastspeech2 import FastSpeech2AcousticOnnx
from utils.hparams import hparams


class DiffSingerAcousticOnnx(CategorizedModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims, frozen_gender=None, frozen_spk_embed=None):
        super().__init__()
        self.fs2 = FastSpeech2AcousticOnnx(
            vocab_size=vocab_size,
            frozen_gender=frozen_gender,
            frozen_spk_embed=frozen_spk_embed
        )
        self.diffusion = GaussianDiffusionOnnx(
            out_dims=out_dims,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    def forward(self, tokens: Tensor, durations: Tensor, f0: Tensor, speedup: Tensor) -> Tensor:
        condition = self.fs2(tokens, durations, f0)
        mel = self.diffusion(condition, speedup=speedup)
        return mel


class DiffSingerVarianceOnnx(CategorizedModule):
    @property
    def category(self):
        return 'variance'
