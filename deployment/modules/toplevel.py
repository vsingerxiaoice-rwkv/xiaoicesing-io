import copy

from torch import Tensor, nn

from deployment.modules.diffusion import GaussianDiffusionONNX
from deployment.modules.fastspeech2 import FastSpeech2AcousticONNX
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from utils.hparams import hparams


class DiffSingerAcousticONNX(DiffSingerAcoustic):
    def __init__(self, vocab_size, out_dims):
        super().__init__(vocab_size, out_dims)
        del self.fs2
        del self.diffusion
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
        try:
            del model.variance_embeds
            del model.variance_adaptor
        except AttributeError:
            pass
        del model.diffusion
        model.forward = model.forward_fs2
        return model

    def view_as_adaptor(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        del model.diffusion
        raise NotImplementedError()
        

    def view_as_diffusion(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        try:
            del model.variance_embeds
            del model.variance_adaptor
        except AttributeError:
            pass
        model.forward = model.forward_diffusion
        return model


class DiffSingerVarianceOnnx(DiffSingerVariance):
    pass
