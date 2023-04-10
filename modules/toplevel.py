from basics.base_module import CategorizedModule
from modules.diffusion.ddpm import GaussianDiffusion
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from utils.hparams import hparams


class DiffSingerAcoustic(CategorizedModule):
    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )
        self.diffusion = GaussianDiffusion(
            out_dims=out_dims,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    @property
    def category(self):
        return 'acoustic'

    def forward(self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
                spk_embed_id=None, gt_mel=None, infer=True, **kwargs):
        condition = self.fs2(txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
                             spk_embed_id=spk_embed_id, **kwargs)
        if infer:
            mel = self.diffusion(condition, infer=True)
            mel *= ((mel2ph > 0).float()[:, :, None])
            return mel
        else:
            loss = self.diffusion(condition, gt_spec=gt_mel, infer=False)
            return loss


class DiffSingerVariance(CategorizedModule):
    @property
    def category(self):
        return 'variance'
