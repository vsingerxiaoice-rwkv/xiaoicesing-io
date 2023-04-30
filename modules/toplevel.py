import torch
import torch.nn.functional as F

from basics.base_module import CategorizedModule
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
)
from modules.diffusion.ddpm import (
    GaussianDiffusion, PitchDiffusion, MultiVarianceDiffusion
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.tts_modules import LengthRegulator
from modules.fastspeech.variance_encoder import FastSpeech2Variance
from utils.hparams import hparams


class DiffSingerAcoustic(CategorizedModule):
    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )
        self.diffusion = GaussianDiffusion(
            out_dims=out_dims,
            num_feats=1,
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

    @property
    def category(self):
        return 'acoustic'

    def forward(
            self, txt_tokens, mel2ph, f0, energy=None, breathiness=None,
            key_shift=None, speed=None,
            spk_embed_id=None, gt_mel=None, infer=True, **kwargs
    ):
        condition = self.fs2(
            txt_tokens, mel2ph, f0, energy=energy, breathiness=breathiness,
            key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, **kwargs
        )
        if infer:
            mel = self.diffusion(condition, infer=True)
            mel *= ((mel2ph > 0).float()[:, :, None])
            return mel
        else:
            loss = self.diffusion(condition, gt_spec=gt_mel, infer=False)
            return loss


class DiffSingerVariance(CategorizedModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.predict_pitch = hparams['predict_pitch']

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        self.predict_variances = len(self.variance_prediction_list) > 0

        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.lr = LengthRegulator()

        if self.predict_pitch:
            pitch_hparams = hparams['pitch_prediction_args']
            self.base_pitch_embed = Linear(1, hparams['hidden_size'])
            self.pitch_predictor = PitchDiffusion(
                vmin=pitch_hparams['pitch_delta_vmin'],
                vmax=pitch_hparams['pitch_delta_vmax'],
                repeat_bins=pitch_hparams['num_pitch_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args=(
                    pitch_hparams['residual_layers'],
                    pitch_hparams['residual_channels']
                )
            )

        if self.predict_variances:
            self.pitch_embed = Linear(1, hparams['hidden_size'])

            ranges = []
            clamps = []

            if predict_energy:
                ranges.append((
                    10. ** (hparams['energy_db_min'] / 20.),
                    10. ** (hparams['energy_db_max'] / 20.)
                ))
                clamps.append((0., 1.))

            if predict_breathiness:
                ranges.append((
                    10. ** (hparams['breathiness_db_min'] / 20.),
                    10. ** (hparams['breathiness_db_max'] / 20.)
                ))
                clamps.append((0., 1.))

            variances_hparams = hparams['variances_prediction_args']
            self.variance_predictor = MultiVarianceDiffusion(
                ranges=ranges,
                clamps=clamps,
                repeat_bins=variances_hparams['repeat_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args=(
                    variances_hparams['residual_layers'],
                    variances_hparams['residual_channels']
                )
            )

    @property
    def category(self):
        return 'variance'

    def collect_variance_inputs(self, **kwargs):
        return [kwargs.get(name) for name in self.variance_prediction_list]

    def collect_variance_outputs(self, variances: list | tuple) -> dict:
        return {
            name: pred
            for name, pred in zip(self.variance_prediction_list, variances)
        }

    def forward(
            self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, mel2ph=None,
            base_pitch=None, delta_pitch=None, infer=True, **kwargs
    ):
        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur, infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, None, ({} if infer else None)

        if mel2ph is None or hparams['dur_cascade']:
            # (extract mel2ph from dur_pred_out)
            raise NotImplementedError()

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.predict_pitch:
            pitch_cond = condition + self.base_pitch_embed(base_pitch[:, :, None])
            pitch_pred_out = self.pitch_predictor(pitch_cond, delta_pitch, infer)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None)

        if delta_pitch is None:
            pitch = base_pitch + pitch_pred_out
        else:
            pitch = base_pitch + delta_pitch
        pitch_embed = self.pitch_embed(pitch[:, :, None])
        condition += pitch_embed

        variance_inputs = self.collect_variance_inputs(**kwargs)
        variance_outputs = self.variance_predictor(condition, variance_inputs, infer)
        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out
