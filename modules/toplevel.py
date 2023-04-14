import torch
import torch.nn.functional as F

from basics.base_module import CategorizedModule
from modules.diffusion.ddpm import GaussianDiffusion
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.tts_modules import LengthRegulator
from modules.fastspeech.variance_encoder import FastSpeech2Variance, DummyPitchPredictor
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
    def __init__(self, vocab_size):
        super().__init__()
        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.lr = LengthRegulator()
        self.pitch_predictor = DummyPitchPredictor(
            vmin=hparams['pitch_delta_vmin'],
            vmax=hparams['pitch_delta_vmax'],
            num_bins=hparams['pitch_delta_num_bins'],
            deviation=hparams['pitch_delta_deviation']
        )

    @property
    def category(self):
        return 'variance'

    def forward(self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None,
                mel2ph=None, base_pitch=None, infer=True):
        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur, infer=infer
        )

        if not hparams['predict_pitch']:
            return dur_pred_out, None

        if mel2ph is None or hparams['dur_cascade']:
            # (extract mel2ph from dur_pred_out)
            raise NotImplementedError()

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)
        pitch_pred, pitch_probs = self.pitch_predictor(condition, base_pitch)
        if infer:
            return dur_pred_out, pitch_pred
        else:
            return dur_pred_out, pitch_probs
