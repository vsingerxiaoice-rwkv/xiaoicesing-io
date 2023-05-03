import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, mel2ph_to_dur
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse
from utils.text_encoder import PAD_INDEX


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = FastSpeech2Encoder(
            self.txt_embed, hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads']
        )

        self.f0_embed_type = hparams.get('f0_embed_type', 'discrete')
        if self.f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, hparams['hidden_size'], PAD_INDEX)
        elif self.f0_embed_type == 'continuous':
            self.pitch_embed = Linear(1, hparams['hidden_size'])
        else:
            raise ValueError('f0_embed_type must be \'discrete\' or \'continuous\'.')

        self.variance_embed_list = []
        if hparams.get('use_energy_embed', False) and not hparams.get('predict_energy', False):
            # energy is embedded but not predicted
            self.variance_embed_list.append('energy')
        if hparams.get('use_breathiness_embed', False) and not hparams.get('predict_breathiness', False):
            # breathiness is embedded but not predicted
            self.variance_embed_list.append('breathiness')
        self.embed_variances = len(self.variance_embed_list) > 0
        if self.embed_variances:
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_embed_list
            })

        self.use_key_shift_embed = hparams.get('use_key_shift_embed', False)
        if self.use_key_shift_embed:
            self.key_shift_embed = Linear(1, hparams['hidden_size'])

        self.use_speed_embed = hparams.get('use_speed_embed', False)
        if self.use_speed_embed:
            self.speed_embed = Linear(1, hparams['hidden_size'])

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

    def forward(
            self, txt_tokens, mel2ph, f0, energy=None, breathiness=None,
            key_shift=None, speed=None, spk_embed_id=None, infer=True,
            **kwargs
    ):
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        dur_embed = self.dur_embed(dur[:, :, None])
        encoder_out = self.encoder(txt_tokens, dur_embed)

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.use_spk_id:
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if spk_mix_embed is not None:
                spk_embed = spk_mix_embed
            else:
                spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
            condition += spk_embed

        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])

        adaptor_cond = condition + pitch_embed
        mel_cond = self.forward_variance_embedding(
            adaptor_cond, key_shift=key_shift, speed=speed, **kwargs
        )

        # During training, the data augmentation parameters (GEN and VEL)
        # are seen to the variance adaptor; but during inference,
        # we will always send the DEFAULT parameters (GEN = 0 and VEL = 1)
        # to the variance adaptor so that the prediction outputs will NOT
        # be influenced by these parameters, which is more reasonable for
        # most users of singing voice synthesis systems.
        if self.use_key_shift_embed:
            if infer:
                key_shift = torch.zeros_like(key_shift)
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            adaptor_cond = adaptor_cond + key_shift_embed

        if self.use_speed_embed:
            if infer:
                speed = torch.ones_like(speed)
            speed_embed = self.speed_embed(speed[:, :, None])
            adaptor_cond = adaptor_cond + speed_embed

        return adaptor_cond, mel_cond

    def forward_variance_embedding(self, condition, key_shift=None, speed=None, **variances):
        if self.embed_variances:
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variances[v_name][:, :, None])  # [B, T] => [B, T, H]
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition = condition + variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition = condition + key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None])
            condition = condition + speed_embed

        return condition
