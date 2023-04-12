import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.hparams import hparams
from utils.text_encoder import PAD_INDEX


class FastSpeech2VarianceEncoder(FastSpeech2Encoder):
    def forward_embedding(self, txt_tokens, midi_embed, midi_dur_embed):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embed + midi_dur_embed
        if hparams['use_pos_embed']:
            if hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embed, midi_dur_embed):
        """
        :param txt_tokens: [B, T]
        :param midi_embed: [B, T, H]
        :param midi_dur_embed: [B, T, H]
        :return: [T x B x H]
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, midi_embed, midi_dur_embed)  # [B, T, H]
        x = super()._forward(x, encoder_padding_mask)
        return x


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.midi_embed = Embedding(128, hparams['hidden_size'], PAD_INDEX)
        self.midi_dur_embed = Linear(1, hparams['hidden_size'])
        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.encoder = FastSpeech2VarianceEncoder(
            self.txt_embed, hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads']
        )

        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            hparams['hidden_size'],
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel']
        )

    def forward(self, txt_tokens, midi, midi_dur, gt_ph_dur, **kwargs):
        raise NotImplementedError()
