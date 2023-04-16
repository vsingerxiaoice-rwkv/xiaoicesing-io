import torch
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
    def forward_embedding(self, txt_tokens, midi_embed, onset_embed, word_dur_embed):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embed + onset_embed + word_dur_embed
        if hparams['use_pos_embed']:
            if hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embed, onset_embed, word_dur_embed):
        """
        :param txt_tokens: [B, T]
        :param midi_embed: [B, T, H]
        :param onset_embed: [B, T, H]
        :param word_dur_embed: [B, T, H]
        :return: [T x B x H]
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, midi_embed, onset_embed, word_dur_embed)  # [B, T, H]
        x = super()._forward(x, encoder_padding_mask)
        return x


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.midi_embed = Embedding(128, hparams['hidden_size'], PAD_INDEX)
        self.onset_embed = Embedding(2, hparams['hidden_size'])
        self.word_dur_embed = Linear(1, hparams['hidden_size'])

        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.encoder = FastSpeech2VarianceEncoder(
            self.txt_embed, hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads']
        )

        dur_hparams = hparams['dur_prediction_args']
        self.wdur_log_offset = dur_hparams['log_offset']
        if hparams['predict_dur']:
            self.dur_predictor = DurationPredictor(
                in_dims=hparams['hidden_size'],
                n_chans=dur_hparams['hidden_size'],
                n_layers=dur_hparams['num_layers'],
                dropout_rate=dur_hparams['dropout'],
                padding=hparams['ffn_padding'],
                kernel_size=dur_hparams['kernel_size'],
                offset=dur_hparams['log_offset'],
                dur_loss_type=dur_hparams['loss_type']
            )

    def forward(self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, infer=True):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param infer: whether inference
        :return: (train) encoder_out, ph_dur_xs; (infer) encoder_out, ph_dur
        """
        b = txt_tokens.shape[0]
        midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
        onset = torch.diff(ph2word, dim=1, prepend=ph2word.new_zeros(b, 1)) > 0
        onset_embed = self.onset_embed(onset.long())  # [B, T_ph, H]
        if word_dur is None or not infer:
            word_dur = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
                1, ph2word, ph_dur
            )[:, 1:]  # [B, T_ph] => [B, T_w]
        word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 1, ph2word)  # [B, T_w] => [B, T_ph]
        word_dur_embed = self.word_dur_embed(word_dur.float()[:, :, None])
        encoder_out = self.encoder(txt_tokens, midi_embed, onset_embed, word_dur_embed)

        if not hparams['predict_dur']:
            return encoder_out, None

        ph_dur_pred = self.dur_predictor(encoder_out, x_masks=txt_tokens == PAD_INDEX, infer=infer)

        return encoder_out, ph_dur_pred
