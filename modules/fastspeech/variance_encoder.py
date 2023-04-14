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
    def forward_embedding(self, txt_tokens, midi_embed, word_dur_embed):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embed + word_dur_embed
        if hparams['use_pos_embed']:
            if hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embed, word_dur_embed):
        """
        :param txt_tokens: [B, T]
        :param midi_embed: [B, T, H]
        :param word_dur_embed: [B, T, H]
        :return: [T x B x H]
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, midi_embed, word_dur_embed)  # [B, T, H]
        x = super()._forward(x, encoder_padding_mask)
        return x


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.midi_embed = Embedding(128, hparams['hidden_size'], PAD_INDEX)
        self.word_dur_embed = Linear(1, hparams['hidden_size'])
        self.dur_log_offset = hparams['dur_log_offset']

        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.encoder = FastSpeech2VarianceEncoder(
            self.txt_embed, hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads']
        )

        predictor_hidden = hparams['dur_predictor_hidden'] \
            if hparams['dur_predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            in_dims=hparams['hidden_size'],
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['dur_predictor_dropout'],
            padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'],
            offset=hparams['dur_log_offset']
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
        midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
        if word_dur is None or not infer:
            b = txt_tokens.shape[0]
            word_dur = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
                1, ph2word, ph_dur
            )[:, 1:]  # [B, T_ph] => [B, T_w]
        word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 1, ph2word)  # [B, T_w] => [B, T_ph]
        word_dur_embed = self.word_dur_embed(torch.log(word_dur.float() + self.dur_log_offset))
        encoder_out = self.encoder(txt_tokens, midi_embed, word_dur_embed)

        if not hparams['predict_dur']:
            return encoder_out, None

        if infer:
            ph_dur, _ = self.dur_predictor(encoder_out, x_mask=txt_tokens == 0, infer=True)
            return encoder_out, ph_dur
        else:
            ph_dur_xs = self.dur_predictor(encoder_out, x_mask=txt_tokens == 0, infer=False)
            return encoder_out, ph_dur_xs


class DummyPitchPredictor(nn.Module):
    def __init__(self, vmin, vmax, num_bins, deviation):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.interval = (vmax - vmin) / (num_bins - 1)  # align with centers of bins
        self.sigma = deviation / self.interval
        self.register_buffer('x', torch.arange(num_bins).float().reshape(1, 1, -1))  # [1, 1, N]

        self.base_pitch_embed = Linear(1, hparams['hidden_size'])
        self.net = nn.Sequential(
            Linear(hparams['hidden_size'], hparams['pitch_predictor_hidden']),
            Linear(hparams['pitch_predictor_hidden'], hparams['num_pitch_bins'])
        )

    def bins_to_values(self, bins):
        return bins * self.interval + self.vmin

    def out2pitch(self, probs):
        logits = probs.sigmoid()  # [B, T, N]
        bins = torch.sum(self.x * logits, dim=2) / torch.sum(logits, dim=2)  # [B, T]
        return self.bins_to_values(bins)

    def forward(self, condition, base_pitch):
        """
        :param condition: [B, T, H]
        :param base_pitch: [B, T]
        :return: pitch_pred [B, T], probs [B, T, N]
        """
        condition = condition + self.base_pitch_embed(base_pitch[:, :, None])
        probs = self.net(condition)
        return self.out2pitch(probs) + base_pitch, probs
