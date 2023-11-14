from typing import Optional
from .pcmer import PCmer
from modules.ddsp.core import frequency_filter, upsample
from modules.vocoders.registry import VOCODERS
from torch.nn.utils import weight_norm
from utils.hparams import hparams
import torch
import torch.nn as nn
import numpy as np


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))
    
    
class Cond2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits,
            num_channels=256,
            num_heads=8,
            num_layers=3, 
            kernel_size=31, 
            dropout_rate=0.0):
        super().__init__()
        self.output_splits = output_splits
        
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, num_channels, 3, 1, 1),
                nn.GroupNorm(4, num_channels),
                nn.LeakyReLU(),
                nn.Conv1d(num_channels, num_channels, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_model=num_channels,
            dim_keys=num_channels,
            dim_values=num_channels,
            kernel_size=kernel_size,
            residual_dropout=dropout_rate)
        self.norm = nn.LayerNorm(num_channels)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(num_channels, self.n_out))

    def forward(self, cond):
        x = self.stack(cond.transpose(1,2)).transpose(1,2)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        return controls 


class CombSub(torch.nn.Module):
    def __init__(self, 
            input_channel,
            n_mag_harmonic=512,
            n_mag_noise=256,
            num_channels=256,
            num_heads=8,
            num_layers=3, 
            kernel_size=31, 
            dropout_rate=0.0):
        super().__init__()

        # params
        self.sampling_rate = hparams['audio_sample_rate']
        self.block_size = hparams['hop_size']
        self.win_length = hparams['fft_size']
        self.register_buffer("window", torch.hann_window(self.win_length))
        
        # Cond2Control
        split_map = {
            'harmonic_phase': self.win_length // 2 + 1,
            'harmonic_magnitude': n_mag_harmonic, 
            'noise_magnitude': n_mag_noise
        }
        self.cond2ctrl = Cond2Control(input_channel, split_map, num_channels, num_heads, num_layers, kernel_size, dropout_rate)

    def forward(self, fs2_frames, f0_frames, initial_phase=None, infer=True, **kwargs):
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.cond2ctrl(fs2_frames)
        
        src_allpass = torch.exp(1.j * np.pi * ctrls['harmonic_phase'])
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        
        # combtooth exciter signal
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1) 
        
        # harmonic part filter (using dynamic-windowed LTV-FIR)
        harmonic = frequency_filter(
                        combtooth,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
               
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            harmonic,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
        
        # noise part filter (using constant-windowed LTV-FIR)
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        torch.complex(noise_param, torch.zeros_like(noise_param)),
                        hann_window = True)
                        
        signal = harmonic + noise

        return signal
        
class DDSPDecoder(nn.Module):
    def __init__(
            self, in_dims, out_dims, /, *,
            n_mag_harmonic=512, n_mag_noise=256,
            num_channels=256, num_heads=8, num_layers=3, kernel_size=31, dropout_rate=0.0
    ):
        super().__init__()
        self.wav_decoder = CombSub(in_dims, n_mag_harmonic, n_mag_noise, num_channels, num_heads, num_layers, kernel_size, dropout_rate)
        self.vocoder_class = VOCODERS[hparams['vocoder']]
        assert out_dims == hparams['audio_num_mel_bins']

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(hparams['spec_min'])[None, None, :].transpose(-3, -2)
        spec_max = torch.FloatTensor(hparams['spec_max'])[None, None, :].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)
    
    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return (x - b) / k
        
    # noinspection PyUnusedLocal
    def forward(self, cond_dict, infer=False):
        fs2_frames = cond_dict['fs2_cond']
        f0_frames = cond_dict['f0'].unsqueeze(-1)
        mask = cond_dict['mel2ph']
        ddsp_wav = self.wav_decoder(fs2_frames, f0_frames, infer=infer)
        ddsp_mel = self.vocoder_class.wav2spec_torch(ddsp_wav)
        output = self.norm_spec(ddsp_mel[:, :fs2_frames.shape[1], :])
        output *= ((mask > 0).float()[:, :, None])
        return output
