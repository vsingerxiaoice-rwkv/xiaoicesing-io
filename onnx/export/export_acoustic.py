import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONPATH'] = f'"{root_dir}"'
sys.path.insert(0, root_dir)

import argparse
import math
import re
import struct
from functools import partial

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from modules.commons.common_layers import Mish
from modules.naive_frontend.encoder import Encoder
from src.diff.net import AttrDict
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    return f0_coarse


class LengthRegulator(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur):
        token_idx = torch.arange(1, dur.shape[1] + 1, device=dur.device)[None, :, None]
        dur_cumsum = torch.cumsum(dur, dim=1)
        dur_cumsum_prev = F.pad(dur_cumsum, (1, -1), mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(dim=1).max(), device=dur.device)[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask).sum(dim=1)
        return mel2ph


class FastSpeech2MIDILess(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.lr = LengthRegulator()
        self.txt_embed = nn.Embedding(len(dictionary), hparams['hidden_size'], dictionary.pad())
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = Encoder(self.txt_embed, hparams['hidden_size'], hparams['enc_layers'],
                               hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads'])
        self.pitch_embed = nn.Embedding(300, hparams['hidden_size'], dictionary.pad())

    def forward(self, tokens, durations, f0):
        durations *= tokens > 0
        mel2ph = self.lr.forward(durations)
        f0 *= mel2ph > 0
        mel2ph = mel2ph[..., None].repeat((1, 1, hparams['hidden_size']))
        dur_embed = self.dur_embed(durations.float()[:, :, None])
        encoded = self.encoder(tokens, dur_embed)
        encoded = F.pad(encoded, (0, 0, 1, 0))
        encoded = torch.gather(encoded, 1, mel2ph)
        pitch = f0_to_coarse(f0)
        pitch_embed = self.pitch_embed(pitch)
        condition = encoded + pitch_embed
        return condition


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * torch.tensor(-emb)).unsqueeze(0))

    def forward(self, x):
        emb = self.emb * x
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.input_projection = nn.Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = nn.Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = nn.Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        # Avoid ConstantOfShape op
        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        # noinspection PyTypeChecker
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip += skip_connection

        x = skip / math.sqrt(len(self.residual_layers))

        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x.unsqueeze(1)


class NaiveNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('clip_min', to_torch(-1.))
        self.register_buffer('clip_max', to_torch(1.))

    def forward(self, x, noise_pred, t):
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * noise_pred
        )
        x_recon = torch.clamp(x_recon, min=self.clip_min, max=self.clip_max)

        model_mean = (
                extract(self.posterior_mean_coef1, t) * x_recon +
                extract(self.posterior_mean_coef2, t) * x
        )
        model_log_variance = extract(self.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


class PLMSNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Below are buffers for TorchScript to pass jit compilation.
        self.register_buffer('_1', to_torch(1))
        self.register_buffer('_2', to_torch(2))
        self.register_buffer('_3', to_torch(3))
        self.register_buffer('_5', to_torch(5))
        self.register_buffer('_9', to_torch(9))
        self.register_buffer('_12', to_torch(12))
        self.register_buffer('_16', to_torch(16))
        self.register_buffer('_23', to_torch(23))
        self.register_buffer('_24', to_torch(24))
        self.register_buffer('_37', to_torch(37))
        self.register_buffer('_55', to_torch(55))
        self.register_buffer('_59', to_torch(59))

    def forward(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * ((self._1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - self._1 / (
                a_t_sq * (((self._1 - a_prev) * a_t).sqrt() + ((self._1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    def predict_stage0(self, noise_pred, noise_pred_prev):
        return (noise_pred
                + noise_pred_prev) / self._2

    def predict_stage1(self, noise_pred, noise_list):
        return (noise_pred * self._3
                - noise_list[-1]) / self._2

    def predict_stage2(self, noise_pred, noise_list):
        return (noise_pred * self._23
                - noise_list[-1] * self._16
                + noise_list[-2] * self._5) / self._12

    def predict_stage3(self, noise_pred, noise_list):
        return (noise_pred * self._55
                - noise_list[-1] * self._59
                + noise_list[-2] * self._37
                - noise_list[-3] * self._9) / self._24


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        return x * d + m


class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, timesteps=1000, k_step=1000, spec_min=None, spec_max=None):
        super().__init__()
        self.mel_bins = out_dims
        self.K_step = k_step

        self.denoise_fn = DiffNet(out_dims)
        self.naive_noise_predictor = NaiveNoisePredictor()
        self.plms_noise_predictor = PLMSNoisePredictor()
        self.mel_extractor = MelExtractor()

        if 'schedule_type' in hparams.keys():
            betas = beta_schedule[hparams['schedule_type']](timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        # Below are buffers for state_dict to load into.
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def build_submodules(self):
        # Move registered buffers into submodules after loading state dict.
        self.naive_noise_predictor.register_buffer('sqrt_recip_alphas_cumprod', self.sqrt_recip_alphas_cumprod)
        self.naive_noise_predictor.register_buffer('sqrt_recipm1_alphas_cumprod', self.sqrt_recipm1_alphas_cumprod)
        self.naive_noise_predictor.register_buffer(
            'posterior_log_variance_clipped', self.posterior_log_variance_clipped)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef1', self.posterior_mean_coef1)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef2', self.posterior_mean_coef2)
        self.plms_noise_predictor.register_buffer('alphas_cumprod', self.alphas_cumprod)
        self.mel_extractor.register_buffer('spec_min', self.spec_min)
        self.mel_extractor.register_buffer('spec_max', self.spec_max)
        del self.sqrt_recip_alphas_cumprod
        del self.sqrt_recipm1_alphas_cumprod
        del self.posterior_log_variance_clipped
        del self.posterior_mean_coef1
        del self.posterior_mean_coef2
        del self.alphas_cumprod
        del self.spec_min
        del self.spec_max

    def forward(self, condition, speedup):
        condition = condition.transpose(1, 2)  # (1, n_frames, 256) => (1, 256, n_frames)

        device = condition.device
        n_frames = condition.shape[2]
        step_range = torch.arange(0, self.K_step, speedup, dtype=torch.long, device=device).flip(0)
        x = torch.randn((1, 1, self.mel_bins, n_frames), device=device)

        if speedup > 1:
            plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
            noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
            for t in step_range:
                noise_pred = self.denoise_fn(x, t, condition)
                t_prev = t - speedup
                t_prev = t_prev * (t_prev > 0)

                if plms_noise_stage == 0:
                    x_pred = self.plms_noise_predictor(x, noise_pred, t, t_prev)
                    noise_pred_prev = self.denoise_fn(x_pred, t_prev, condition)
                    noise_pred_prime = self.plms_noise_predictor.predict_stage0(noise_pred, noise_pred_prev)
                elif plms_noise_stage == 1:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage1(noise_pred, noise_list)
                elif plms_noise_stage == 2:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage2(noise_pred, noise_list)
                else:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage3(noise_pred, noise_list)

                noise_pred = noise_pred.unsqueeze(0)
                if plms_noise_stage < 3:
                    noise_list = torch.cat((noise_list, noise_pred), dim=0)
                    plms_noise_stage = plms_noise_stage + 1
                else:
                    noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)

                x = self.plms_noise_predictor(x, noise_pred_prime, t, t_prev)

            # from dpm_solver import NoiseScheduleVP, model_wrapper, DpmSolver
            # ## 1. Define the noise schedule.
            # noise_schedule = NoiseScheduleVP(betas=self.betas)
            #
            # ## 2. Convert your discrete-time `model` to the continuous-time
            # # noise prediction model. Here is an example for a diffusion model
            # ## `model` with the noise prediction type ("noise") .
            #
            # model_fn = model_wrapper(
            #     self.denoise_fn,
            #     noise_schedule,
            #     model_kwargs={"cond": condition}
            # )
            #
            # ## 3. Define dpm-solver and sample by singlestep DPM-Solver.
            # ## (We recommend singlestep DPM-Solver for unconditional sampling)
            # ## You can adjust the `steps` to balance the computation
            # ## costs and the sample quality.
            # dpm_solver = DpmSolver(model_fn, noise_schedule)
            #
            # steps = t // hparams["pndm_speedup"]
            # x = dpm_solver.sample(x, steps=steps)
        else:
            for t in step_range:
                pred = self.denoise_fn(x, t, condition)
                x = self.naive_noise_predictor(x, pred, t)

        mel = self.mel_extractor(x)
        return mel


def build_fs2_model(device):
    model = FastSpeech2MIDILess(
        dictionary=TokenTextEncoder(vocab_list=build_phoneme_list())
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model.fs2', strict=True)
    model.to(device)
    return model


def build_diff_model(device):
    model = GaussianDiffusion(
        out_dims=hparams['audio_num_mel_bins'],
        timesteps=hparams['timesteps'],
        k_step=hparams['K_step'],
        spec_min=hparams['spec_min'],
        spec_max=hparams['spec_max'],
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model', strict=False)
    model.build_submodules()
    model.to(device)
    return model


class ModuleWrapper(nn.Module):
    def __init__(self, model, name='model'):
        super().__init__()
        self.wrapped_name = name
        setattr(self, name, model)

    def forward(self, *args, **kwargs):
        return getattr(self, self.wrapped_name)(*args, **kwargs)


class FastSpeech2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = ModuleWrapper(model, name='fs2')

    def forward(self, tokens, durations, f0):
        return self.model(tokens, durations, f0)


class DiffusionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, condition, speedup):
        return self.model(condition, speedup)


def _fix_cast_nodes(graph, logs=None):
    if logs is None:
        logs = []
    for sub_node in graph.node:
        if sub_node.op_type == 'If':
            for attr in sub_node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                _fix_cast_nodes(branch, logs)
        elif sub_node.op_type == 'Loop':
            for attr in sub_node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    _fix_cast_nodes(body, logs)
        elif sub_node.op_type == 'Cast':
            for i, sub_attr in enumerate(sub_node.attribute):
                if sub_attr.name == 'to':
                    to = onnx.helper.get_attribute_value(sub_attr)
                    if to == onnx.TensorProto.DOUBLE:
                        float32 = onnx.helper.make_attribute('to', onnx.TensorProto.FLOAT)
                        sub_node.attribute.remove(sub_attr)
                        sub_node.attribute.insert(i, float32)
                        logs.append(sub_node.name)
                        break
    return logs


def _fold_shape_gather_equal_if_to_squeeze(graph, subgraph, logs=None):
    if logs is None:
        logs = []

    # Do folding in sub-graphs recursively.
    for node in subgraph.node:
        if node.op_type == 'If':
            for attr in node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                _fold_shape_gather_equal_if_to_squeeze(graph, branch, logs)
        elif node.op_type == 'Loop':
            for attr in node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    _fold_shape_gather_equal_if_to_squeeze(graph, body, logs)

    # Do folding in current graph.
    i_shape = 0
    while i_shape < len(subgraph.node):
        if subgraph.node[i_shape].op_type == 'Shape':
            shape_node = subgraph.node[i_shape]
            shape_out = shape_node.output[0]
            i_gather = i_shape + 1
            while i_gather < len(subgraph.node):
                if subgraph.node[i_gather].op_type == 'Gather' and subgraph.node[i_gather].input[0] == shape_out:
                    gather_node = subgraph.node[i_gather]
                    gather_out = gather_node.output[0]
                    i_equal = i_gather + 1
                    while i_equal < len(subgraph.node):
                        if subgraph.node[i_equal].op_type == 'Equal' and (
                                subgraph.node[i_equal].input[0] == gather_out
                                or subgraph.node[i_equal].input[1] == gather_out):
                            equal_node = subgraph.node[i_equal]
                            equal_out = equal_node.output[0]
                            i_if = i_equal + 1
                            while i_if < len(subgraph.node):
                                if subgraph.node[i_if].op_type == 'If' and subgraph.node[i_if].input[0] == equal_out:
                                    # Found the substructure to be folded.
                                    if_node = subgraph.node[i_if]
                                    # Search and clean initializer values.
                                    squeeze_axes_tensor = None
                                    for tensor in subgraph.initializer:
                                        if tensor.name == gather_node.input[1]:
                                            squeeze_axes_tensor = tensor
                                            subgraph.initializer.remove(tensor)
                                        elif tensor.name == equal_node.input[1]:
                                            subgraph.initializer.remove(tensor)
                                    # Create 'Squeeze' node.
                                    squeeze_node = onnx.helper.make_node(
                                        op_type='Squeeze',
                                        inputs=shape_node.input,
                                        outputs=if_node.output
                                    )
                                    squeeze_axes = onnx.helper.make_attribute(
                                        key='axes',
                                        value=[struct.unpack('q', squeeze_axes_tensor.raw_data)[0]]  # unpack int64
                                    )
                                    squeeze_node.attribute.extend([squeeze_axes])
                                    # Replace 'Shape', 'Gather', 'Equal', 'If' with 'Squeeze'.
                                    subgraph.node.insert(i_shape, squeeze_node)
                                    subgraph.node.remove(shape_node)
                                    subgraph.node.remove(gather_node)
                                    subgraph.node.remove(equal_node)
                                    subgraph.node.remove(if_node)
                                    logs.append((shape_node.name, gather_node.name, equal_node.name, if_node.name))
                                    break
                                i_if += 1
                            else:
                                break
                        i_equal += 1
                    else:
                        break
                i_gather += 1
            else:
                break
        i_shape += 1
    return logs


def _extract_conv_nodes(graph, weight_pattern, alias_prefix):
    node_dict = {}  # key: pattern match, value: (alias, node)
    logs = []

    def _extract_conv_nodes_recursive(subgraph):
        to_be_removed = []
        for sub_node in subgraph.node:
            if sub_node.op_type == 'If':
                for attr in sub_node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _extract_conv_nodes_recursive(branch)
            elif sub_node.op_type == 'Loop':
                for attr in sub_node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _extract_conv_nodes_recursive(body)
            elif sub_node.op_type == 'Conv' and re.match(weight_pattern, sub_node.input[1]):
                # Found node to extract
                cached = node_dict.get(sub_node.input[1])
                if cached is None:
                    out_alias = f'{alias_prefix}.{len(node_dict)}'
                    node_dict[sub_node.input[1]] = (out_alias, sub_node)
                else:
                    out_alias = cached[0]
                out = sub_node.output[0]
                # Search for nodes downstream the extracted node and match them to the renamed output
                for dep_node in subgraph.node:
                    for dep_idx, dep_input in enumerate(dep_node.input):
                        if dep_input == out:
                            dep_node.input.remove(out)
                            dep_node.input.insert(dep_idx, out_alias)
                # Add the node to the remove list
                to_be_removed.append(sub_node)
                logs.append(sub_node.name)
        [subgraph.node.remove(_n) for _n in to_be_removed]

    for i, n in enumerate(graph.node):
        if n.op_type == 'If':
            for a in n.attribute:
                b = onnx.helper.get_attribute_value(a)
                _extract_conv_nodes_recursive(b)
            for key in reversed(node_dict):
                alias, node = node_dict[key]
                # Rename output of the node
                out_name = node.output[0]
                node.output.remove(node.output[0])
                node.output.insert(0, alias)
                # Insert node into the main graph
                graph.node.insert(i, node)
                # Rename value info of the output
                for v in graph.value_info:
                    if v.name == out_name:
                        v.name = alias
                        break
            break
    return logs


def _remove_unused_values(graph):
    used_values = set()
    cleaned_values = []

    def _record_usage_recursive(subgraph):
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, do recording recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _record_usage_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _record_usage_recursive(body)
            # For each node, record its inputs and outputs
            for input_value in node.input:
                used_values.add(input_value)
            for output_value in node.output:
                used_values.add(output_value)

    def _clean_unused_recursively(subgraph):
        # Do cleaning in sub-graphs recursively.
        for node in subgraph.node:
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _clean_unused_recursively(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _clean_unused_recursively(body)

        # Do cleaning in current graph.
        i = 0
        while i < len(subgraph.initializer):
            if subgraph.initializer[i].name not in used_values:
                cleaned_values.append(subgraph.initializer[i].name)
                subgraph.initializer.remove(subgraph.initializer[i])
            else:
                i += 1
        i = 0
        while i < len(subgraph.value_info):
            if subgraph.value_info[i].name not in used_values:
                cleaned_values.append(subgraph.value_info[i].name)
                subgraph.value_info.remove(subgraph.value_info[i])
            else:
                i += 1

    _record_usage_recursive(graph)
    _clean_unused_recursively(graph)
    return cleaned_values


def _add_prefixes(model,
                  initializer_prefix=None,
                  value_info_prefix=None,
                  node_prefix=None,
                  dim_prefix=None,
                  ignored_pattern=None):
    initializers = set()
    value_infos = set()

    def _record_initializers_and_value_infos_recursive(subgraph):
        # Record names in current graph
        for initializer in subgraph.initializer:
            if re.match(ignored_pattern, initializer.name):
                continue
            initializers.add(initializer.name)
        for value_info in subgraph.value_info:
            if re.match(ignored_pattern, value_info.name):
                continue
            value_infos.add(value_info.name)
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, do recording recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _record_initializers_and_value_infos_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _record_initializers_and_value_infos_recursive(body)

    def _add_prefixes_recursive(subgraph):
        # Add prefixes in current graph
        if initializer_prefix is not None:
            for initializer in subgraph.initializer:
                if re.match(ignored_pattern, initializer.name):
                    continue
                initializer.name = initializer_prefix + initializer.name
        for value_info in subgraph.value_info:
            if dim_prefix is not None:
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_param is None or dim.dim_param == '' or re.match(ignored_pattern, dim.dim_param):
                        continue
                    dim.dim_param = dim_prefix + dim.dim_param
            if value_info_prefix is None or re.match(ignored_pattern, value_info.name):
                continue
            value_info.name = value_info_prefix + value_info.name
        if node_prefix is not None:
            for node in subgraph.node:
                if re.match(ignored_pattern, node.name):
                    continue
                node.name = node_prefix + node.name
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, rename recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _add_prefixes_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _add_prefixes_recursive(body)
            # For each node, rename its inputs and outputs
            for i, input_value in enumerate(node.input):
                if input_value in initializers and initializer_prefix is not None:
                    node.input.remove(input_value)
                    node.input.insert(i, initializer_prefix + input_value)
                if input_value in value_infos and value_info_prefix is not None:
                    node.input.remove(input_value)
                    node.input.insert(i, value_info_prefix + input_value)
            for i, output_value in enumerate(node.output):
                if output_value in initializers and initializer_prefix is not None:
                    node.output.remove(output_value)
                    node.output.insert(i, initializer_prefix + output_value)
                if output_value in value_infos and value_info_prefix is not None:
                    node.output.remove(output_value)
                    node.output.insert(i, value_info_prefix + output_value)

    _record_initializers_and_value_infos_recursive(model.graph)
    _add_prefixes_recursive(model.graph)


def fix(src, target):
    model = onnx.load(src)

    # The output dimension are wrongly hinted by TorchScript
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    out_dims.remove(out_dims[1])
    out_dims.insert(1, in_dims[1])
    print(f'| annotate output: \'{model.graph.output[0].name}\'')

    # Fix 'Cast' nodes in sub-graphs that wrongly cast tensors to float64
    fixed_casts = _fix_cast_nodes(model.graph)
    print('| fix node(s): ')
    for i, log in enumerate(fixed_casts):
        if i == len(fixed_casts) - 1:
            end = '\n'
        elif i % 10 == 9:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Run #1 of the simplifier to fix missing value info and type hints and remove unnecessary 'Cast'.
    print('Running ONNX simplifier...')
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim

    then_branch = None
    for node in model.graph.node:
        if node.op_type == 'If':
            # Add type hint to let the simplifier fold 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze'
            if_out = node.output[0]
            for info in model.graph.value_info:
                if info.name == if_out:
                    if_out_dim = info.type.tensor_type.shape.dim
                    while len(if_out_dim) > 0:
                        if_out_dim.remove(if_out_dim[0])
                    if_out_dim.insert(0, in_dims[0])  # batch_size
                    if_out_dim.insert(1, in_dims[0])  # 1
                    if_out_dim.insert(2, out_dims[2])  # mel_bins
                    if_out_dim.insert(3, in_dims[1])  # n_frames
                    print(f'| annotate node: \'{node.name}\'')

            # Manually fold 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze' in sub-graphs
            folded_groups = []
            for attr in node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                folded_groups += _fold_shape_gather_equal_if_to_squeeze(model.graph, branch)
                if attr.name == 'then_branch':
                    # Save branch for future use
                    then_branch = branch
            print('| fold node group(s): ')
            print(', '.join(['[' + ', '.join([f'\'{n}\'' for n in log]) + ']' for log in folded_groups]))
            break

    # Optimize 'Concat' nodes for shapes
    concat_node = None
    shape_prefix_name = 'noise.shape.prefix'
    list_length_name = 'list.length'
    for node in model.graph.node:
        if node.op_type == 'Concat':
            concat_node = node
            for i, ini in enumerate(model.graph.initializer):
                if ini.name == node.input[0]:
                    shape_prefix = onnx.helper.make_tensor(
                        name=shape_prefix_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=(3,),
                        vals=[out_dims[0].dim_value, 1, out_dims[2].dim_value]
                    )
                    list_length = onnx.helper.make_tensor(
                        name=list_length_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=(1,),
                        vals=[0]
                    )
                    model.graph.initializer.extend([shape_prefix, list_length])
                    break
            for i in range(3):
                node.input.remove(node.input[0])
            node.input.insert(0, shape_prefix_name)
            print(f'| optimize node: \'{node.name}\'')
            break
    for node in then_branch.node:
        if node.op_type == 'Concat':
            concat_inputs = list(node.input)
            dep_nodes = []
            for dep_node in then_branch.node:
                if dep_node.op_type == 'Unsqueeze' and dep_node.output[0] in concat_inputs:
                    dep_nodes.append(dep_node)
            [then_branch.node.remove(d_n) for d_n in dep_nodes]
            while len(node.input) > 0:
                node.input.remove(node.input[0])
            node.input.extend([list_length_name, concat_node.output[0]])
            print(f'| optimize node: \'{node.name}\'')
            break

    # Extract 'Conv' nodes and cache results of conditioner projection
    # of each residual layer from loop bodies to improve performance.
    extracted_convs = _extract_conv_nodes(
        model.graph,
        r'model\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
        'cache'
    )

    print(f'| extract node(s):')
    for i, log in enumerate(extracted_convs):
        if i == len(extracted_convs) - 1:
            end = '\n'
        elif i % 10 == 9:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Remove unused initializers and value infos
    cleaned_values = _remove_unused_values(model.graph)
    print(f'| clean value(s):')
    for i, log in enumerate(cleaned_values):
        if i == len(cleaned_values) - 1:
            end = '\n'
        elif i % 15 == 14:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Run #2 of the simplifier to further optimize the graph and reduce dangling sub-graphs.
    print('Running ONNX simplifier...')
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    onnx.save(model, target)
    print('Graph fixed and optimized.')


def export(fs2_path, diff_path):
    set_hparams(print_hparams=False)
    if hparams.get('use_midi', True):
        raise NotImplementedError('Only checkpoints of MIDI-less mode are supported.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fs2 = FastSpeech2Wrapper(
        model=build_fs2_model(device)
    )
    diffusion = DiffusionWrapper(
        model=build_diff_model(device)
    )

    with torch.no_grad():
        tokens = torch.tensor([[3]], dtype=torch.long, device=device)
        durations = torch.tensor([[1]], dtype=torch.long, device=device)
        f0 = torch.tensor([[440.]], dtype=torch.float32, device=device)
        print('Exporting FastSpeech2...')
        torch.onnx.export(
            fs2,
            (
                tokens,
                durations,
                f0
            ),
            fs2_path,
            input_names=[
                'tokens',
                'durations',
                'f0'
            ],
            output_names=[
                'condition'
            ],
            dynamic_axes={
                'tokens': {
                    1: 'n_tokens'
                },
                'durations': {
                    1: 'n_tokens'
                },
                'f0': {
                    1: 'n_frames'
                }
            },
            opset_version=11
        )
        model = onnx.load(fs2_path)
        model, check = onnxsim.simplify(model, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, fs2_path)

        n_frames = 10
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        noise_t = torch.randn(shape, device=device)
        noise_list = torch.randn((3, *shape), device=device)
        condition = torch.rand((1, hparams['hidden_size'], n_frames), device=device)
        step = (torch.rand((), device=device) * hparams['K_step']).long()
        speedup = (torch.rand((), device=device) * step / 10.).long()
        step_prev = torch.maximum(step - speedup, torch.tensor(0, dtype=torch.long, device=device))

        print('Tracing GaussianDiffusion submodules...')
        diffusion.model.denoise_fn = torch.jit.trace(
            diffusion.model.denoise_fn,
            (
                noise_t,
                step,
                condition
            )
        )
        diffusion.model.naive_noise_predictor = torch.jit.trace(
            diffusion.model.naive_noise_predictor,
            (
                noise_t,
                noise_t,
                step
            ),
            check_trace=False
        )
        diffusion.model.plms_noise_predictor = torch.jit.trace_module(
            diffusion.model.plms_noise_predictor,
            {
                'forward': (
                    noise_t,
                    noise_t,
                    step,
                    step_prev
                ),
                'predict_stage0': (
                    noise_t,
                    noise_t
                ),
                'predict_stage1': (
                    noise_t,
                    noise_list
                ),
                'predict_stage2': (
                    noise_t,
                    noise_list
                ),
                'predict_stage3': (
                    noise_t,
                    noise_list
                ),
            }
        )
        diffusion.model.mel_extractor = torch.jit.trace(
            diffusion.model.mel_extractor,
            (
                noise_t
            )
        )

        diffusion = torch.jit.script(diffusion)
        condition = torch.rand((1, n_frames, hparams['hidden_size']), device=device)
        speedup = torch.tensor(10, dtype=torch.long, device=device)
        dummy = diffusion.forward(condition, speedup)

        torch.onnx.export(
            diffusion,
            (
                condition,
                speedup
            ),
            diff_path,
            input_names=[
                'condition',
                'speedup'
            ],
            output_names=[
                'mel'
            ],
            dynamic_axes={
                'condition': {
                    1: 'n_frames'
                }
            },
            opset_version=11,
            example_outputs=(
                dummy
            )
        )
        print('PyTorch ONNX export finished.')


def merge(fs2_path, diff_path, target_path):
    fs2_model = onnx.load(fs2_path)
    diff_model = onnx.load(diff_path)

    # Add prefixes to names inside the model graph.
    print('Adding prefixes to models...')
    _add_prefixes(
        fs2_model, initializer_prefix='fs2.', value_info_prefix='fs2.',
        node_prefix='Enc_', ignored_pattern=r'model\.fs2\.'
    )
    _add_prefixes(
        fs2_model, dim_prefix='enc__', ignored_pattern=r'(n_tokens)|(n_frames)'
    )
    _add_prefixes(
        diff_model, initializer_prefix='diffusion.', value_info_prefix='diffusion.',
        node_prefix='Dec_', ignored_pattern=r'model.'
    )
    _add_prefixes(
        diff_model, dim_prefix='dec__', ignored_pattern='n_frames'
    )
    # Official onnx API does not consider sub-graphs.
    # onnx.compose.add_prefix(fs2_model, prefix='fs2.', inplace=True)
    # onnx.compose.add_prefix(diff_model, prefix='diffusion.', inplace=True)

    merged_model = onnx.compose.merge_models(
        fs2_model, diff_model, io_map=[('condition', 'condition')],
        prefix1='', prefix2='', name=fs2_model.graph.name, doc_string=''
    )
    print('FastSpeech2 and GaussianDiffusion models merged.')
    onnx.save(merged_model, target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DiffSinger acoustic model to ONNX')
    parser.add_argument('--exp', type=str, required=True, help='Experiment to export')
    parser.add_argument('--target', required=False, type=str, help='Path of the target ONNX model')
    args = parser.parse_args()

    cwd = os.getcwd()
    if args.target:
        target = os.path.join(cwd, args.target)
    else:
        target = None
    os.chdir(root_dir)
    exp = args.exp
    sys.argv = [
        'inference/ds_cascade.py',
        '--config',
        f'checkpoints/{exp}/config.yaml',
        '--exp_name',
        exp
    ]

    diff_model_path = f'onnx/assets/{exp}.onnx' if not target else target
    os.makedirs(f'onnx/assets/temp', exist_ok=True)
    fs2_model_path = f'onnx/assets/temp/fs2.onnx'
    export(fs2_path=fs2_model_path, diff_path=diff_model_path)
    fix(diff_model_path, diff_model_path)
    merge(fs2_path=fs2_model_path, diff_path=diff_model_path, target_path=diff_model_path)
    os.remove(fs2_model_path)

    os.chdir(cwd)
    if args.target:
        log_path = os.path.abspath(args.target)
    else:
        log_path = diff_model_path
    print(f'| export \'model\' to \'{log_path}\'.')
