from typing import List

import torch
from torch import Tensor

from modules.diffusion.ddpm import GaussianDiffusion


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


# noinspection PyMethodOverriding
class GaussianDiffusionONNX(GaussianDiffusion):
    def p_sample(self, x, t, cond):
        x_pred = self.denoise_fn(x, t, cond)
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * x_pred
        )
        x_recon = torch.clamp(x_recon, min=-1., max=1.)

        model_mean = (
                extract(self.posterior_mean_coef1, t) * x_recon +
                extract(self.posterior_mean_coef2, t) * x
        )
        model_log_variance = extract(self.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def plms_get_x_pred(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    def p_sample_plms(self, x_prev, t, interval, cond, noise_list: List[Tensor], stage: int):
        noise_pred = self.denoise_fn(x_prev, t, cond)
        t_prev = t - interval
        t_prev = t_prev * (t_prev > 0)
        if stage == 0:
            x_pred = self.plms_get_x_pred(x_prev, noise_pred, t, t_prev)
            noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2.
        elif stage == 1:
            noise_pred_prime = (3. * noise_pred - noise_list[-1]) / 2.
        elif stage == 2:
            noise_pred_prime = (23. * noise_pred - 16. * noise_list[-1] + 5. * noise_list[-2]) / 12.
        else:
            noise_pred_prime = (55. * noise_pred - 59. * noise_list[-1] + 37.
                                * noise_list[-2] - 9. * noise_list[-3]) / 24.
        x_prev = self.plms_get_x_pred(x_prev, noise_pred_prime, t, t_prev)
        return noise_pred, x_prev

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        return x * d + m

    def forward(self, condition, speedup):
        condition = condition.transpose(1, 2)  # [1, T, H] => [1, H, T]
        device = condition.device
        n_frames = condition.shape[2]

        step_range = torch.arange(0, self.k_step, speedup, dtype=torch.long, device=device).flip(0)[:, None]
        x = torch.randn((1, 1, self.out_dims, n_frames), device=device)

        if speedup > 1:
            plms_noise_stage: int = 0
            noise_list: List[Tensor] = []
            for t in step_range:
                noise_pred, x = self.p_sample_plms(
                    x, t, interval=speedup, cond=condition,
                    noise_list=noise_list, stage=plms_noise_stage
                )
                if plms_noise_stage == 0:
                    noise_list = [noise_pred]
                    plms_noise_stage = plms_noise_stage + 1
                else:
                    if plms_noise_stage >= 3:
                        noise_list.pop(0)
                    else:
                        plms_noise_stage = plms_noise_stage + 1
                    noise_list.append(noise_pred)
        else:
            for t in step_range:
                x = self.p_sample(x, t, cond=condition)

        x = x.squeeze(1).permute(0, 2, 1)  # [B, T, M]
        x = self.denorm_spec(x)
        return x
