from modules.diffusion.ddpm import GaussianDiffusion


class GaussianDiffusionOnnx(GaussianDiffusion):
    # noinspection PyMethodOverriding
    def forward(self, condition, speedup):
        pass
