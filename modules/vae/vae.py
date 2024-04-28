from typing import Optional

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0

    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1,)  # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dropout(x)

        x = residual + self.drop_path(x)
        return x


class ConvNeXtModule(nn.Module):
    def __init__(self, dim, lays, intermediate_dim=None, layer_scale_init_value=1e-6, drop_out=0.):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = dim * 4

        self.layers = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value, drop_out) for _ in range(lays)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


class ConvNeXtModel(nn.Module):
    def __init__(self, in_dim, dims=None, lays=None, intermediate_dim=None, layer_scale_init_value=1e-6, drop_out=0.):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        if lays is None:
            lays = [3, 6, 2, 1]
        dims = [in_dim] + dims
        self.num = len(lays)
        self.encoder = nn.ModuleList()
        self.dim_proj = nn.ModuleList()
        for i in range(len(lays)):
            self.dim_proj.append(nn.Conv1d(dims[i], dims[i + 1],1))
            self.encoder.append(
                ConvNeXtModule(dims[i + 1], lays[i], intermediate_dim, layer_scale_init_value, drop_out))

    def forward(self, x,mask=None):
        for i in range(self.num):
            x = self.dim_proj[i](x)
            if mask is not None:
                x=x*mask
            x = self.encoder[i](x)
            if mask is not None:
                x=x*mask

        return x


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, dims=None, lays=None, intermediate_dim=None, layer_scale_init_value=1e-6,
                 drop_out=0.):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        if lays is None:
            lays = [3, 6, 2, 1]
        self.enc_out_proj = nn.Conv1d(dims[-1], latent_dim * 2,1)
        self.encoder = ConvNeXtModel(in_dim, dims, lays, intermediate_dim, layer_scale_init_value, drop_out)
        self.decoder = ConvNeXtModel(latent_dim, list(reversed(dims)), list(reversed(lays)), intermediate_dim,
                                     layer_scale_init_value, drop_out)
        self.dec_out_proj = nn.Conv1d(dims[0], in_dim,1)

    def reparameterize(self, mean, log):
        std = torch.exp(0.5 * log)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    def enc(self,x):
        x = self.encoder(x)
        mean, log= self.enc_out_proj(x).chunk(2, dim=1)
        z = self.reparameterize(mean, log)
        return z
    def dec(self,z):
        x = self.decoder(z)
        x = self.dec_out_proj(x)
        return x

    def forward(self, x,mask=None):
        x = self.encoder(x)
        mean, log= self.enc_out_proj(x).chunk(2, dim=1)
        z = self.reparameterize(mean, log)
        if mask is not None:
            z = z * mask
        x_r = self.decoder(z)
        x_r = self.dec_out_proj(x_r)
        if mask is not None:
            x_r = x_r * mask
        return x_r, mean, log

if __name__ == '__main__':
    m=VAE(1,32)
    x1=torch.randn(1,1,128)
    x2=m(x1)
    print(x2[0].shape)
