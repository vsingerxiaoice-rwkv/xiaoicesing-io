import math

import torch


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def generate_pe(self, length, device):
        """Generate the positional encodings."""
        position = torch.arange(0, length, 1., device=device).unsqueeze(1)
        if self.reverse:
            position = position.flip(0)
        div_term = torch.exp(
            torch.arange(0., self.d_model, 2., device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.stack([
            torch.sin(position * div_term),
            torch.cos(position * div_term)
        ], dim=2).view(-1, self.d_model).unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        x = x * self.xscale + self.generate_pe(x.size(1), x.device)
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, d_model, dropout_rate):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        x = x + self.alpha * self.generate_pe(x.size(1), x.device)
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, d_model, dropout_rate):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.generate_pe(x.size(1), x.device)
        return self.dropout(x) + self.dropout(pos_emb)
