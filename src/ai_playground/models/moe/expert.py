import torch
from torch import nn


class Expert(nn.Module):
    """
    Feedforward expert used in Mixture-of-Experts (MoE) layers.
    This module represents a single expert, implemented as a standard
    2-layer MLP with non-linearity and dropout.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initialize Expert.

        Args:
            d_model (int): Input and output dimensionality.
            d_ff (int): Hidden layer dimensionality.
            dropout (float): Dropout probability applied after activation.

        Attributes:
            net (nn.Sequential): Feedforward network:
                Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (..., D),
                where D = d_model. Typically comes from a subset
                of tokens routed to this expert.

        Returns:
            torch.Tensor: Output tensor of shape (..., D), same as input.
        """
        return self.net(x)
