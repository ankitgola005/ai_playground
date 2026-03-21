from typing import TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from typing import List
    from ai_playground.configs import ModelConfig


class MNIST(nn.Module):
    """
    Fully connected feedforward network for MNIST classification.

    Model architecture is configurable via ConfigProtocol:
    - input_dims: int, input size (flattened image)
    - hidden_dims: List[int], sizes of hidden layers
    - output_dims: int, number of output classes
    - dropout: float, dropout probability (0 = no dropout)
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the MNIST model.

        Args:
            config (ConfigProtocol): Configuration object containing model hyperparameters.
        """
        super().__init__()

        model_kwargs = model_config.model_kwargs
        dims = (
            [model_kwargs["input_dims"]]
            + model_kwargs["hidden_dims"]
            + [model_kwargs["output_dims"]]
        )

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Apply ReLU and Dropout to hidden layers only
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if model_kwargs.get("dropout", 0) > 0:
                    layers.append(nn.Dropout(model_kwargs["dropout"]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, output_dims).
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.net(x)
