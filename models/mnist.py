from ai_playground.configs.config import Config

import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dims = (
            [config.model.model_kwargs["input_dims"]]
            + config.model.model_kwargs["hidden_dims"]
            + config.model.model_kwargs["output_dims"]
        )
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if config.model.model_kwargs["dropout"] > 0:
                    layers.append(nn.Dropout(config.model.model_kwargs["dropout"]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
