import torch.nn as nn
from ai_playground.distributed.base import Parallel


class SingleDevice(Parallel):
    def __init__(self, config):
        super().__init__(config)
        if self.world_size != 1:
            raise ValueError("SingleDevice strategy requires world_size=1.")

    def setup_environment(self):
        pass

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)
