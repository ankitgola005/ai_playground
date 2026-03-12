import torch.nn as nn
from ai_playground.distributed.base import Parallel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import DistributedConfigProtocol


class SingleDevice(Parallel):
    def __init__(self, config: DistributedConfigProtocol):
        super().__init__(config)
        if self.world_size != 1:
            raise ValueError("SingleDevice strategy requires world_size=1.")

    def wrap_model(self, model: nn.Module, stage: str = "train") -> nn.Module:
        return model.to(self.device)
