from typing import TYPE_CHECKING

from ai_playground.distributed import Parallel

if TYPE_CHECKING:
    import torch.nn as nn
    from ai_playground.configs.config import DistributedConfigProtocol


class SingleDevice(Parallel):
    """
    Single-device training strategy.
    """

    def __init__(self, config: "DistributedConfigProtocol"):
        """
        Initialize SingleDevice strategy.

        Args:
            config (DistributedConfigProtocol): Configuration object containing
                device and world_size information.

        Raises:
            ValueError: If world_size is not 1.
        """
        super().__init__(config)
        if self.world_size != 1:
            raise ValueError("SingleDevice strategy requires world_size=1.")

    def setup_environment(self, stage: str = "train") -> None:
        """
        Setup environment. No-op for single device.

        Args:
            stage (str): Stage of training or evaluation.
        """
        pass

    def wrap_model(self, model: nn.Module, stage: str = "train") -> nn.Module:
        """
        Wrap model if required. Move model to the selected device.

        Args:
            model (nn.Module): Model to train.
            stage (str): Stage of training (ignored for SingleDevice).

        Returns:
            nn.Module: Model moved to device.
        """
        return model.to(self.device)
