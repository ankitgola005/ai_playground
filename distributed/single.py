import torch
import torch.nn as nn
from .base import Parallel

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


class SingleDevice(Parallel):
    def __init__(self, device="cpu", num_devices: int = 1):
        if num_devices != 1:
            raise ValueError(
                "SingleDevice can only handle one device. Set num_devices=1."
            )
        super().__init__(device=device, num_devices=1)

    def setup(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but 'cuda' device was specified.")
        if self.device == "cuda":
            torch.cuda.set_device(self.device)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def setup_training(
        self, model: nn.Module, optimizer: Optional[Optimizer], dataloader: DataLoader
    ) -> tuple[nn.Module, Optional[Optimizer], DataLoader]:
        self.setup()
        model = self.wrap_model(model)
        if optimizer is not None:
            optimizer = self.setup_optimizer(optimizer, model)

        if dataloader is not None:
            dataloader = self.prepare_dataloader(dataloader)

        return model, optimizer, dataloader
