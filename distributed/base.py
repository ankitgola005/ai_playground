from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import torch.nn as nn
import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.amp.grad_scaler import GradScaler
    from torch.utils.data import DataLoader


class Parallel(ABC):
    def __init__(self, device="cpu", num_devices: int = 1):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but 'cuda' device was specified.")

        self._device: torch.device = torch.device(device)
        self.rank: int = 0
        self.world_size: int = num_devices
        self.backend: Optional[str] = "nccl" if self.device_type == "cuda" else "gloo"

    def init_distributed(self, backend: str, rank: int, world_size: int):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )

        self.backend = backend
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def setup_training(
        self, model: nn.Module, optimizer: Optional[Optimizer], dataloader: DataLoader
    ) -> tuple[nn.Module, Optional[Optimizer], DataLoader]:
        pass

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    @property
    def device(self):
        return self._device

    @property
    def device_type(self):
        return self._device.type

    def is_distributed(self):
        return self.world_size > 1

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def setup_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        return optimizer

    def optimizer_step(self, optimizer: Optimizer, scaler: Optional[GradScaler] = None):
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def barrier(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def is_main_process(self):
        return self.rank == 0

    def launch(self, trainer_fn, *args, **kwargs):
        trainer_fn(*args, **kwargs)

    def prepare_dataloader(self, dataloader):
        return dataloader

    def cleanup(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
