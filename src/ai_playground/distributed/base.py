from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import torch.nn as nn
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
    from torch.amp.grad_scaler import GradScaler
    from ai_playground.configs.config import DistributedConfigProtocol


class Parallel(ABC):
    def __init__(self, config: DistributedConfigProtocol):
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but 'cuda' device was specified.")

        self._device: torch.device = torch.device(device)
        self.rank: int = 0
        self.world_size: int = config.world_size
        self.backend: str = "nccl" if self.device_type == "cuda" else "gloo"

    def init_distributed(self, rank: int, world_size: int):
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend, rank=rank, world_size=world_size
            )
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def setup_environment(self, stage: str = "train"):
        pass

    @abstractmethod
    def wrap_model(self, model: nn.Module, stage: str = "train") -> nn.Module:
        pass

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, device: torch.device):
        self._device = device

    @property
    def device_type(self) -> str:
        return self._device.type

    def is_distributed(self):
        return dist.is_initialized() and self.world_size > 1

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
        if dist.is_initialized():
            dist.barrier()

    def is_main_process(self) -> bool:
        return self.rank == 0

    def rank_zero_only(self, fn):
        def wrapper(*args, **kwargs):
            if self.is_main_process():
                return fn(*args, **kwargs)

        return wrapper

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not self.is_distributed():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce(tensor)
        tensor.div_(self.world_size)
        return tensor

    def launch(self, trainer_fn, *args, **kwargs):
        trainer_fn(*args, **kwargs)

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        return dataloader

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
