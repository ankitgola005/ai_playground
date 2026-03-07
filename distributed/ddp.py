from typing import Optional, TYPE_CHECKING

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from .base import Parallel
from .training_worker import training_worker


class DDParallel(Parallel):
    def __init__(self, device: str = "cuda", num_devices: int = 1, rank: int = 0):
        super().__init__(device, num_devices)
        self.rank = rank
        self.world_size = num_devices

    def setup(self):
        pass

    def launch(self, trainer_fn, *args, **kwargs):
        mp.spawn(
            training_worker,
            args=(trainer_fn, args, kwargs),
            nprocs=self.world_size,
            join=True,
        )
    
    @staticmethod
    def _worker_entry(rank, strategy, trainer_fn, args, kwargs):
        # torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=strategy.backend,
            rank=strategy.rank,
            world_size=strategy.world_size,
        )

        trainer_fn(*args, **kwargs)
        strategy.cleanup()

    def setup_training(
        self, model: nn.Module, optimizer: Optimizer | None, dataloader: DataLoader
    ) -> tuple[Module, Optimizer | None, DataLoader]:
        model = self.wrap_model(model)

        if optimizer is not None:
            optimizer = self.setup_optimizer(optimizer, model)

        if self.is_distributed():
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self._sampler = sampler
            dataloader.sampler = sampler
        return model, optimizer, dataloader

    def prepare_dataloader(self, dataloader):
        if self.is_distributed() and hasattr(self, "_sampler"):
            self._sampler.set_epoch(self.rank)  # can be overridden per epoch in trainer
        return dataloader

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.is_distributed():
            model = DDP(model)
        return model
    
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module
        return model
