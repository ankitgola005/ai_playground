from typing import Optional, TYPE_CHECKING

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from ai_playground.distributed.base import Parallel
from ai_playground.distributed.training_worker import training_worker


class DDParallel(Parallel):
    def __init__(self, device: str = "cpu", num_devices: int = 1):
        super().__init__(device, num_devices)
        self.rank = 0
        self.world_size = num_devices
        self.backend = "gloo"
        self._sampler = None

    def setup(self):
        pass

    def launch(self, trainer_fn, *args, **kwargs):

        mp.spawn(
            self._worker_entry,
            args=(self, trainer_fn, args, kwargs),
            nprocs=self.world_size,
            join=True,
        )

    @staticmethod
    def _worker_entry(rank, strategy, trainer_fn, args, kwargs):
        strategy.rank = rank
        dist.init_process_group(
            backend=strategy.backend,
            rank=rank,
            world_size=strategy.world_size,
        )

        trainer_fn(*args, **kwargs)

        strategy.cleanup()

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setup_training(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        dataloader: DataLoader,
    ):

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

            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
            )

        return model, optimizer, dataloader

    def set_epoch(self, epoch: int):
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

    def wrap_model(self, model: nn.Module):

        model = model.to(self.device)

        if self.is_distributed():
            model = DDP(model, device_ids=[self.rank])

        return model

    def unwrap_model(self, model):

        if isinstance(model, DDP):
            return model.module

        return model