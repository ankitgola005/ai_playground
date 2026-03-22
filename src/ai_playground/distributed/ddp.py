from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from ai_playground.distributed import Parallel

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from ai_playground.configs.config import DistributedConfig
    from typing import Optional, Callable, Tuple


class DDParallel(Parallel):
    """
    DistributedDataParallel (DDP) strategy for multi-process training.
    Extends the base Parallel class.
    """

    def __init__(self, config: "DistributedConfig"):
        """
        Initialize DDParallel strategy.

        Args:
            config (DistributedConfigProtocol): Configuration object containing
                device and world_size information.
        """
        super().__init__(config)
        self._sampler: Optional[DistributedSampler] = None

    def setup_environment(self, stage: str = "train") -> None:
        """
        Placeholder for environment setup.
        """
        pass

    def launch(self, trainer_fn: Callable, *args, **kwargs) -> None:
        """
        Launch training across multiple processes using torch.multiprocessing.spawn.

        Args:
            trainer_fn (Callable): Training function to execute in each process.
        """
        if self.world_size == 1:
            trainer_fn(*args, **kwargs)
        else:
            mp.spawn(  # type: ignore
                self._worker_entry,
                args=(self, trainer_fn, args, kwargs),
                nprocs=self.world_size,
                join=True,
            )

    @staticmethod
    def _worker_entry(
        rank: int,
        strategy: "DDParallel",
        trainer_fn: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """
        Worker entry function for each spawned process.
        Initializes process group and executes the trainer.

        Args:
            rank (int): Process rank.
            strategy (DDParallel): DDP strategy instance.
            trainer_fn (Callable): Training function.
            args (tuple): Positional arguments for trainer_fn.
            kwargs (dict): Keyword arguments for trainer_fn.
        """
        strategy.rank = rank
        dist.init_process_group(
            backend=strategy.backend,
            rank=rank,
            world_size=strategy.world_size,
        )

        trainer_fn(*args, **kwargs)

        strategy.cleanup()

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def setup_training(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        dataloader: DataLoader,
    ) -> Tuple[nn.Module, Optional[Optimizer], DataLoader]:
        """
        Wrap model, optimizer, and dataloader for DDP training.

        Args:
            model (nn.Module): Model to train.
            optimizer (Optional[Optimizer]): Optimizer for training.
            dataloader (DataLoader): Training dataloader.

        Returns:
            Tuple[nn.Module, Optional[Optimizer], DataLoader]: Wrapped model, optimizer, and dataloader.
        """
        model = self.wrap_model(model)

        if optimizer is not None:
            optimizer = self.setup_optimizer(optimizer, model)

        if self.is_distributed():
            sampler: DistributedSampler = DistributedSampler(
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
                num_workers=getattr(dataloader, "num_workers", 0),
                pin_memory=getattr(dataloader, "pin_memory", False),
            )

        return model, optimizer, dataloader

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for DistributedSampler (for shuffling).

        Args:
            epoch (int): Current epoch number.
        """
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

    def wrap_model(self, model: nn.Module, stage: str = "train") -> nn.Module:
        """
        Move model to device and wrap in DDP if necessary.

        Args:
            model (nn.Module): Model to wrap.
            stage (str): Stage of training ('train' or 'eval').

        Returns:
            nn.Module: Wrapped model.
        """
        model = model.to(self.device)
        if stage == "train" and self.is_distributed():
            device_ids = [self.rank] if self.device.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
        return model

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap DDP model to get original nn.Module.

        Args:
            model (nn.Module): Model potentially wrapped in DDP.

        Returns:
            nn.Module: Original model.
        """
        if isinstance(model, DDP):
            return model.module
        return model
