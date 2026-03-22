from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast, Literal

import torch
import torch.distributed as dist
from ai_playground.utils import resolve_device

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
    from torch.amp.grad_scaler import GradScaler
    from ai_playground.configs.config import DistributedConfig
    from typing import Optional, Callable, Any


class Parallel(ABC):
    """
    Base class for distributed and parallel training strategies.
    """

    def __init__(self, config: "DistributedConfig"):
        """
        Initialize the parallel strategy.

        Args:
            config (DistributedConfigProtocol): Configuration object containing
                device type and distributed settings like world_size.

        Raises:
            ValueError: If 'cuda' is specified but not available.
        """
        device: Literal["cpu", "cuda"] = resolve_device(config.device)
        self._device: torch.device = torch.device(device)
        self.rank: int = 0
        self.world_size: int = config.world_size
        self.backend: str = "nccl" if self.device_type == "cuda" else "gloo"

    def init_distributed(self, rank: int, world_size: int) -> None:
        """
        Initialize the process group for distributed training.

        Args:
            rank (int): Rank of the current process.
            world_size (int): Total number of processes.
        """
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend, rank=rank, world_size=world_size
            )
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def setup_environment(self, stage: str = "train") -> None:
        """
        Set up the environment for the given stage (e.g., 'train' or 'eval').

        Args:
            stage (str): Stage of training or evaluation.
        """
        pass

    @abstractmethod
    def wrap_model(self, model: nn.Module, stage: str = "train") -> nn.Module:
        """
        Wrap the model for distributed training as per selected strategy.

        Args:
            model (nn.Module): The model to wrap.
            stage (str): Stage of training or evaluation.

        Returns:
            nn.Module: The wrapped model.
        """
        pass

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwrap the model, if previously wrapped by strategy.

        Args:
            model (nn.Module): The model to unwrap.

        Returns:
            nn.Module: Unwrapped model.
        """
        return model

    @property
    def device(self) -> torch.device:
        """Return the device this parallel strategy is using."""
        return self._device

    def set_device(self, device: torch.device) -> None:
        """Set the device manually."""
        self._device = device

    @property
    def device_type(self) -> Literal["cpu", "cuda"]:
        """
        Return the type of device ('cpu' or 'cuda').
        """
        return cast(Literal["cpu", "cuda"], self._device.type)

    def is_distributed(self) -> bool:
        """Check if distributed training is active."""
        return dist.is_initialized() and self.world_size > 1

    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass on the loss tensor.

        Args:
            loss (torch.Tensor): Loss tensor to backpropagate.
        """
        loss.backward()

    def setup_optimizer(self, optimizer: "Optimizer", model: nn.Module) -> "Optimizer":
        """
        Optionally wrap or modify the optimizer for distributed training.

        Args:
            optimizer (Optimizer): Optimizer to wrap/setup.
            model (nn.Module): Model being optimized.

        Returns:
            Optimizer: The modified or original optimizer.
        """
        return optimizer

    def optimizer_step(
        self, optimizer: "Optimizer", scaler: Optional["GradScaler"] = None
    ) -> None:
        """
        Perform an optimizer step, optionally using AMP GradScaler.

        Args:
            optimizer (Optimizer): Optimizer to step.
            scaler (Optional[GradScaler]): Automatic mixed precision scaler.
        """
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def rank_zero_only(self, fn: Callable) -> Callable:
        """
        Decorator to run a function only on the main process.
        """

        def wrapper(*args, **kwargs):
            if self.is_main_process():
                return fn(*args, **kwargs)

        return wrapper

    def all_reduce(
        self, tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Perform all-reduce on a tensor across all processes.

        Args:
            tensor (torch.Tensor): Tensor to reduce.
            op (dist.ReduceOp): Reduction operation (default: SUM).

        Returns:
            torch.Tensor: Reduced tensor.
        """
        if not self.is_distributed():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor across processes and take the mean.

        Args:
            tensor (torch.Tensor): Tensor to reduce.

        Returns:
            torch.Tensor: Reduced mean tensor.
        """
        self.all_reduce(tensor)
        tensor.div_(self.world_size)
        return tensor

    def launch(self, trainer_fn: Callable, *args, **kwargs) -> None:
        """
        Launch the training function (can be overridden for multi-process).

        Args:
            trainer_fn (Callable): The training function to execute.
        """
        trainer_fn(*args, **kwargs)

    def prepare_dataloader(self, dataloader: "DataLoader") -> "DataLoader":
        """
        Optionally wrap a dataloader for distributed training.

        Args:
            dataloader (DataLoader): Dataloader to prepare.

        Returns:
            DataLoader: Prepared dataloader (default: original).
        """
        return dataloader

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
