import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs import LRConfig


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_config: "LRConfig",
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a learning rate scheduler with optional independent warmup.

    Supported schedulers:
        - "constant"          : Constant LR
        - "linear_decay"      : Linear decay from initial LR to 0
        - "cosine"            : Cosine decay
        - "cosine_restart"    : Cosine decay with restarts
        - "exponential_decay" : Exponential decay
        - "polynomial_decay"  : Polynomial decay
        - "one_cycle"         : One-cycle LR schedule (increase then decrease)

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which to schedule the LR.
        config (ConfigProtocol): Configuration object with attributes:
            - config.trainer.lr_config : Dict containing scheduler parameters.
            - config.trainer.max_steps : Total number of training steps.
            - config.trainer.warmup_steps : Number of warmup steps.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured LR scheduler.
    """

    def lr_lambda(step: int) -> float:
        """Compute LR multiplier for the given step."""
        # Warmup phase
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)

        # Progress after warmup
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        # Compute decay
        if lr_config.scheduler == "constant":
            decay = 1.0
        elif lr_config.scheduler == "linear_decay":
            decay = 1.0 - progress
        elif lr_config.scheduler == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif lr_config.scheduler == "cosine_restart":
            assert lr_config.cycle_steps is not None and lr_config.cycle_steps > 0
            cycle_pos = (step - warmup_steps) % lr_config.cycle_steps
            cycle_progress = cycle_pos / lr_config.cycle_steps
            decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        elif lr_config.scheduler == "exponential_decay":
            assert lr_config.gamma is not None
            decay = lr_config.gamma ** (step - warmup_steps)
        elif lr_config.scheduler == "polynomial_decay":
            assert lr_config.power is not None
            decay = (1.0 - progress) ** lr_config.power
        elif lr_config.scheduler == "one_cycle":
            assert (
                lr_config.one_cycle_pct is not None
                and lr_config.one_cycle_pct > 0
                and lr_config.min_lr_ratio is not None
            )
            if progress < lr_config.one_cycle_pct:
                cycle_progress = progress / lr_config.one_cycle_pct
                decay = (
                    lr_config.min_lr_ratio
                    + (1 - lr_config.min_lr_ratio) * cycle_progress
                )
            else:
                cycle_progress = (progress - lr_config.one_cycle_pct) / (
                    1 - lr_config.one_cycle_pct
                )
                decay = 1 - cycle_progress
        else:
            raise ValueError(f"Unknown scheduler: {lr_config.scheduler}")

        # Apply min LR floor for all except constant or exponential
        if lr_config.scheduler not in ["constant", "exponential_decay"]:
            assert lr_config.min_lr_ratio is not None
            decay = decay * (1.0 - lr_config.min_lr_ratio) + lr_config.min_lr_ratio

        return decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
