import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs import TrainerConfigProtocol


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: "TrainerConfigProtocol"
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
    lr_config = config.lr_config
    max_steps = int(config.max_steps)
    warmup_steps = int(config.warmup_steps)

    scheduler = lr_config.get("scheduler", "cosine")
    min_lr_ratio = float(lr_config.get("min_lr_ratio", 0.1))
    gamma = float(lr_config.get("exp_gamma", 0.95))
    power = float(lr_config.get("poly_power", 2.0))
    cycle_steps = int(lr_config.get("cycle_steps", max_steps))
    one_cycle_pct = float(lr_config.get("one_cycle_pct", 0.3))

    def lr_lambda(step: int) -> float:
        """Compute LR multiplier for the given step."""
        # Warmup phase
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)

        # Progress after warmup
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        # Compute decay
        if scheduler == "constant":
            decay = 1.0
        elif scheduler == "linear_decay":
            decay = 1.0 - progress
        elif scheduler == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler == "cosine_restart":
            cycle_pos = (step - warmup_steps) % cycle_steps
            cycle_progress = cycle_pos / cycle_steps
            decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        elif scheduler == "exponential_decay":
            decay = gamma ** (step - warmup_steps)
        elif scheduler == "polynomial_decay":
            decay = (1.0 - progress) ** power
        elif scheduler == "one_cycle":
            if progress < one_cycle_pct:
                cycle_progress = progress / one_cycle_pct
                decay = min_lr_ratio + (1 - min_lr_ratio) * cycle_progress
            else:
                cycle_progress = (progress - one_cycle_pct) / (1 - one_cycle_pct)
                decay = 1 - cycle_progress
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        # Apply min LR floor for all except constant or exponential
        if scheduler not in ["constant", "exponential_decay"]:
            decay = decay * (1.0 - min_lr_ratio) + min_lr_ratio

        return decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
