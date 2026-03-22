from pathlib import Path
from typing import TYPE_CHECKING

import torch
import shutil

if TYPE_CHECKING:
    from torch import nn
    from torch.amp.grad_scaler import GradScaler
    from typing import Optional, Dict, Any
    from ai_playground.configs import TrainerConfig


def _get_checkpoint_dir(trainer_config: TrainerConfig) -> Path:
    """
    Resolve checkpoint directory from config.

    Args:
        trainer_config: TrainerConfig

    Returns:
        Path to checkpoint directory
    """
    assert trainer_config.ckpt_dir is not None
    return Path(trainer_config.ckpt_dir)


def get_latest_checkpoint_path(trainer_config: TrainerConfig) -> Optional[Path]:
    """
    Get latest checkpoint path if it exists.

    Args:
        trainer_config: TrainerConfig

    Returns:
        Path to latest checkpoint or None
    """
    ckpt_path: Path = _get_checkpoint_dir(trainer_config) / "ckpt_latest.pt"
    return ckpt_path if ckpt_path.exists() else None


def save_checkpoint(
    trainer_config: TrainerConfig,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    step: int,
    unwrap_fn,
) -> None:
    """
    Save training checkpoint safely (atomic latest update).

    Args:
        trainer_config: TrainerConfig
        model: Model (possibly wrapped)
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: GradScaler (for AMP)
        step: Current global step
        unwrap_fn: Function to unwrap model (e.g. DDP/FSDP)
    """
    ckpt_dir: Path = _get_checkpoint_dir(trainer_config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model": unwrap_fn(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
    }

    step_path = ckpt_dir / f"ckpt_step_{step}.pt"
    torch.save(checkpoint, step_path)

    # atomic copy to latest ckpt
    latest_path = ckpt_dir / "ckpt_latest.pt"
    temp_path = ckpt_dir / "ckpt_latest.pt_"

    shutil.copy2(step_path, temp_path)
    temp_path.replace(latest_path)


def load_checkpoint(
    trainer_config: TrainerConfig,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
) -> int:
    """
    Load latest checkpoint if available.

    Args:
        config: Global config
        model: Model to load into
        device: Target device
        optimizer: Optimizer (can be None initially)
        scheduler: LR scheduler
        scaler: GradScaler

    Returns:
        Restored global step (0 if no checkpoint)
    """
    ckpt_path: Path | None = get_latest_checkpoint_path(trainer_config)

    if ckpt_path is None:
        return -1

    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint.get("step", 0)
