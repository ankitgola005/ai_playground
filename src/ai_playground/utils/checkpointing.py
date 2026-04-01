from pathlib import Path
from typing import TYPE_CHECKING

import torch
import shutil

if TYPE_CHECKING:
    from torch import nn
    from torch.amp.grad_scaler import GradScaler
    from typing import Dict, Any
    from ai_playground.configs import TrainerConfig


def _get_checkpoint_dir(trainer_config: TrainerConfig) -> Path | None:
    """
    Resolve checkpoint directory from config.

    Args:
        trainer_config: TrainerConfig

    Returns:
        Path to checkpoint directory
    """
    return (
        Path(trainer_config.ckpt_dir) if trainer_config.ckpt_dir is not None else None
    )


def get_latest_checkpoint_path(trainer_config: TrainerConfig) -> Path | None:
    """
    Get latest checkpoint path if it exists.

    Args:
        trainer_config: TrainerConfig

    Returns:
        Path to latest checkpoint or None
    """
    ckpt_path: Path | None = _get_checkpoint_dir(trainer_config)
    if ckpt_path is None:
        return

    ckpt_path = ckpt_path / "ckpt_latest.pt"
    return ckpt_path if ckpt_path.exists() else None


def _unwrap_checkpoint_model(model: nn.Module) -> nn.Module:
    """Unwrap compiled / parallel wrappers to the original model."""
    # torch.compile wraps models with an OptimizedModule that stores the original
    # model on `_orig_mod`.
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod

    # DDP / DataParallel wrappers expose the raw module on `module`.
    while hasattr(model, "module") and isinstance(getattr(model, "module"), torch.nn.Module):
        model = getattr(model, "module")

    return model


def save_checkpoint(
    trainer_config: TrainerConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: GradScaler | None,
    step: int,
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
    """
    ckpt_dir: Path | None = _get_checkpoint_dir(trainer_config)
    if ckpt_dir is None:
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = _unwrap_checkpoint_model(model)
    checkpoint: Dict[str, Any] = {
        "model": model.state_dict(),
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
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: GradScaler | None,
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

    model = _unwrap_checkpoint_model(model)
    current_ckpt_path = getattr(model, "_loaded_checkpoint_path", None)
    current_ckpt_mtime = getattr(model, "_loaded_checkpoint_mtime", None)
    new_ckpt_mtime = ckpt_path.stat().st_mtime
    if current_ckpt_path == str(ckpt_path) and current_ckpt_mtime == new_ckpt_mtime:
        return getattr(model, "_loaded_checkpoint_step", -1)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if optimizer and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    model._loaded_checkpoint_path = str(ckpt_path)
    model._loaded_checkpoint_mtime = new_ckpt_mtime
    model._loaded_checkpoint_step = checkpoint.get("step", 0)
    return model._loaded_checkpoint_step
