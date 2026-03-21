from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from ai_playground.configs import TrainerConfig


def resolve_run_name(run_name: str) -> str:
    """
    Resolve run name. Generates timestamp-based name if not provided.

    Args:
        str: run_name

    Returns:
        str: Resolved run name
    """
    return run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_dirs(cfg: TrainerConfig) -> Tuple[Path, Path, Path]:
    """
    Resolve logging and checkpoint dirs:

    Args:
        TrainerConfig

    return:
        Paths run_dir, log_dir, and ckpt_dir
    """

    # Get base run dir
    run_dir = Path(cfg.base_dir) / resolve_run_name(cfg.run_name)

    # Resolve log and ckpt dir
    log_dir = Path(cfg.log_dir) if cfg.log_dir else (run_dir / "logs")
    ckpt_dir = Path(cfg.ckpt_dir) if cfg.ckpt_dir else (run_dir / "checkpoints")

    # Create dirs
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, ckpt_dir
