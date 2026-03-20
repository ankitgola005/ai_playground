from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from ai_playground.configs import TrainerConfig


def resolve_run_name(cfg: TrainerConfig) -> str:
    """
    Resolve run name. Generates timestamp-based name if not provided.

    Args:
        cfg: TrainerConfig

    Returns:
        str: Resolved run name
    """
    if cfg.run_name:
        return cfg.run_name

    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_dirs(cfg: TrainerConfig) -> Tuple[Optional[Path], Path, Path]:
    """
    Resolve run, log, and checkpoint directories.

    Logic:
        - If log_dir / ckpt_dir provided → use them
        - Else derive from base_dir + run_name

    Args:
        cfg: TrainerConfig

    Returns:
        (run_dir, log_dir, ckpt_dir)
    """
    run_name = resolve_run_name(cfg)

    run_dir: Optional[Path] = None
    if cfg.base_dir:
        run_dir = Path(cfg.base_dir) / run_name
    assert run_dir is not None

    log_dir = cfg.log_dir or (run_dir / "logs")
    ckpt_dir = cfg.ckpt_dir or (run_dir / "checkpoints")

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, ckpt_dir
