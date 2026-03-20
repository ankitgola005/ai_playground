import random
import torch
import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "datasets/"


def precision_to_dtype(precision: str) -> torch.dtype:
    """
    Convert a string precision to a PyTorch dtype.

    Args:
        precision (str): One of "fp32", "fp16", or "bf16".

    Returns:
        torch.dtype: Corresponding PyTorch data type.

    Raises:
        ValueError: If precision string is not supported.
    """
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_progress_bar(
    initial_step: int = 0, total_steps: int = 0, desc: str = "Training"
) -> tqdm:
    """
    Create a tqdm progress bar for training or evaluation.

    Args:
        initial_step (int, optional): Starting step for the progress bar. Defaults to 0.
        total_steps (int, optional): Total number of steps. Defaults to 0 (unknown).
        desc (str, optional): Description of the progress bar. Defaults to "Training".

    Returns:
        tqdm: A tqdm progress bar instance.
    """
    total = total_steps if total_steps > 0 else None
    return tqdm(
        total=total,
        initial=initial_step,
        dynamic_ncols=True,
        leave=True,
        desc=desc,
    )


def get_git_info() -> str:
    """
    Get information about the current Git repository: commit hash, branch, and dirty state.

    Returns:
        str: Git info string. If Git commands fail, returns unknown placeholders.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        )
        return f"commit: {commit}, branch: {branch}, dirty: {dirty}"
    except Exception:
        return "commit: unknown, branch: unknown, dirty: unknown"
