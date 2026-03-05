import random
import torch
import torch.nn as nn
import numpy as np
import subprocess
from tqdm import tqdm

from configs.config import Config
from data import dataset
from data.char_tokenizer import CharTokenizer


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_pipeline(config: Config):
    with open(config.data.data_path, "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = dataset.train_val_split(encoded, split=config.data.split)
    train_loader = dataset.build_dataloader(config, train_data)
    val_loader = dataset.build_dataloader(config, val_data)

    return tokenizer, train_loader, val_loader


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(step: int):
        # Warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_grad_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total**0.5


def get_weight_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        total += p.norm(2).item() ** 2
    return total**0.5


def get_norm_info(model: nn.Module, lr: float):
    grad_norm = get_grad_norm(model)
    weight_norm = get_weight_norm(model)
    update_ratio = (lr * grad_norm) / (weight_norm + 1e-8)
    return grad_norm, weight_norm, update_ratio


def setup_progress_bar(
    initial_step: int = 0, total_steps: int = 0, desc: str = "Training"
) -> tqdm:
    total = total_steps if total_steps > 0 else None
    return tqdm(
        total=total,
        initial=initial_step,
        dynamic_ncols=True,
        leave=True,
        desc=desc,
    )


def get_git_info():
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
        return f"commit: unknown, branch: unknown, dirty: unknown"
