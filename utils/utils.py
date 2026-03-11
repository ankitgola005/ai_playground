import random
import torch
import torch.nn as nn
import numpy as np
import math
import subprocess
from tqdm import tqdm
from typing import Type, TYPE_CHECKING

from ai_playground.configs.config import ConfigProtocol, DistributedConfigProtocol
from ai_playground.data import dataset
from ai_playground.data.char_tokenizer import CharTokenizer

if TYPE_CHECKING:
    from ai_playground.distributed.base import Parallel


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


def build_model(config: ConfigProtocol) -> Type[nn.Module]:
    model: Type[nn.Module] | None = None
    if config.model.model_name == "minigpt":
        from ai_playground.models.miniGPT import MiniGPT

        model = MiniGPT
    elif config.model.model_name == "bigram":
        from ai_playground.models.bigram import BiGram

        model = BiGram
    elif config.model.model_name == "mnist":
        from ai_playground.models.mnist import MNIST

        model = MNIST
    else:
        raise NotImplementedError(
            f"Model {config.model.model_name} is currently not supported."
        )
    return model


def get_strategy(config: DistributedConfigProtocol) -> Parallel:
    strategy: Parallel | None = None
    if config.distributed == "single":
        from ai_playground.distributed.single import SingleDevice

        strategy = SingleDevice(config)
    elif config.distributed == "ddp":
        from ai_playground.distributed.ddp import DDParallel

        strategy = DDParallel(config)
    else:
        raise NotImplementedError(
            f"Strategy {config.distributed} is currently not supported."
        )
    return strategy


def build_data_pipeline(config: ConfigProtocol):
    with open(config.data.data_path, "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = dataset.train_val_split(encoded, split=config.data.split)
    train_loader = dataset.build_dataloader(config, train_data)
    val_loader = dataset.build_dataloader(config, val_data)

    return tokenizer, train_loader, val_loader


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: ConfigProtocol):
    """
    Build LR scheduler with independent warmup.
    Supported schedulers:
    constant
    linear_decay
    cosine
    cosine_restart
    exponential_decay
    polynomial_decay
    one_cycle
    """
    lr_config = config.trainer.lr_config
    max_steps = config.trainer.max_steps
    warmup_steps = config.trainer.warmup_steps
    
    scheduler = lr_config.get("scheduler", "cosine")
    min_lr_ratio = lr_config.get("min_lr_ratio", 0.1)
    
    gamma = lr_config.get("exp_gamma", 0.95)
    power = lr_config.get("poly_power", 2.0)
    cycle_steps = lr_config.get("cycle_steps", max_steps)
    one_cycle_pct = lr_config.get("one_cycle_pct", 0.3)

    def lr_lambda(step: int):
        # Warmup
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # normalized progress after warmup
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        # Schedulers
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
                # LR increase phase
                cycle_progress = progress / one_cycle_pct
                decay = min_lr_ratio + (1 - min_lr_ratio) * cycle_progress
            else:
                # LR decay phase
                cycle_progress = (progress - one_cycle_pct) / (1 - one_cycle_pct)
                decay = (1 - cycle_progress)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        # Apply min LR floor
        if scheduler not in ["constant", "exponential_decay"]:
            decay = decay * (1.0 - min_lr_ratio) + min_lr_ratio
        return decay

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
        return "commit: unknown, branch: unknown, dirty: unknown"
