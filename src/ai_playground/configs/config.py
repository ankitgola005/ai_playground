# Utility for type hints
from typing import Protocol, Mapping, Sequence, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from typing import Tuple


class DataConfigProtocol(Protocol):
    dataset: str
    split: float
    num_workers: int


class ModelConfigProtocol(Protocol):
    model_name: str
    model_kwargs: Mapping[str, int | str | float]


class TrainerConfigProtocol(Protocol):
    auto_restart: bool
    batch_size: int
    max_epochs: int
    max_steps: int
    val_interval: int

    lr: float
    lr_config: Mapping[
        str, int | str | float
    ]  # e.g., {"schedule": str, "min_lr_ratio": float, ...}

    betas: Tuple[float, float]
    warmup_steps: int
    weight_decay: float
    grad_clip: float

    precision: str
    use_fp16: bool

    use_progress_bar: bool

    logger: Sequence[str]
    log_dir: str | Path
    log_interval: int

    use_profiler: bool
    profiler_wait: int
    profiler_warmup: int
    profiler_active: int
    profiler_repeat: int

    save_path: str | Path
    save_interval: int


class ExperimentalConfigProtocol(Protocol):
    seed: int
    compile: bool
    experiment_name: str


class DistributedConfigProtocol(Protocol):
    device: str
    rank: int
    world_size: int
    distributed: str


class ConfigProtocol(Protocol):
    data: DataConfigProtocol
    model: ModelConfigProtocol
    trainer: TrainerConfigProtocol
    distributed: DistributedConfigProtocol
    experimental: ExperimentalConfigProtocol
