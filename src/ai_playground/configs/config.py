# ai_playground/configs/config.py

from pydantic import BaseModel, Field, model_validator
from typing import Dict, Tuple, Sequence, Literal, Any
from pathlib import Path


# Profiler Config
class ProfilerConfig(BaseModel):
    record_shapes: bool = Field(default=False)
    with_stack: bool = Field(default=True)
    profile_memory: bool = Field(default=True)
    wait: int = Field(ge=0)
    warmup: int = Field(ge=0)
    active: int = Field(ge=0)
    repeat: int = Field(ge=0)


class LRConfig(BaseModel):
    scheduler: Literal[
        "cosine",
        "exponential_decay",
        "polynomial_decay",
        "linear_decay",
        "one_cycle",
        "constant",
        "cosine_restart",
    ]
    lr: float = Field(gt=0.0)
    min_lr_ratio: float | None = Field(default=None, gt=0.0)
    gamma: float | None = Field(default=None, gt=0.0)
    power: float | None = Field(default=None, gt=0.0)
    cycle_steps: float | None = Field(default=None, gt=0.0)
    one_cycle_pct: float | None = Field(default=None, gt=0.0)


# Data config
class DataConfig(BaseModel):
    dataset: str
    split: float = Field(ge=0.0, le=1.0)
    num_workers: int = Field(ge=0)
    block_size: int = Field(gt=1)


# Model config
class ModelConfig(BaseModel):
    model_name: str
    compile: bool
    model_kwargs: Dict[str, bool | int | float] = Field(default_factory=dict)


# Trainer config
class TrainerConfig(BaseModel):
    # Training control
    seed: int
    auto_restart: bool
    batch_size: int = Field(gt=0)
    eval_batch_size: int | None = Field(default=None, gt=0)
    max_epochs: int | None = Field(default=None, gt=0)
    max_steps: int = Field(ge=0)
    val_interval: int = Field(ge=0)

    # Optimization
    lr_config: LRConfig

    betas: Tuple[float, float]
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(ge=0)

    # Precision
    precision: Literal["fp32", "fp16", "bf16"]

    # Logging
    use_progress_bar: bool
    logger: Sequence[str]
    log_interval: int = Field(gt=0)
    log_dir: Path | None = None

    base_dir: Path
    run_name: str = ""

    # Profiler
    use_profiler: bool
    profiler_config: ProfilerConfig

    # Checkpointing
    ckpt_dir: Path | None = None
    save_interval: int = Field(ge=0)

    @model_validator(mode="after")
    def check_train_limits(self):
        if self.max_epochs is not None and self.max_steps is not None:
            raise ValueError("Use either max_epochs OR max_steps, not both")
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("One of max_epochs or max_steps must be set")
        return self

    @model_validator(mode="after")
    def check_paths(self):
        if (self.log_dir or self.ckpt_dir) and not self.base_dir:
            raise ValueError("base_dir must exist if using derived paths")
        return self


# Distributed config
class DistributedConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"]
    distributed: Literal["single", "ddp"]
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)


# Main Config
class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    distributed: DistributedConfig
