from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_path: str = "data/datasets/text_datasets/shakespeare.txt"
    split: float = 0.9
    num_workers: int = 0


@dataclass
class ModelConfig:
    block_size: int = 8  # Maximum context length
    n_embed: int = 4
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1


@dataclass
class TrainerConfig:
    # Training
    auto_restart: bool = True
    batch_size: int = 32  # How many independent sequences will be processed
    max_epochs: int = 1
    max_steps: int = 500
    val_interval: int = 50
    lr: float = 1e-2
    min_lr_ratio: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "fp16"  # "fp16", "bf16" or "fp32"

    # Precision
    use_fp16: bool = False

    # Debugging
    use_progress_bar: bool = True
    logger: str = "tensorboard"  # "basic", "tensorboard", or None
    log_dir: str = "runs/"
    log_interval: int = 10
    use_profiler: bool = False
    profiler_wait: int = 1
    profiler_warmup: int = 1
    profiler_active: int = 3
    profiler_repeat: int = 2

    # Checkpointing
    save_path: str = "checkpoints"
    save_interval: int = 0


@dataclass
class ExperimentalConfig:
    seed: int = 42
    compile: bool = False
    experiment_name: str = ""


@dataclass
class DistributedConfig:
    device: str = "cuda"
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    rank: int = 0  # Rank of the current process
    world_size: int = 1  # Total number of processes
    distributed: str = "single"  # "single", "ddp", "deepspeed", etc.


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
