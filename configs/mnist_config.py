from dataclasses import dataclass, field
from ai_playground.configs.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    DistributedConfig,
    ExperimentalConfig,
)


@dataclass
class ModelMNISTConfig(ModelConfig):
    model_name: str = "mnist"
    model_kwargs: dict = field(
        default_factory=lambda: {
            "input_dims": 28 * 28,
            "hidden_dims": [256],
            "output_dims": 10,
            "droutput": 0.0,
        }
    )


@dataclass
class TrainerMNISTConfig(TrainerConfig):
    # Training
    batch_size: int = 64
    max_steps: int = 500
    val_interval: int = 100
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Checkpointing
    save_path: str = "checkpoints/mini_gpt"
    save_interval: int = 0


@dataclass
class ExperimentalMNISTConfig(ExperimentalConfig):
    experiment_name: str = "single"


@dataclass
class DistributedMNISTConfig(DistributedConfig):
    device: str = "cuda"
    rank: int = 0  # Rank of the current process
    world_size: int = 1  # Total number of processes
    distributed: str = "single"  # "single", "ddp", "deepspeed", etc.


@dataclass
class MNISTConfig(Config):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelMNISTConfig = field(default_factory=ModelMNISTConfig)  # type: ignore
    trainer: TrainerMNISTConfig = field(default_factory=TrainerMNISTConfig)  # type: ignore
    distributed: DistributedMNISTConfig = field(default_factory=DistributedMNISTConfig)  # type: ignore
    experimental: ExperimentalMNISTConfig = field(default_factory=ExperimentalMNISTConfig)  # type: ignore
