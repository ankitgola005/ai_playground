from dataclasses import dataclass, field
from configs.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    ExperimentalConfig,
)


@dataclass
class ModelGPTConfig(ModelConfig):
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_embed: int = 384
    hidden_dim: int = 1536  # Typically 4 * n_embed
    ffn_dropout: float = 0.1
    attn_dropout: float = 0.1
    residual_dropout: float = 0.1


@dataclass
class TrainerGPTConfig(TrainerConfig):
    # Training
    batch_size: int = 64
    max_steps: int = 1000
    val_interval: int = 100
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Checkpointing
    save_path: str = "checkpoints/mini_gpt"
    save_interval: int = 0


@dataclass
class ExperimentalGPTConfig(ExperimentalConfig):
    experiment_name: str = ""


@dataclass
class GPTConfig(Config):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelGPTConfig = field(default_factory=ModelGPTConfig)  # type: ignore
    trainer: TrainerGPTConfig = field(default_factory=TrainerGPTConfig)  # type: ignore
    experimental: ExperimentalGPTConfig = field(default_factory=ExperimentalGPTConfig)  # type: ignore
