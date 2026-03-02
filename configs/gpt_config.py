from dataclasses import dataclass
from config import Config, DataConfig, ModelConfig, TrainerConfig, ExperimentalConfig


@dataclass
class ModelGPTConfig(ModelConfig):
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embed: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TrainerGPTConfig(TrainerConfig):
    # Training
    batch_size: int = 64
    max_steps: int = 5000
    val_interval: int = 100
    lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Checkpointing
    save_path: str = "checkpoints/mini_gpt"
    save_interval: int = 0


@dataclass
class GPTConfig(Config):
    dptdata: DataConfig = DataConfig()
    gptmodel: ModelGPTConfig = ModelGPTConfig()
    gpttrainer: TrainerGPTConfig = TrainerGPTConfig()
    gptexperimental: ExperimentalConfig = ExperimentalConfig()
