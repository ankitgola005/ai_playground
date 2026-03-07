
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ai_playground.trainer import Trainer
from ai_playground.utils.utils import build_data_pipeline, build_model

if TYPE_CHECKING:
    from configs.config import Config

def training_worker(rank: int, config: Config):
    if config.distributed.distributed != "single":
        backend = "cuda" if config.distributed.device == "cuda" and torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=config.distributed.world_size,
        )
    
    trainer = Trainer(config)
    trainer.strategy.rank = rank
    trainer.strategy.world_size = config.distributed.world_size

    model = build_model(config)
    tokenizer, train_loader, val_loader = build_data_pipeline(config)

    trainer.fit(model(), train_loader, val_loader)

    if dist.is_initialized():
        dist.destroy_process_group()    