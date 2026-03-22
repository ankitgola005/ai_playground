import pytest
from pathlib import Path
from pydantic import ValidationError
from ai_playground.configs.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    LRConfig,
    ProfilerConfig,
    DistributedConfig,
)


def make_minimal_config():
    """Helper to create a valid base config"""
    return Config(
        data=DataConfig(
            dataset="shakespeare", split=0.9, num_workers=0, block_size=128
        ),
        model=ModelConfig(
            model_name="minigpt", compile=True, model_kwargs={"n_layer": 4}
        ),
        trainer=TrainerConfig(
            seed=42,
            auto_restart=True,
            batch_size=64,
            eval_batch_size=4,
            max_steps=1000,
            val_interval=100,
            lr_config=LRConfig(scheduler="constant", lr=0.0003),
            betas=(0.9, 0.95),
            warmup_steps=100,
            weight_decay=0.01,
            grad_clip=1.0,
            precision="fp16",
            use_progress_bar=True,
            logger=["tensorboard"],
            log_interval=10,
            log_dir=Path("runs/logs"),
            base_dir=Path("runs"),
            run_name="experiment",
            use_profiler=True,
            profiler_config=ProfilerConfig(
                record_shapes=False,
                with_stack=True,
                profile_memory=True,
                wait=1,
                warmup=1,
                active=3,
                repeat=0,
            ),
            ckpt_dir=Path("runs/ckpt"),
            save_interval=10,
        ),
        distributed=DistributedConfig(
            device="auto", distributed="single", rank=0, world_size=1
        ),
    )


def test_data_config_validation():
    # valid
    data = DataConfig(dataset="abc", split=0.8, num_workers=0, block_size=128)
    assert data.dataset == "abc"

    # invalid split
    with pytest.raises(ValidationError):
        DataConfig(dataset="abc", split=1.5, num_workers=0, block_size=128)

    # invalid block_size
    with pytest.raises(ValidationError):
        DataConfig(dataset="abc", split=0.8, num_workers=0, block_size=0)


def test_lr_config_validation():
    # valid
    lr = LRConfig(scheduler="cosine", lr=0.001)
    assert lr.lr > 0

    # invalid lr
    with pytest.raises(ValidationError):
        LRConfig(scheduler="constant", lr=-0.1)

    # invalid scheduler
    with pytest.raises(ValidationError):
        LRConfig(scheduler="invalid", lr=0.01)


def test_trainer_config_update():
    cfg = make_minimal_config()

    # update LR dynamically
    new_lr = {"scheduler": "cosine", "lr": 0.002}
    cfg.trainer.lr_config = LRConfig(**new_lr)
    assert cfg.trainer.lr_config.lr == 0.002
    assert cfg.trainer.lr_config.scheduler == "cosine"

    # update grad_clip
    cfg.trainer.grad_clip = 0.5
    assert cfg.trainer.grad_clip == 0.5


def test_model_config_update():
    cfg = make_minimal_config()
    cfg.model.model_kwargs["n_head"] = 4
    assert cfg.model.model_kwargs["n_head"] == 4


def test_distributed_config_validation():
    cfg = make_minimal_config()
    assert cfg.distributed.device in ["auto", "cpu", "cuda"]
    assert cfg.distributed.distributed in ["single", "ddp"]

    # invalid rank
    with pytest.raises(ValidationError):
        DistributedConfig(device="cpu", distributed="single", rank=-1, world_size=1)


def test_full_config_dynamic_update():
    cfg = make_minimal_config()

    # update nested lr_config
    cfg.trainer.lr_config = LRConfig(scheduler="linear_decay", lr=0.0005)
    assert cfg.trainer.lr_config.scheduler == "linear_decay"
    assert cfg.trainer.lr_config.lr == 0.0005

    # update block size
    cfg.data.block_size = 256
    assert cfg.data.block_size == 256
