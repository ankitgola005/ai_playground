from pathlib import Path

import torch
from ai_playground.utils.checkpointing import (
    get_latest_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
)
from ai_playground.configs.config import TrainerConfig, LRConfig, ProfilerConfig


def make_trainer_config(ckpt_dir: Path) -> TrainerConfig:
    """Helper to generate minimal TrainerConfig for tests."""
    return TrainerConfig(
        seed=42,
        auto_restart=False,
        batch_size=1,
        max_steps=1,
        val_interval=1,
        max_val_steps=1,
        lr_config=LRConfig(scheduler="constant", lr=0.1),
        betas=(0.9, 0.95),
        warmup_steps=0,
        weight_decay=0.0,
        grad_clip=1.0,
        precision="fp32",
        use_progress_bar=False,
        logger=[],
        log_interval=1,
        base_dir=ckpt_dir.parent,
        run_name="test",
        use_profiler=False,
        profiler_config=ProfilerConfig(
            record_shapes=False,
            with_stack=True,
            profile_memory=True,
            wait=0,
            warmup=0,
            active=0,
            repeat=0,
        ),
        ckpt_dir=ckpt_dir,
        save_interval=1,
        log_dir=ckpt_dir / "logs",
    )


def test_get_latest_checkpoint_path_exists(tmpdir):
    ckpt_dir = Path(tmpdir) / "ckpts"
    ckpt_dir.mkdir()
    latest = ckpt_dir / "ckpt_latest.pt"
    latest.touch()

    config = make_trainer_config(ckpt_dir)
    path = get_latest_checkpoint_path(config)
    assert path == latest


def test_get_latest_checkpoint_path_not_exists(tmpdir):
    ckpt_dir = Path(tmpdir) / "ckpts"

    config = make_trainer_config(ckpt_dir)
    path = get_latest_checkpoint_path(config)
    assert path is None


def test_save_checkpoint_and_load(tmpdir):
    ckpt_dir = Path(tmpdir) / "ckpts"
    ckpt_dir.mkdir()
    config = make_trainer_config(ckpt_dir)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = None
    scaler = None
    step = 10

    def unwrap_fn(m):
        return m

    # Save checkpoint
    save_checkpoint(config, model, optimizer, scheduler, scaler, step)

    latest_path = ckpt_dir / "ckpt_latest.pt"
    step_path = ckpt_dir / f"ckpt_step_{step}.pt"
    assert latest_path.exists()
    assert step_path.exists()

    # Modify model weights
    model.weight.data.zero_()
    model.bias.data.zero_()

    # Load checkpoint
    loaded_step = load_checkpoint(
        config, model, torch.device("cpu"), optimizer, scheduler, scaler
    )
    assert loaded_step == step
    # Check that model weights were restored
    assert not torch.equal(model.weight.data, torch.zeros_like(model.weight.data))
