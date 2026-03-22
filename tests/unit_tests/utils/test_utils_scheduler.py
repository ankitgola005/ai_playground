import torch
import pytest
from ai_playground.utils.scheduler import build_lr_scheduler
from ai_playground.configs.config import LRConfig


@pytest.mark.parametrize(
    "scheduler_type, min_lr_ratio, steps_to_check, expected_range",
    [
        ("constant", None, [0, 5, 10], (0.1, 0.1)),
        ("linear_decay", 0.1, [0, 5, 10], (0.01, 0.1)),
        ("cosine", 0.1, [0, 5, 9], (0.01, 0.1)),
    ],
)
def test_build_lr_scheduler_variants(
    scheduler_type, min_lr_ratio, steps_to_check, expected_range
):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1)
    lr_config = LRConfig(scheduler=scheduler_type, lr=0.1, min_lr_ratio=min_lr_ratio)
    scheduler = build_lr_scheduler(optimizer, lr_config, warmup_steps=0, max_steps=10)

    for step in range(steps_to_check[-1] + 1):
        if step in steps_to_check:
            lr = scheduler.get_last_lr()[0]
            assert expected_range[0] - 1e-6 <= lr <= expected_range[1] + 1e-6
        optimizer.step()
        scheduler.step()


def test_build_lr_scheduler_warmup_progression():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1)
    lr_config = LRConfig(scheduler="constant", lr=0.1)
    warmup_steps = 5
    scheduler = build_lr_scheduler(
        optimizer, lr_config, warmup_steps=warmup_steps, max_steps=10
    )

    for step in range(warmup_steps + 2):
        lr = scheduler.get_last_lr()[0]
        if step < warmup_steps:
            expected = 0.1 * step / warmup_steps
            assert abs(lr - expected) < 1e-6
        else:
            assert abs(lr - 0.1) < 1e-6
        optimizer.step()
        scheduler.step()


def test_build_lr_scheduler_unknown_raises():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1)

    # Create a dummy LRConfig object without validation
    lr_config = LRConfig(scheduler="constant", lr=0.1)
    lr_config.scheduler = "unknown"

    with pytest.raises(ValueError):
        build_lr_scheduler(optimizer, lr_config, warmup_steps=0, max_steps=10)
