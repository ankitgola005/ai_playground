import random
import torch
import numpy as np
import pytest
from unittest.mock import patch

from ai_playground.utils.utils import (
    set_seed,
    resolve_device,
    precision_to_dtype,
    setup_progress_bar,
    get_git_info,
)


def test_set_seed_reproducibility():
    seed = 123

    # Random
    set_seed(seed)
    rand_val = random.randint(0, 100)
    set_seed(seed)
    assert random.randint(0, 100) == rand_val

    # Numpy
    set_seed(seed)
    np_val = np.random.randint(0, 100)
    set_seed(seed)
    assert np.random.randint(0, 100) == np_val

    # Torch
    set_seed(seed)
    torch_val = torch.randint(0, 100, (1,)).item()
    set_seed(seed)
    assert torch.randint(0, 100, (1,)).item() == torch_val


@pytest.mark.parametrize(
    "input_device,cuda_available,expected",
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_resolve_device_param(input_device, cuda_available, expected):
    with patch("torch.cuda.is_available", return_value=cuda_available):
        if input_device == "cuda" and not cuda_available:
            with pytest.raises(RuntimeError):
                resolve_device(input_device)
        else:
            assert resolve_device(input_device) == expected


@pytest.mark.parametrize(
    "precision,dtype",
    [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_precision_to_dtype_valid(precision, dtype):
    assert precision_to_dtype(precision) == dtype


def test_precision_to_dtype_invalid():
    with pytest.raises(ValueError):
        precision_to_dtype("invalid")


def test_setup_progress_bar():
    bar = setup_progress_bar(initial_step=10, total_steps=100, desc="Test")
    assert bar.n == 10
    assert bar.total == 100
    assert bar.desc == "Test"


@pytest.mark.parametrize(
    "side_effects,expected",
    [
        ([b"abc123\n", b"main\n", b""], "commit: abc123, branch: main, dirty: "),
        (
            Exception("Git not found"),
            "commit: unknown, branch: unknown, dirty: unknown",
        ),
    ],
)
def test_get_git_info(side_effects, expected):
    with patch("subprocess.check_output") as mock:
        mock.side_effect = side_effects
        info = get_git_info()
        assert info.startswith(expected.split(", dirty")[0])
