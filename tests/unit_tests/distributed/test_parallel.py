import torch
import pytest

from ai_playground.configs.config import DistributedConfig
from ai_playground.distributed.base import Parallel


from torch.optim import SGD
from torch.amp.grad_scaler import GradScaler


class DummyParallel(Parallel):
    def setup_environment(self, stage="train"):
        pass

    def wrap_model(self, model, stage="train"):
        return model


@pytest.fixture
def config():
    return DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)


@pytest.fixture
def parallel(config):
    return DummyParallel(config)


def test_device_properties(parallel):
    assert parallel.device.type == "cpu"
    assert parallel.device_type == "cpu"

    # Can set device manually
    new_device = torch.device("cpu")
    parallel.set_device(new_device)
    assert parallel.device == new_device


def test_invalid_device(parallel):
    with pytest.raises(RuntimeError):
        parallel.set_device(torch.device("fake"))


def test_backward_and_optimizer_step(parallel):
    # Simple parameter and optimizer
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = SGD([param], lr=0.1)

    # Backward + optimizer step without GradScaler
    loss = (param - 2.0).pow(2).mean()
    parallel.backward(loss)
    assert param.grad is not None

    old_value = param.item()
    parallel.optimizer_step(optimizer)
    assert param.item() != old_value

    # Reset param for GradScaler test
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = SGD([param], lr=0.1)
    scaler = GradScaler()

    # Scale loss before backward
    loss = (param - 2.0).pow(2).mean()
    scaled_loss = scaler.scale(loss)
    parallel.backward(scaled_loss)
    parallel.optimizer_step(optimizer, scaler=scaler)

    # Check that scaler is still valid
    assert hasattr(scaler, "_enabled")


def test_rank_and_main_process(parallel):
    # By default, rank 0
    assert parallel.rank == 0
    assert parallel.is_main_process()
    # rank_zero_only decorator
    called = []

    @parallel.rank_zero_only
    def dummy_fn():
        called.append(True)
        return 123

    res = dummy_fn()
    assert res == 123
    assert called == [True]


def test_all_reduce_and_reduce_mean(parallel):
    t = torch.tensor([1.0])
    # Not distributed, should return original tensor
    out = parallel.all_reduce(t)
    assert out is t
    out_mean = parallel.reduce_mean(t)
    assert out_mean.item() == 1.0


def test_rank_zero_only_decorator():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = DummyParallel(config)

    calls = []

    @p.rank_zero_only
    def f(x):
        calls.append(x)

    p.rank = 0
    f(1)
    assert calls == [1]

    p.rank = 1
    f(2)
    assert calls == [1]


def test_barrier_no_distribution(monkeypatch):
    import ai_playground.distributed.base as base

    called = False

    def fake_barrier():
        nonlocal called
        called = True

    monkeypatch.setattr(base.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(base.dist, "barrier", fake_barrier)

    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = DummyParallel(config)
    p.barrier()
    assert called is False


def test_cleanup_does_not_fail(parallel):
    # Should be safe even if dist not initialized
    parallel.cleanup()
