import pytest
import torch
from ai_playground.configs.config import DistributedConfig
from ai_playground.distributed.single import SingleDevice


def test_single_device_requires_world_size_one():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = SingleDevice(config)
    assert p.world_size == 1
    assert p.device.type == "cpu"


def test_single_device_world_size_ne_one():
    config2 = DistributedConfig(
        device="cpu", distributed="single", rank=0, world_size=2
    )
    with pytest.raises(ValueError):
        SingleDevice(config2)


def test_single_device_wrap_model_moves_model_cpu():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = SingleDevice(config)
    model = torch.nn.Linear(2, 2)
    wrapped = p.wrap_model(model)

    for param in wrapped.parameters():
        assert param.device.type == "cpu"

    x = torch.randn(1, 2)
    out = wrapped(x)
    assert out.shape == (1, 2)


def test_single_device_unwrap_model():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = SingleDevice(config)
    model = torch.nn.Linear(2, 2)
    wrapped = p.wrap_model(model)
    unwrapped = p.unwrap_model(wrapped)
    assert unwrapped is wrapped


def test_single_device_backward_and_optimizer_step():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = SingleDevice(config)

    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    loss = (param - 2.0).pow(2).mean()

    p.backward(loss)
    assert param.grad is not None

    old_value = param.item()
    p.optimizer_step(optimizer)
    assert param.item() != old_value

    # GradScaler path
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    loss = (param - 2.0).pow(2).mean()
    scaler = torch.amp.grad_scaler.GradScaler()
    scaled_loss = scaler.scale(loss)
    p.backward(scaled_loss)
    p.optimizer_step(optimizer, scaler=scaler)
    assert hasattr(scaler, "_enabled")


def test_single_device_barrier_and_rank_zero_only():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    p = SingleDevice(config)
    p.barrier()
    called = []

    @p.rank_zero_only
    def fn():
        called.append(True)

    fn()
    assert called == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_single_device_wrap_model_cuda():
    config = DistributedConfig(
        device="cuda", distributed="single", rank=0, world_size=1
    )
    p = SingleDevice(config)
    model = torch.nn.Linear(2, 2)
    wrapped = p.wrap_model(model)
    for param in wrapped.parameters():
        assert param.device.type == "cuda"
