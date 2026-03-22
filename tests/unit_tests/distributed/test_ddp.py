import os
import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from ai_playground.configs.config import DistributedConfig
from ai_playground.distributed.ddp import DDParallel

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "12345"


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def test_ddp_wrap_model_non_distributed():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)
    m = DummyModel()

    wrapped = p.wrap_model(m, stage="train")
    assert wrapped is m
    first_param = next(wrapped.parameters())
    assert first_param.device.type == "cpu"


def test_ddp_setup_training_no_sampler_when_not_distributed():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    dataset = TensorDataset(torch.randn(4, 2), torch.randn(4, 2))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    wrapped_model, wrapped_opt, wrapped_loader = p.setup_training(
        model, optimizer, loader
    )

    assert wrapped_model is model
    assert wrapped_opt is optimizer
    assert wrapped_loader is loader


def test_ddp_worker_entry_initializes_group_and_calls_trainer(monkeypatch):
    from ai_playground.distributed.ddp import DDParallel

    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=2)
    p = DDParallel(config)

    called = []

    def fake_init(backend, rank, world_size):
        called.append((backend, rank, world_size))

    def fake_cleanup():
        called.append("cleanup")

    monkeypatch.setattr(p, "cleanup", fake_cleanup)
    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init)

    def trainer_fn(a, b):
        called.append(("trainer", a, b))

    p._worker_entry(rank=0, strategy=p, trainer_fn=trainer_fn, args=(1, 2), kwargs={})

    assert ("gloo", 0, 2) in called
    assert ("trainer", 1, 2) in called
    assert "cleanup" in called


def test_ddp_unwrap_model():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)
    model = DummyModel()
    wrapped = p.wrap_model(model)
    unwrapped = p.unwrap_model(wrapped)
    assert unwrapped is model


def test_ddp_backward_and_optimizer_step():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)

    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    loss = (param - 2.0).pow(2).mean()

    p.backward(loss)
    assert param.grad is not None

    old_val = param.item()
    p.optimizer_step(optimizer)
    assert param.item() != old_val

    # GradScaler path
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.grad_scaler.GradScaler()
    scaled_loss = scaler.scale((param - 2.0).pow(2).mean())
    p.backward(scaled_loss)
    p.optimizer_step(optimizer, scaler=scaler)
    assert hasattr(scaler, "_enabled")


def test_ddp_barrier_and_all_reduce_noop():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)
    x = torch.tensor([1.0, 2.0])
    y = p.all_reduce(x.clone())
    assert torch.equal(x, y)
    p.barrier()


def test_ddp_reduce_mean_single_process():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)

    x = torch.tensor([2.0, 4.0])
    y = p.reduce_mean(x.clone())
    # In single process, should remain same
    assert torch.equal(x, y)


def test_ddp_is_main_process_and_launch():
    config = DistributedConfig(device="cpu", distributed="ddp", rank=0, world_size=1)
    p = DDParallel(config)

    # rank 0 is main process
    assert p.is_main_process()

    def _fn(a, b, called_list):
        called_list.append((a, b))

    called = []
    p.launch(_fn, 1, 2, called)
    assert (1, 2) in called


def _worker_fn(rank, strategy, called_list):
    """
    Worker function to be run in each DDP process.
    Moves model to device, performs a single optimizer step, and logs rank.
    """
    strategy.rank = rank
    model = DummyModel().to(strategy.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # forward + loss
    x = torch.ones(1, 2).to(strategy.device)
    y = torch.ones(1, 2).to(strategy.device)
    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()

    strategy.optimizer_step(optimizer)

    # record that this rank ran
    called_list.append(rank)

    strategy.cleanup()


def test_ddp_multi_process_run():
    world_size = 2
    config = DistributedConfig(
        device="cpu", distributed="ddp", rank=0, world_size=world_size
    )
    strategy = DDParallel(config)

    manager = mp.Manager()
    called_list = manager.list()

    mp.spawn(
        _worker_fn,
        args=(strategy, called_list),
        nprocs=world_size,
        join=True,
    )

    assert set(called_list) == set(range(world_size))
