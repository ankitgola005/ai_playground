import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

from ai_playground.trainer import Trainer
from ai_playground.callbacks.callback import Callback


class DummyStrategy:
    device_type = "cpu"
    device = torch.device("cpu")
    world_size = 1
    rank = 0

    def setup_environment(self, stage=None):
        pass

    def wrap_model(self, model, stage=None):
        return model

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self, optimizer, scaler=None):
        optimizer.step()

    def is_main_process(self):
        return True


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, x, y):
        out = self.lin(x)
        return {"logits": out, "loss": ((out - y) ** 2).mean(), "aux_metrics": None}


@pytest.fixture
def trainer(monkeypatch):
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            seed=42,
            precision="fp32",
            warmup_steps=0,
            max_steps=2,
            max_val_steps=1,
            save_interval=0,
            val_interval=1,
            log_interval=0,
            use_progress_bar=False,
            use_profiler=False,
            log_dir=None,
            ckpt_dir=None,
            profiler_config=None,
            run_name="test",
            weight_decay=0.0,
            betas=(0.9, 0.95),
            grad_clip=0.0,
            lr_config=SimpleNamespace(scheduler="constant", lr=1e-3),
        ),
        model=SimpleNamespace(compile=False),
    )

    class DummyLogger:
        log_frequency = 1

        def log_metrics(self, *a, **kw):
            pass

        def log_config(self, *a, **kw):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr(
        "ai_playground.trainer.trainer.create_loggers",
        lambda *a, **kw: DummyLogger(),
    )
    monkeypatch.setattr(
        "ai_playground.trainer.trainer.Generator",
        lambda *a, **kw: None,
    )

    t = Trainer(config, DummyStrategy())
    t.callbacks = []
    return t


class DummyCallback(Callback):
    def __init__(self):
        self.train_start_called = False
        self.train_steps = 0
        self.validation_calls = 0

    def on_train_start(self, trainer):
        self.train_start_called = True

    def on_train_step_end(self, trainer, loss, metrics):
        self.train_steps += 1

    def on_validation_end(self, trainer, val_loss):
        self.validation_calls += 1


def test_callback_hooks_are_called(trainer):
    cb = DummyCallback()
    trainer.callbacks = [cb]

    data = TensorDataset(torch.randn(8, 16), torch.randn(8, 16))
    loader = DataLoader(data, batch_size=4)

    trainer.fit(DummyModel(), loader, loader)

    assert cb.train_start_called
    assert cb.train_steps > 0
    assert cb.validation_calls > 0
