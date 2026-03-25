import pytest
import torch
import torch.nn as nn

from types import SimpleNamespace

from ai_playground.trainer import Trainer


class DummyStrategy:
    def __init__(self):
        self.device_type = "cpu"
        self.device = torch.device("cpu")
        self.world_size = 1
        self.rank = 0

    def setup_environment(self, stage="train"):
        pass

    def wrap_model(self, model, stage="train"):
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
        loss = ((out - y) ** 2).mean()
        return out, loss, None


class NaNModel(nn.Module):
    def forward(self, x, y):
        return x, torch.tensor(float("nan")), None


class DummyGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompts, max_tokens, use_cache):
        return ["ok"] * len(prompts)


@pytest.fixture
def config():
    return SimpleNamespace(
        trainer=SimpleNamespace(
            seed=42,
            precision="fp32",
            warmup_steps=0,
            max_steps=2,
            save_interval=0,
            val_interval=1,
            log_interval=1,
            use_progress_bar=False,
            use_profiler=False,
            log_dir=None,
            profiler_config=None,
            run_name="test",
            weight_decay=0.01,
            betas=(0.9, 0.95),
            grad_clip=1.0,
            lr_config=SimpleNamespace(lr=1e-3, scheduler="cosine", min_lr_ratio=0.1),
        ),
        model=SimpleNamespace(compile=False),
    )


@pytest.fixture
def trainer(config, monkeypatch):
    strategy = DummyStrategy()

    class DummyLoggerManager:
        log_frequency = 1

        def log_metrics(self, *args, **kwargs):
            pass

        def log_config(self, *args, **kwargs):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr("ai_playground.trainer.trainer.Generator", DummyGenerator)

    monkeypatch.setattr(
        "ai_playground.trainer.trainer.create_loggers",
        lambda *args, **kwargs: DummyLoggerManager(),
    )

    return Trainer(config, strategy)


def test_train_step_runs(trainer):
    model = DummyModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb = torch.randn(2, 16)
    yb = torch.randn(2, 16)
    logits, loss = trainer._train_step(
        model, xb, yb, trainer.optimizer, trainer.lr_scheduler
    )

    assert loss.item() > 0
    assert logits.shape == xb.shape


def test_nan_loss_raises(trainer):
    model = NaNModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb = torch.randn(2, 16)
    yb = torch.randn(2, 16)
    with pytest.raises(RuntimeError):
        trainer._train_step(model, xb, yb, trainer.optimizer, trainer.lr_scheduler)


def test_validate_runs(trainer):
    model = DummyModel()
    data = [
        (torch.randn(2, 16), torch.randn(2, 16)),
        (torch.randn(2, 16), torch.randn(2, 16)),
    ]
    val_loss = trainer._validate(model, data)

    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_scheduler_steps(trainer):
    model = DummyModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb = torch.randn(2, 16)
    yb = torch.randn(2, 16)

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer._train_step(model, xb, yb, trainer.optimizer, trainer.lr_scheduler)
    new_lr = trainer.optimizer.param_groups[0]["lr"]

    assert new_lr != initial_lr


def test_predict(trainer):
    model = DummyModel()
    outputs = trainer.predict(model, tokenizer=None, prompts=["hi"])

    assert isinstance(outputs, list)
    assert outputs[0] == "ok"


def test_should_stop(trainer):
    trainer.global_step = 2
    trainer.max_steps = 2

    assert trainer._should_stop() is True


def test_grad_check_flag(trainer):
    model = DummyModel()
    trainer.configure_optimizer_and_scheduler(model)
    trainer.check_finite_grads = True
    xb = torch.randn(2, 16)
    yb = torch.randn(2, 16)

    # Should not raise
    trainer._train_step(model, xb, yb, trainer.optimizer, trainer.lr_scheduler)
