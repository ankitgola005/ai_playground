import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from ai_playground.trainer import Trainer


class DummyStrategy:
    device_type = "cpu"
    device = torch.device("cpu")
    world_size = 1
    rank = 0

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


class DummyGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompts, max_tokens, use_cache):
        return ["ok"] * len(prompts)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, x, y):
        out = self.lin(x)
        loss = ((out - y) ** 2).mean()
        return {"logits": out, "loss": loss, "aux_metrics": None}


class NaNModel(DummyModel):
    def forward(self, x, y):
        return {"logits": x, "loss": torch.tensor(float("nan")), "aux_metrics": None}


class AuxModel(DummyModel):
    def forward(self, x, y):
        out = self.lin(x)
        loss = ((out - y) ** 2).mean()
        return {
            "logits": out,
            "loss": loss,
            "aux_metrics": {"accuracy": torch.tensor(0.8)},
        }


class MoeModel(DummyModel):
    def forward(self, x, y):
        out = self.lin(x)
        loss = ((out - y) ** 2).mean()
        return {
            "logits": out,
            "loss": loss,
            "aux_metrics": {
                "block_0": {"moe": {"score": torch.tensor([1.0, 2.0]), "value": 42}}
            },
        }


@pytest.fixture
def config():
    return SimpleNamespace(
        trainer=SimpleNamespace(
            seed=42,
            precision="fp32",
            warmup_steps=0,
            max_steps=2,
            max_val_steps=0,
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

        def log_metrics(self, *a, **kw):
            pass

        def log_config(self, *a, **kw):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr("ai_playground.trainer.trainer.Generator", DummyGenerator)
    monkeypatch.setattr(
        "ai_playground.trainer.trainer.create_loggers",
        lambda *a, **kw: DummyLoggerManager(),
    )

    return Trainer(config, strategy)


@pytest.fixture(params=[DummyModel, AuxModel, MoeModel])
def model_class(request):
    return request.param


@pytest.fixture
def sample_data():
    xb = torch.randn(2, 16)
    yb = torch.randn(2, 16)
    return xb, yb


def test_train_step_runs(trainer, model_class, sample_data):
    model = model_class()
    trainer.configure_optimizer_and_scheduler(model)
    xb, yb = sample_data

    logits, loss, aux_metrics = trainer._train_step(
        model, xb, yb, trainer.optimizer, trainer.lr_scheduler
    )

    assert loss.item() > 0
    assert logits.shape == xb.shape
    if model_class is DummyModel:
        assert aux_metrics is None
    else:
        assert isinstance(aux_metrics, dict)


def test_nan_loss_raises(trainer, sample_data):
    model = NaNModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb, yb = sample_data
    with pytest.raises(RuntimeError):
        trainer._train_step(model, xb, yb, trainer.optimizer, trainer.lr_scheduler)


def test_validate_runs(trainer):
    model = DummyModel()
    data = [(torch.randn(2, 16), torch.randn(2, 16)) for _ in range(2)]
    val_loss = trainer._validate(model, data)

    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_scheduler_steps(trainer, sample_data):
    model = DummyModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb, yb = sample_data

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


def test_grad_check_flag(trainer, sample_data):
    model = DummyModel()
    trainer.configure_optimizer_and_scheduler(model)
    trainer.check_finite_grads = True
    xb, yb = sample_data

    # Should not raise
    trainer._train_step(model, xb, yb, trainer.optimizer, trainer.lr_scheduler)


def test_train_step_with_moe_logging(trainer, sample_data):
    captured_metrics = {}

    class DummyLoggerManager:
        log_frequency = 1

        def log_metrics(self, metrics, step=None):
            captured_metrics.update(metrics)

        def log_config(self, *a, **kw):
            pass

        def finalize(self):
            pass

    trainer.logger_manager = DummyLoggerManager()
    trainer.logger_metrics.add("moe")  # enable MoE logging
    model = MoeModel()
    trainer.configure_optimizer_and_scheduler(model)
    xb, yb = sample_data

    logits, loss, aux_metrics = trainer._train_step(
        model, xb, yb, trainer.optimizer, trainer.lr_scheduler
    )

    trainer._maybe_log(model, trainer.lr_scheduler, loss, aux_metrics)

    assert aux_metrics is not None
    assert any(
        "moe" in block_data
        for block_data in aux_metrics.values()
        if isinstance(block_data, dict)
    ), f"No MoE metrics in aux_metrics={aux_metrics}"
    assert any(
        "moe" in k or k.startswith("block_") for k in captured_metrics
    ), f"No MoE metrics logged. captured_metrics={captured_metrics}, aux_metrics={aux_metrics}"
