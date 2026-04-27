# tests/test_early_stopping.py

from ai_playground.callbacks import EarlyStopping


class DummyTrainer:
    def __init__(self):
        self.global_step = 0


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=3, min_delta=0.0)
    trainer = DummyTrainer()

    losses = [1.0, 0.9, 0.9, 0.9, 0.9]  # improvement stops

    for loss in losses:
        es.on_validation_end(trainer, loss)

    assert es.should_stop(trainer) is True


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, min_delta=0.0)
    trainer = DummyTrainer()

    losses = [1.0, 0.9, 0.95, 0.85]  # improvement again

    for loss in losses:
        es.on_validation_end(trainer, loss)

    assert es.should_stop(trainer) is False


def test_early_stopping_min_delta_respected():
    es = EarlyStopping(patience=2, min_delta=0.1)
    trainer = DummyTrainer()

    losses = [1.0, 0.95, 0.94, 0.93]  # small improvements ignored

    for loss in losses:
        es.on_validation_end(trainer, loss)

    assert es.should_stop(trainer) is True
