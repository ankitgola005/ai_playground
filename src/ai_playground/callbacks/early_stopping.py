from ai_playground.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_counts = 0

    def on_validation_end(self, trainer, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_counts = 0
        else:
            self.bad_counts += 1

    def should_stop(self, trainer) -> bool:
        return self.bad_counts >= self.patience
