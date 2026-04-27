class Callback:
    def on_train_start(self, trainer):
        pass

    def on_train_step_end(self, trainer, loss, metrics):
        pass

    def on_validation_end(self, trainer, val_loss):
        pass

    def should_stop(self, trainer) -> bool:
        return False
