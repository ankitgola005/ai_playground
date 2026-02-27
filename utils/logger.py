import time
from collections import defaultdict


class Logger:
    def __init__(self, log_interval=10) -> None:
        self.log_interval = log_interval
        self.start_time = time.time()
        self.log_metrics = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.log_metrics[key] += value
            self.count[key] += 1

    def log(self, step):
        if step % self.log_interval != 0:
            return
        elapsed_time = time.time() - self.start_time
        log_str = f"Step {step} | Elapsed Time: {elapsed_time:.2f}s | "
        for key in self.log_metrics:
            avg_value = self.log_metrics[key] / self.count[key]
            log_str += f"{key}: {avg_value:.4f} | "
        print(log_str)
        self.reset()

    def reset(self):
        self.log_metrics.clear()
        self.count.clear()
