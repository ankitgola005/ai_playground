import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
import torch

from ai_playground.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.trainer import Trainer

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol
    import torch.nn as nn
    from typing import Callable, List


def get_seq(bm: str = "perf"):
    seq_lens = []
    seq_lens += list(range(1, 33281, 512))
    if bm == "perf":
        seq_lens.reverse()
    seq_lens = [128]
    return seq_lens


def benchmark(
    trainer: "Trainer",
    model: "nn.Module",
    tokenizer,
    seq_lens: list,
    use_cache: bool,
    metric_fn: Callable[[Trainer], dict],
    warmup: int = 10,
):
    """
    Generic benchmark for prefill/decode or memory tracking.

    metric_fn: function that receives trainer and returns a dict of metrics
    """

    metrics: dict["str", List] = {"ctx_len": []}

    # Warmup
    while warmup > 0:
        prompt = "a" * warmup
        tokens = tokenizer.encode(prompt)
        trainer.predict(
            model,
            tokenizer,
            prompts=[prompt],
            max_tokens=100,
            use_cache=use_cache,
            max_cache_len=len(tokens) + 100,
        )
        warmup -= 1

    # Benchmark
    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        prompt = "a" * seq_len
        tokens = tokenizer.encode(prompt)
        trainer.predict(
            model,
            tokenizer,
            prompts=[prompt],
            max_tokens=100,
            use_cache=use_cache,
            max_cache_len=len(tokens) + 100,
        )
        metric_data = metric_fn(trainer)
        metrics["ctx_len"].append(seq_len)
        for key, val in metric_data.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)

    return metrics


# Metric functions
def time_metrics(trainer: "Trainer"):
    """Return prefill and decode times"""
    return {
        "prefill_time": trainer.generator.time_dict["prefill_time"],
        "decode_time": trainer.generator.time_dict["decode_time"],
    }


def memory_metrics(trainer: "Trainer"):
    """Return peak GPU memory usage in MB"""
    return {"gpu_mem_mb": torch.cuda.max_memory_allocated() / (1024**2)}


def run_training(config: "ConfigProtocol", metric_fn: Callable, filename: str):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config.model)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    seq_lens = get_seq(bm="perf")
    no_kv_metrics = benchmark(
        trainer, model, tokenizer, seq_lens, use_cache=False, metric_fn=metric_fn
    )
    seq_lens = get_seq(bm="mem")
    kv_metrics = benchmark(
        trainer, model, tokenizer, seq_lens, use_cache=True, metric_fn=metric_fn
    )

    plot_metrics(kv_metrics, no_kv_metrics, filename)


def plot_metrics(metrics_kv, metrics_nokv, filename: str):
    plt.figure(figsize=(12, 6))

    ctx = metrics_kv["ctx_len"]
    keys = [k for k in metrics_kv.keys() if k != "ctx_len"]

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "d"]

    for i, key in enumerate(keys):
        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        plt.plot(
            ctx,
            metrics_kv[key],
            color="red",
            linestyle=style,
            marker=marker,
            label=f"{key} (KV cache)",
        )

        plt.plot(
            ctx,
            metrics_nokv[key],
            color="blue",
            linestyle=style,
            marker=marker,
            label=f"{key} (No KV cache)",
        )

    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Value")
    plt.title("Benchmark Metrics With/Without KV Cache")
    plt.grid(True)
    plt.legend()

    plt.savefig(filename, dpi=300)
    plt.show()


def main():
    config: "ConfigProtocol" = load_yaml_config("gpt_config.yaml")  # type: ignore
    strategy = get_strategy(config.distributed)

    # --- Timing benchmark ---
    strategy.launch(
        run_training,
        config,
        metric_fn=time_metrics,
        filename="prefill_decode_times.png",
    )

    # --- Memory benchmark ---
    strategy.launch(
        run_training, config, metric_fn=memory_metrics, filename="memory_usage.png"
    )


if __name__ == "__main__":
    main()
