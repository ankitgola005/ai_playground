from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.runner.trainer import Trainer
import torch
import time
from torch.cuda import memory_allocated, max_memory_allocated
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def run_training(config: ConfigProtocol):
    # Build data, model, trainer
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))

    # Train the model
    trainer.fit(model, train_loader, val_loader)


def run_infer(
    config: "ConfigProtocol",
    prompts: list[str] = [],
    max_tokens: int = 100,
    use_cache: bool = False,
    results: dict[str, list] = {},
):
    # Build data, model, trainer
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))

    model.eval()

    # Reset GPU memory stats
    if trainer.strategy.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    # Use Trainer.predict in baseline mode (no KV cache)
    # By default, predict generates without past_key_values
    print(f"{use_cache=}")
    output = trainer.predict(
        model, tokenizer, prompts=prompts, max_tokens=max_tokens, use_cache=use_cache
    )

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    tokens_generated = len(prompts) * max_tokens
    tps = tokens_generated / elapsed

    results["time"].append(elapsed)
    results["tps"].append(tps)
    results["allocated"].append(memory_allocated() / 1024**2)
    results["peak_memory"].append(max_memory_allocated() / 1024**2)
    results["prompt"].append(prompts[0])
    results["generated"].append(output[0])


def main():
    config: "ConfigProtocol" = load_yaml_config("gpt_config.yaml")  # type: ignore
    strategy = get_strategy(config.distributed)

    # Train or load benchmark
    # strategy.launch(run_training, config)

    # Run inference
    max_tokens = 192
    prompt = "hello" * max_tokens
    num_runs = 5
    warmup = int(num_runs * 0.1)
    kv_cache = [True, False]

    while warmup > 0:
        result: dict[str, list] = {
            "time": [],
            "tps": [],
            "allocated": [],
            "peak_memory": [],
            "prompt": [],
            "generated": [],
        }
        strategy.launch(run_infer, config, prompt, max_tokens, False, result)
        warmup -= 1

    for cache_status in kv_cache:
        result: dict[str, list] = {
            "time": [],
            "tps": [],
            "allocated": [],
            "peak_memory": [],
            "prompt": [],
            "generated": [],
        }
        for i in range(num_runs):
            print(f"Starting run {i}")
            strategy.launch(run_infer, config, prompt, max_tokens, cache_status, result)
            print(f"Finishing run {i}")

        print(f"{cache_status=}")
        for key, values in result.items():
            if key in ["time", "tps", "allocated", "peak_memory"]:
                print(f"Averatge {key}: {sum(values) / len(values)}")


if __name__ == "__main__":
    main()
