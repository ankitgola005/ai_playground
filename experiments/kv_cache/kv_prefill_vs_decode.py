import time
import torch

from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.runner.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def benchmark_prefill_decode(trainer, model, tokenizer):
    seq_lens = [16, 32, 64, 128, 256]
    max_new_tokens = 50
    results = []

    for seq_len in seq_lens:
        prompt = " ".join(["hello"] * seq_len)
        torch.cuda.synchronize()
        start = time.perf_counter()

        trainer.predict(
            model,
            tokenizer,
            prompts=[prompt],
            max_tokens=max_new_tokens,
            use_cache=True,
            max_cache_len=seq_len + max_new_tokens,
        )

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        results.append((seq_len, total_time))

        print(f"prompt_len={seq_len:4d} | total_time={total_time:.4f}s")

    print("\nSummary")
    print("-----------------------------------")
    for seq_len, t in results:
        print(f"{seq_len:4d} tokens -> {t:.4f}s")

    return results


def run_training(config: "ConfigProtocol"):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    trainer.fit(model, train_loader, val_loader)
    benchmark_prefill_decode(trainer, model, tokenizer)


def main():
    config: "ConfigProtocol" = load_yaml_config("gpt_config.yaml")  # type: ignore
    strategy = get_strategy(config.distributed)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
