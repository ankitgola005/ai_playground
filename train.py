from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.runner.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def run_training(config: ConfigProtocol):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    trainer.fit(model, train_loader, val_loader)
    trainer.predict(model, tokenizer, prompts=["The quickest fox is "])


def main():
    config: ConfigProtocol = load_yaml_config("gpt_config.yaml")  # type: ignore
    strategy = get_strategy(config.distributed)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
