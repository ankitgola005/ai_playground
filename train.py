from ai_playground.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.config import get_config
from ai_playground.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import Config


def run_training(config: Config):
    tokenizer, train_loader, val_loader = build_data_pipeline(
        config.data, config.trainer.batch_size, config.trainer.seed
    )
    model = build_model(config.model)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    trainer.fit(model, train_loader, val_loader)
    trainer.predict(model, tokenizer, prompts=["The quickest fox is "])


def main():
    config: Config = get_config("gpt_config.yaml")
    strategy = get_strategy(config.distributed)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
