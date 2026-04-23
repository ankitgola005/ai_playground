from ai_playground.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.config import get_config
from ai_playground.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import Config


def run_training(config: Config):
    print("Building data pipeline...")
    tokenizer, train_loader, val_loader = build_data_pipeline(
        config.data, config.trainer.batch_size, config.trainer.seed
    )
    print("vocab_size:", tokenizer.vocab_size)
    print("Built data pipeline")
    print("Building model...")
    model = build_model(config.model)(
        config.model, tokenizer.vocab_size, config.data.block_size, True
    )
    print("Built model")
    print("Building Trainer...")
    trainer = Trainer(
        config, strategy=get_strategy(config.distributed), logger_metrics=["moe"]
    )
    print("Built Trainer")
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Finished training")
    preds = trainer.predict(model, tokenizer, prompts=["The quickest fox is "])
    print(f"{preds=}")


def main():
    config: Config = get_config("minigpt_config.yaml")
    strategy = get_strategy(config.distributed)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
