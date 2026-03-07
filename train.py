from ai_playground.configs.gpt_config import GPTConfig
from ai_playground.utils.utils import build_data_pipeline, build_model
from ai_playground.distributed import ddp, single
from ai_playground.trainer.trainer import Trainer


def run_training(config):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=ddp.DDParallel(config))
    trainer.fit(model, train_loader, val_loader)


def main():
    config = GPTConfig()
    strategy = ddp.DDParallel(config)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
