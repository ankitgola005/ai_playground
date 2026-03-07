from ai_playground.configs.gpt_config import GPTConfig
from ai_playground.utils.utils import build_data_pipeline, build_model
from ai_playground.distributed import ddp
from ai_playground.trainer import Trainer

def run_training(config):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=ddp.DDParallel())
    trainer.fit(model, train_loader, val_loader)


def main():
    config = GPTConfig()
    strategy = ddp.DDParallel(
        device="cpu",
        num_devices=2,
    )
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()