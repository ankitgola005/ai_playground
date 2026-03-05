from configs.gpt_config import GPTConfig
from trainer import Trainer
from models.miniGPT import MiniGPT
from utils.utils import build_data_pipeline


def main():
    config = GPTConfig()
    device = config.experimental.device
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = MiniGPT(tokenizer.vocab_size, config=config).to(device)
    trainer = Trainer(config)
    trainer.fit(model, train_loader, val_loader)
    # trainer.load_checkpoint(model)
    generated = trainer.predict(
        model, tokenizer, prompts=["What is capital of France"], max_tokens=100
    )
    print(f"{generated=}")


if __name__ == "__main__":
    main()
