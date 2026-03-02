from configs.gpt_config import GPTConfig
from trainer import Trainer
from utils.utils import set_seed
from models.miniGPT import MiniGPT
from utils.utils import build_data_pipeline


def main():
    config = GPTConfig()
    set_seed(config.experimental.seed)
    device = config.experimental.device
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = MiniGPT(tokenizer.vocab_size, config=config).to(device)
    trainer = Trainer(config)
    trainer.fit(model, train_loader, val_loader)
    generated = trainer.generate(model, tokenizer, num_tokens=100)
    print(f"{generated=}")


if __name__ == "__main__":
    main()
