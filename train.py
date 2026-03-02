from configs.config import Config
from trainer import Trainer
from utils.utils import set_seed
from models.bigram import BiGram
from utils.utils import build_data_pipeline


def main():
    config = Config()
    set_seed(config.experimental.seed)
    device = config.experimental.device
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = BiGram(vocab_size=tokenizer.vocab_size, config=config).to(device)
    trainer = Trainer(config)
    trainer.fit(model, train_loader, val_loader)
    generated = trainer.generate(model, tokenizer, num_tokens=100)
    print(f"{generated=}")


if __name__ == "__main__":
    main()
