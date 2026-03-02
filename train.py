import torch
from data.char_tokenizer import CharTokenizer
from data import dataset
from configs.config import Config
from trainer import Trainer
from utils.utils import set_seed
from models.bigram import BiGram
from utils.logger import Logger


def main():
    config = Config()
    set_seed(config.experimental.seed)
    device = config.experimental.device

    with open(config.data.data_path, "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = dataset.train_val_split(encoded, split=config.data.split)
    train_loader = dataset.build_dataloader(config, train_data)
    val_loader = dataset.build_dataloader(config, val_data)

    model = BiGram(vocab_size=tokenizer.vocab_size, config=config).to(device)
    trainer = Trainer(config)
    trainer.fit(model, train_loader, val_loader)
    generated = trainer.generate(model, tokenizer, num_tokens=100)
    print(f"{generated=}")


if __name__ == "__main__":
    main()
