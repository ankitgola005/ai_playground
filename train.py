import torch
from data.char_tokenizer import CharTokenizer
from data import dataset
from config import Config
from trainer import Trainer
from utils.utils import set_seed
from models.bigram import BiGram
from utils.logger import Logger


def main():
    config = Config()
    set_seed(config.seed)

    cuda = False  # torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    with open("data/datasets/text_datasets/shakespeare.txt", "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_loader = dataset.build_dataloader(config, encoded)

    model = BiGram(vocab_size=tokenizer.vocab_size, config=config).to(device)
    trainer = Trainer(logger=Logger(), max_steps=config.max_steps)
    trainer.fit(model, train_loader)
    generated = trainer.generate(model, tokenizer, num_tokens=100)
    print(f"{generated=}")


if __name__ == "__main__":
    main()
