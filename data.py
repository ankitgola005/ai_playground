import torch
from pathlib import Path


class Tokenizer:
    # A simple character level tokenizer
    def __init__(self, dictionary):
        self.stoi = {ch: i for i, ch in enumerate(dictionary)}
        self.itos = {i: ch for i, ch in enumerate(dictionary)}

    def encode(self, chars):
        return [self.stoi[c] for c in chars]

    def decode(self, enc_char):
        return "".join([self.itos[i] for i in enc_char])


class Data:
    # Load, tokenize, etc
    def __init__(self, config, fpath="input.txt", split=0.9, device="cpu", rng=42):
        torch.manual_seed(rng)
        self.device = device
        self.block_size = config.block_size
        self.batch_size = config.batch_size

        self.fpath = fpath
        self.split = split
        raw_data = sorted(list(set(self.get_data())))
        self.tokenizer = Tokenizer(dictionary=raw_data)
        self.vocab_size = len(raw_data)
        self.data = torch.tensor(
            self.tokenizer.encode(raw_data), dtype=torch.long, device=self.device
        )
        self.train_data = None
        self.test_data = None
        if self.data is not None:
            self.train_data, self.test_data = self.train_test_split()

    def get_data(self):
        if Path.is_file(self.fpath):
            with open(self.fpath, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def train_test_split(self):
        # Return train data, test data
        n = int(self.split * len(self.data))
        return self.data[:n], self.data[n:]

    def get_batch(self, train="train"):
        data = self.train_data if train == "train" else self.test_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
