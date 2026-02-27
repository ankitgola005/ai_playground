class CharTokenizer:
    # A simple character level tokenizer
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])
