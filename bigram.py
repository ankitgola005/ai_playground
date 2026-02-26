import torch
import torch.nn as nn
from torch.nn import functional as F


class BiGram(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.vocab_size)

    def forward(self, idx, targets=None):
        logits = self.tok_emb(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=1):
        # idx is (B, T) array
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # B, C, focus only on last timestamp
            probs = F.softmax(logits, dim=-1)  # B, C, get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
