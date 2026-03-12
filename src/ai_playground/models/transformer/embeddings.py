import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.embedding(positions)


class EmbeddingWrapper(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionEmbedding(block_size, embed_dim)

    def forward(self, x):
        return self.token_embedding(x) + self.position_embedding(x)
