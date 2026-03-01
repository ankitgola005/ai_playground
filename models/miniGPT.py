from xml.parsers.expat import model

import torch.nn as nn
from torch.nn import functional as F

from transformer.embeddings import EmbeddingWrapper
from transformer.transformer import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self, vocab_size, block_size, embed_dim, n_heads, hidden_dim, n_layers
    ):
        super().__init__()

        # Embeddings
        self.embeddings = EmbeddingWrapper(vocab_size, block_size, embed_dim)

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, n_heads, block_size, hidden_dim)
                for _ in range(n_layers)
            ]
        )

        # Final Layernorm + Linear head
        self.layernorm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.transformer_blocks(x)
        x = self.layernorm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
