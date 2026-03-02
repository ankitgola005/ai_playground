import torch.nn as nn
from torch.nn import functional as F

from transformer.embeddings import EmbeddingWrapper
from transformer.transformer import TransformerBlock

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.gpt_config import GPTConfig


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, config: GPTConfig):
        super().__init__()

        # Embeddings
        self.embeddings = EmbeddingWrapper(
            vocab_size, config.model.block_size, config.model.n_embed
        )

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config.model.n_embed,
                    config.model.n_head,
                    config.model.block_size,
                    config.model.hidden_dim,
                )
                for _ in range(config.model.n_layer)
            ]
        )

        # Final Layernorm + Linear head
        self.layernorm = nn.LayerNorm(config.model.n_embed)
        self.head = nn.Linear(config.model.n_embed, vocab_size, bias=False)

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
