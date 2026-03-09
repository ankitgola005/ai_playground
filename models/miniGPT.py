import torch.nn as nn
from torch.nn import functional as F

from .transformer.embeddings import EmbeddingWrapper
from .transformer.transformer import TransformerBlock

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, config: ConfigProtocol):
        super().__init__()

        # Embeddings
        self.embeddings = EmbeddingWrapper(
            vocab_size,
            config.model.model_kwargs["block_size"],
            config.model.model_kwargs["n_embed"],
        )

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config.model.model_kwargs["n_embed"],
                    config.model.model_kwargs["n_head"],
                    config.model.model_kwargs["block_size"],
                    config.model.model_kwargs["hidden_dim"],
                    config.model.model_kwargs["attn_dropout"],
                    config.model.model_kwargs["residual_dropout"],
                    config.model.model_kwargs["ffn_dropout"],
                )
                for _ in range(config.model.model_kwargs["n_layer"])
            ]
        )

        # Final Layernorm + Linear head
        self.layernorm = nn.LayerNorm(config.model.model_kwargs["n_embed"])
        self.head = nn.Linear(
            config.model.model_kwargs["n_embed"], vocab_size, bias=False
        )

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
