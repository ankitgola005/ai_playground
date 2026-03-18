import torch.nn as nn
from torch.nn import functional as F

from ai_playground.models.transformer.embeddings import EmbeddingWrapper
from ai_playground.models.transformer.transformer import TransformerBlock
from ai_playground.models.transformer.kv_cache import KVCache, PagedKVCache

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, config: ConfigProtocol):
        super().__init__()

        self.config = config

        # Embeddings
        self.embeddings = EmbeddingWrapper(
            vocab_size,
            self.config.model.model_kwargs["block_size"],
            self.config.model.model_kwargs["n_embed"],
        )

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=self.config.model.model_kwargs["n_embed"],
                    n_head=self.config.model.model_kwargs["n_head"],
                    n_kv_head=self.config.model.model_kwargs["n_kv_head"],
                    block_size=self.config.model.model_kwargs["block_size"],
                    hidden_dim=self.config.model.model_kwargs["hidden_dim"],
                    use_flash_attention=self.config.model.model_kwargs[
                        "use_flash_attention"
                    ],
                    attn_dropout=self.config.model.model_kwargs["attn_dropout"],
                    residual_dropout=self.config.model.model_kwargs["residual_dropout"],
                    ffn_dropout=self.config.model.model_kwargs["ffn_dropout"],
                )
                for _ in range(self.config.model.model_kwargs["n_layer"])
            ]
        )

        # Final Layernorm + Linear head
        self.layernorm = nn.LayerNorm(self.config.model.model_kwargs["n_embed"])
        self.head = nn.Linear(
            self.config.model.model_kwargs["n_embed"], vocab_size, bias=False
        )

        # KV Cache
        self.use_kv_cache = self.config.model.model_kwargs["use_kv_cache"]
        self.kv_cache_max_len = self.config.model.model_kwargs["kv_cache_max_len"]
        self.use_paged_kv_cache = (
            self.config.model.model_kwargs["use_paged_kv_cache"] and self.use_kv_cache
        )
        self.paged_kv_cache_block_size = self.config.model.model_kwargs[
            "paged_kv_cache_block_size"
        ]

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        # Embeddings
        x = self.embeddings(idx)

        # Transformer blocks
        new_past_key_values = []
        for i, block in enumerate(self.transformer_blocks):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_values=pkv, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present)

        # Final LN + head
        x = self.layernorm(x)
        logits = self.head(x)

        # Loss
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_past_key_values if use_cache else None

    def init_kv_cache(self, B: int, max_len: int = 1024, device: str = "cuda"):
        assert self.use_kv_cache, "Trying to init KV cache, but `use_kv_cache=False`"

        cache_impl = None
        _size = 0
        if self.use_paged_kv_cache:
            cache_impl = PagedKVCache
            _size = self.kv_cache_max_len
        else:
            cache_impl = KVCache
            _size = self.paged_kv_cache_block_size
        caches = []

        for block in self.transformer_blocks:
            caches.append(
                cache_impl(
                    B,
                    self.config.model.model_kwargs["n_kv_head"],
                    self.config.model.model_kwargs["n_embed"]
                    // self.config.model.model_kwargs["n_head"],
                    _size,
                    device,
                    next(self.parameters()).dtype,
                )
            )
        return caches
