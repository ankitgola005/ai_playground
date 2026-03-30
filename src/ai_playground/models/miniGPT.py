import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import TYPE_CHECKING

from ai_playground.models.transformer.embeddings import EmbeddingWrapper
from ai_playground.models.transformer.transformer import TransformerBlock
from ai_playground.inference.cache import KVCache, PagedKVCache

if TYPE_CHECKING:
    from typing import Optional, List, Tuple
    from ai_playground.configs import ModelConfig


class MiniGPT(nn.Module):
    """
    MiniGPT: a small GPT-style model with optional KV caching.

    Attributes:
        embeddings (EmbeddingWrapper): Token and positional embeddings.
        transformer_blocks (nn.Sequential): Stack of Transformer blocks.
        layernorm (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Linear head projecting to vocab size.
        use_kv_cache (bool): Whether to use KV caching.
        kv_cache_max_len (int): Maximum KV cache length.
        use_paged_kv_cache (bool): Whether to use paged KV cache.
        paged_kv_cache_block_size (int): Block size for paged KV cache.
    """

    def __init__(self, model_config: "ModelConfig", vocab_size: int, block_size: int):
        """
        Initialize MiniGPT.

        Args:
            vocab_size (int): Size of the vocabulary.
            config (ConfigProtocol): Configuration object with model hyperparameters.
        """
        super().__init__()
        self.model_config = model_config
        self.block_size = block_size
        # Embeddings
        self.embeddings: EmbeddingWrapper = EmbeddingWrapper(
            vocab_size,
            self.block_size,
            int(self.model_config.model_kwargs["n_embed"]),
        )

        # Transformer blocks
        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=int(self.model_config.model_kwargs["n_embed"]),
                    n_head=int(self.model_config.model_kwargs["n_head"]),
                    n_kv_head=int(self.model_config.model_kwargs["n_kv_head"]),
                    block_size=self.block_size,
                    hidden_dim=int(self.model_config.model_kwargs["hidden_dim"]),
                    use_flash_attention=bool(
                        self.model_config.model_kwargs["use_flash_attention"]
                    ),
                    attn_dropout=float(self.model_config.model_kwargs["attn_dropout"]),
                    residual_dropout=float(
                        self.model_config.model_kwargs["residual_dropout"]
                    ),
                    ffn_dropout=float(self.model_config.model_kwargs["ffn_dropout"]),
                )
                for _ in range(int(self.model_config.model_kwargs["n_layer"]))
            ]
        )

        # Final LN + Linear head
        self.layernorm: nn.LayerNorm = nn.LayerNorm(
            int(self.model_config.model_kwargs["n_embed"])
        )
        self.head: nn.Linear = nn.Linear(
            int(self.model_config.model_kwargs["n_embed"]), vocab_size, bias=False
        )

        # KV Cache settings
        self.use_kv_cache: bool = bool(self.model_config.model_kwargs["use_kv_cache"])
        self.kv_cache_max_len: int = int(
            self.model_config.model_kwargs["kv_cache_max_len"]
        )
        self.use_paged_kv_cache: bool = (
            bool(self.model_config.model_kwargs["use_paged_kv_cache"])
            and self.use_kv_cache
        )
        self.paged_kv_cache_block_size: int = int(
            self.model_config.model_kwargs["paged_kv_cache_block_size"]
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """
        Forward pass of MiniGPT.

        Args:
            idx (torch.Tensor): Input token indices (B, T).
            targets (Optional[torch.Tensor]): Target token indices (B, T), for loss computation.
            past_key_values (Optional[List]): Cached KV pairs from previous tokens.
            use_cache (bool): Whether to store KV caches.

        Returns:
            logits (torch.Tensor): Predicted logits (B, T, C).
            loss (Optional[torch.Tensor]): Cross-entropy loss if targets provided.
            new_past_key_values (Optional[List]): Updated KV cache if use_cache=True.
        """
        # Embeddings
        x: torch.Tensor = self.embeddings(idx)

        # Transformer blocks
        new_past_key_values: List = []
        for i, block in enumerate(self.transformer_blocks):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_values=pkv, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present)

        # Final LN + head
        x = self.layernorm(x)
        logits: torch.Tensor = self.head(x)

        # Loss
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_past_key_values if use_cache else None

    def init_kv_cache(self, B: int, device: str = "cuda") -> List:
        """
        Initialize KV cache for all transformer blocks.

        Args:
            B (int): Batch size.
            device (str): Device to place the cache on.

        Returns:
            List: List of KVCache or PagedKVCache instances, one per block.
        """
        assert (
            self.use_kv_cache
        ), "KV cache initialization requested, but use_kv_cache=False."

        cache_cls = PagedKVCache if self.use_paged_kv_cache else KVCache
        _size = (
            self.kv_cache_max_len
            if not self.use_paged_kv_cache
            else self.paged_kv_cache_block_size
        )

        caches: List = []
        for block in self.transformer_blocks:
            caches.append(
                cache_cls(
                    B,
                    int(self.model_config.model_kwargs["n_kv_head"]),
                    int(self.model_config.model_kwargs["n_embed"])
                    // int(self.model_config.model_kwargs["n_head"]),
                    _size,
                    torch.device(device),
                    next(self.parameters()).dtype,
                )
            )
        return caches
