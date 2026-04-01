import torch.nn as nn
from typing import TYPE_CHECKING

from ai_playground.models.attention import MultiHeadAttention
from ai_playground.models.moe import MoELayer

if TYPE_CHECKING:
    import torch
    from typing import Tuple, List, Dict, Any


class FFN(nn.Module):
    """
    Feedforward network.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        """
        Initialize FFN

        Args:
            embed_dim (int): Input/output embedding dimension.
            hidden_dim (int): Hidden layer dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, T, embed_dim).

        Returns:
            Tensor: Output tensor of shape (B, T, embed_dim).
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer block with MHA and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        n_kv_head: int,
        block_size: int,
        hidden_dim: int,
        use_flash_attention: bool,
        num_experts: int,
        attn_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        moe_dropout: float,
        sparse_selector: str | None = "topk",
        logging: bool = False,
    ):
        """
        Initialize a Transformer block.

        Args:
            embed_dim (int): Embedding dimension.
            n_head (int): Number of attention heads.
            n_kv_head (int): Number of key-value heads (can be < n_head for efficiency).
            block_size (int): Maximum sequence length.
            hidden_dim (int): Hidden layer dimension in FFN.
            use_flash_attention (bool): Whether to use flash attention.
            attn_dropout (float): Dropout probability for attention.
            residual_dropout (float): Dropout probability for residual connections.
            ffn_dropout (float): Dropout probability for feedforward network.
            sparse_selector (str): Select sparsity method (topk / strided).
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            block_size=block_size,
            use_flash_attention=use_flash_attention,
            attn_droupout=attn_dropout,
            residual_droupout=residual_dropout,
        )
        if sparse_selector == "topk":
            from ai_playground.models.attention.sparse_selector import TopKSelector

            self.attention.set_sparse_selector(TopKSelector(1))
        elif sparse_selector == "strided":
            from ai_playground.models.attention.sparse_selector import StrideSelector

            self.attention.set_sparse_selector(StrideSelector(2))

        self.num_experts = num_experts
        if self.num_experts <= 0:
            self.ffn = FFN(embed_dim, hidden_dim, ffn_dropout)
        else:
            self.ffn = MoELayer(
                d_model=embed_dim,
                d_ff=hidden_dim,
                num_experts=self.num_experts,
                dropout=moe_dropout,
                logging=logging,
            )

        self.linear1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: List | None = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List | None, Dict | None]:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, T, embed_dim).
            past_key_values (Optional[list]): Cached KV tensors from previous steps.
            use_cache (bool): Whether to return updated KV cache.

        Returns:
            x (Tensor): Output tensor of shape (B, T, embed_dim).
            present (Optional[list]): Updated KV cache if use_cache=True, else None.
        """
        # Attention + residual
        attn_out, present = self.attention(self.linear1(x), past_key_values, use_cache)
        x = x + attn_out

        # FFN + residual
        aux_stats: Dict[str, Any] = {}
        if self.num_experts <= 0:
            x = self.ffn(self.linear2(x))
        else:
            x, aux_stats["moe"] = self.ffn(self.linear2(x))

        return x, present, aux_stats
