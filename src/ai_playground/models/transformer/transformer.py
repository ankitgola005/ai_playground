import torch.nn as nn
from typing import TYPE_CHECKING

from ai_playground.models.transformer.attention import MultiHeadAttention

if TYPE_CHECKING:
    import torch
    from typing import Optional, Tuple, List


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
        attn_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
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
        self.ffn = FFN(embed_dim, hidden_dim, ffn_dropout)
        self.linear1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: List | None = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
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
        x = x + self.ffn(self.linear2(x))
        return x, present
