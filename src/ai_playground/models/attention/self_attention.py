import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Shape:
        Input: (B, T, C)
        Output: (B, T, head_dim)
    """

    def __init__(self, embed_dim: int, head_dim: int, block_size: int) -> None:
        """Initialize self attention.

        Args:
            embed_dim (int): Input embedding dimension.
            head_dim (int): Dimension of the attention head.
            block_size (int): Maximum sequence length (used for causal masking).
        """
        super().__init__()
        self.key: nn.Linear = nn.Linear(embed_dim, head_dim, bias=False)
        self.query: nn.Linear = nn.Linear(embed_dim, head_dim, bias=False)
        self.value: nn.Linear = nn.Linear(embed_dim, head_dim, bias=False)

        # Causal mask: upper triangular positions are False
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )
        self.scale: float = head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_dim)
        """
        B, T, C = x.shape
        k: torch.Tensor = self.key(x)  # (B, T, head_dim)
        q: torch.Tensor = self.query(x)  # (B, T, head_dim)
        v: torch.Tensor = self.value(x)  # (B, T, head_dim)

        # Compute attention scores
        attn_scores: torch.Tensor = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (B, T, T)
        attn_scores = attn_scores.masked_fill(
            ~self.mask[:T, :T], float("-inf")  # type: ignore
        )  # causal masking
        attn_probs: torch.Tensor = torch.softmax(
            attn_scores, dim=-1
        )  # softmax over key dimension

        # Weighted sum of values
        out: torch.Tensor = torch.matmul(attn_probs, v)  # (B, T, head_dim)
        return out
