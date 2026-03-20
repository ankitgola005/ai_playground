import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input token indices of shape (B, T).

        Returns:
            Tensor: Embedded tokens of shape (B, T, embed_dim).
        """
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    """
    Positional embedding layer.
    """

    def __init__(self, block_size: int, embed_dim: int):
        """
        Args:
            block_size (int): Maximum sequence length.
            embed_dim (int): Dimensionality of embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input token indices of shape (B, T).

        Returns:
            Tensor: Positional embeddings of shape (B, T, embed_dim).
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.embedding(positions)


class EmbeddingWrapper(nn.Module):
    """
    Wrapper combining token and positional embeddings.
    """

    def __init__(self, vocab_size: int, block_size: int, embed_dim: int):
        """
        Args:
            vocab_size (int): Vocabulary size.
            block_size (int): Maximum sequence length.
            embed_dim (int): Embedding dimensionality.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionEmbedding(block_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input token indices of shape (B, T).

        Returns:
            Tensor: Combined token + positional embeddings (B, T, embed_dim).
        """
        return self.token_embedding(x) + self.position_embedding(x)
