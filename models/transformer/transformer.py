import torch.nn as nn

from .attention import MultiHeadAttention


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_head, block_size, hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_head, block_size)
        self.ffn = FFN(embed_dim, hidden_dim)
        self.linear1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention + residual
        x = x + self.attention(self.linear1(x))
        # FFN + residual
        x = x + self.ffn(self.linear2(x))
        return x
