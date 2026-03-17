import torch.nn as nn

from .attention import MultiHeadAttention


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        block_size,
        hidden_dim,
        use_flash_attention,
        attn_dropout,
        residual_dropout,
        ffn_dropout,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            block_size=block_size,
            use_flash_attention=use_flash_attention,
            attn_droupout=attn_dropout,
            residual_droupout=residual_dropout,
        )
        self.ffn = FFN(embed_dim, hidden_dim, ffn_dropout)
        self.linear1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.LayerNorm(embed_dim)

    def forward(self, x, past_key_values=None, use_cache=False):
        # Attention + residual
        attn_out, present = self.attention(self.linear1(x), past_key_values, use_cache)
        # FFN + residual
        x = x + attn_out
        x = x + self.ffn(self.linear2(x))
        return x, present
