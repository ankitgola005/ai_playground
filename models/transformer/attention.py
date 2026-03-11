import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        super(SelfAttention, self).__init__()
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )
        self.scale = head_dim**-0.5

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, embed_dim) -> (B, T, head_dim)
        q = self.query(x)  # (B, T, embed_dim) -> (B, T, head_dim)
        v = self.value(x)  # (B, T, embed_dim) -> (B, T, head_dim)

        atten = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (B, T, h) @ (B, h, T) -> (B, T, T)
        atten = atten.masked_fill(~self.mask[:T, :T], float("-inf"))  # (B, T, T)
        atten = torch.softmax(atten, dim=-1)  # (B, T, T) Softmax over the k dim
        return atten @ v  # (B, T, T) @ (B, T, h) -> (B, T, h)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, block_size, attn_droupout, residual_droupout):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_head
        self.n_head = n_head
        self.last_attn = 0.0

        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"

        # Single qkv projection for all heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )
        self.scale = self.head_dim**-0.5

        self.attn_dropout = nn.Dropout(attn_droupout)
        self.resid_dropout = nn.Dropout(residual_droupout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, T, embed_dim) * 3

        # Reshape for multi-head attention
        # (B, T, embed_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores
        atten = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        atten = atten.masked_fill(~self.mask[:T, :T], float("-inf"))
        atten = torch.softmax(atten, dim=-1)  # (B, n_head, T, T)
        self.last_attn = atten
        atten = self.attn_dropout(atten)  # (B, n_head, T, T)
        atten = torch.matmul(
            atten, v
        )  # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)

        # Concatenate heads and project
        atten = (
            atten.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, T, n_head*head_dim) -> (B, T, embed_dim)
        out = self.out_proj(atten)  # (B, T, embed_dim) -> (B, T, embed_dim)
        return self.resid_dropout(out)
