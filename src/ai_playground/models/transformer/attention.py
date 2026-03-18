import torch
import torch.nn as nn
import torch.nn.functional as F


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
        atten = atten.masked_fill(
            ~self.mask[:T, :T], float("-inf")
        )  # (B, T, T)   # type: ignore
        atten = torch.softmax(atten, dim=-1)  # (B, T, T) Softmax over the k dim
        return atten @ v  # (B, T, T) @ (B, T, h) -> (B, T, h)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        n_kv_head,
        block_size,
        use_flash_attention,
        attn_droupout,
        residual_droupout,
    ):
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.group_size = self.n_head // self.n_kv_head
        self.use_flash_attention = use_flash_attention
        self.last_attn = 0.0

        # Single qkv projection for all heads
        # self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        # GQA
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )
        self.scale = self.head_dim**-0.5

        self.attn_dropout = nn.Dropout(attn_droupout)
        self.resid_dropout = nn.Dropout(residual_droupout)

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        # qkv = self.qkv(x)  # (B, T, 3*embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q, k, v = qkv.chunk(3, dim=-1)  # (B, T, embed_dim) * 3

        # Reshape for multi-head attention
        # (B, T, embed_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # if using KV caching
        present = None

        if use_cache:
            cache = past_key_value

            if cache is not None:
                cache.append(k, v)
                k, v = cache.get_kv()
                present = cache

        # Expand kv heads
        k, v = self.expand_kv(k, v, self.group_size)
        # k = k.repeat_interleave(self.group_size, dim=1)
        # v = v.repeat_interleave(self.group_size, dim=1)

        # Compute attention scores
        # Training: T > 1
        # Prefill: T > 1
        # Decode: T == 1
        if self.use_flash_attention:
            atten = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(T > 1),
            )
        else:
            atten = (
                torch.matmul(q, k.transpose(-2, -1)) * self.scale
            )  # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
            key_len = k.size(2)

            if T > 1:
                atten = atten.masked_fill(~self.mask[:T, :key_len], float("-inf"))  # type: ignore
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
        out = self.resid_dropout(out)

        return out, present

    def expand_kv(self, k, v, group_size):
        B, H_kv, T, D = k.shape

        k = k[:, :, None, :, :]  # (B, H_kv, 1, T, D)
        v = v[:, :, None, :, :]

        k = k.expand(B, H_kv, group_size, T, D)
        v = v.expand(B, H_kv, group_size, T, D)

        k = k.contiguous().view(B, H_kv * group_size, T, D)
        v = v.contiguous().view(B, H_kv * group_size, T, D)

        return k, v
