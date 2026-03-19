import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.inference.cache.base_kv_cache import BaseKVCache


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
    """
    Multi-Head Attention module with support for:
    - Grouped Query Attention (GQA)
    - Flash Attention (PyTorch SDPA)
    - KV caching (standard + paged/blockwise)
    - Streaming decode attention

    Args:
        embed_dim (int): Model embedding dimension (C)
        n_head (int): Number of query heads (H_q)
        n_kv_head (int): Number of key/value heads (H_kv)
        block_size (int): Maximum sequence length (for causal mask)
        use_flash_attention (bool): Whether to use PyTorch SDPA
        attn_droupout (float): Dropout probability for attention weights
        residual_droupout (float): Dropout probability after output projection

    Shapes:
        Input:
            x: (B, T, C)

        Internal:
            q: (B, H_q, T, D)
            k/v: (B, H_kv, T, D)

        Output:
            out: (B, T, C)

    Raises:
        AssertionError: embed_dim % n_head != 0
        AssertionError: H_q % H_kv != 0 (for GQA)

    Notes:
        - Flash attention is only used for T > 1 (prefill/training)
        - Decode (T == 1) uses either:
            - blockwise streaming attention (paged cache), or
            - standard attention fallback
    """

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

        # GQA projections
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
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            past_key_value: KV cache object.
            use_cache (bool): Enable / Disable KV caching

        Returns:
            Tuple:
                - out (torch.Tensor): (B, T, C)
                - present: cache object, same as input, updated inplace.
        """
        B, T, C = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Cache setup
        k_raw, v_raw = k, v
        cache: BaseKVCache | None = past_key_value if use_cache else None
        present = cache
        is_decode = (T == 1) and use_cache

        # Prefill (T > 1)
        if use_cache and cache is not None and not is_decode:
            cache.append(k_raw, v_raw)
            k, v = cache.get_kv()

        # Decode (T == 1)
        elif use_cache and cache is not None and is_decode:
            k_past, v_past = cache.get_kv()
            k = torch.cat([k_past, k_raw], dim=2)
            v = torch.cat([v_past, v_raw], dim=2)

        # Attention
        if self.use_flash_attention and not is_decode:
            # Flash attention, only when T > 1
            k, v = self.expand_kv(k, v, self.group_size)

            atten = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )

        elif is_decode and cache is not None and cache.supports_blocks():
            # PagedKVCache decode
            atten = self.blockwise_decode_attention(q, cache, k_raw, v_raw)

        else:
            # Standard attention (training, prefill fallback, KVCache decode)
            k, v = self.expand_kv(k, v, self.group_size)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Mask ONLY if T > 1
            if T > 1:
                key_len = k.size(2)
                scores = scores.masked_fill(
                    ~self.mask[:T, :key_len], float("-inf")  # type: ignore
                )

            probs = torch.softmax(scores, dim=-1)
            probs = self.attn_dropout(probs)
            atten = torch.matmul(probs, v)

        # Update cache
        if use_cache and cache is not None and is_decode:
            cache.append(k_raw, v_raw)

        # Output projection
        atten = atten.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(atten)
        out = self.resid_dropout(out)

        return out, present

    def expand_kv(
        self, k: torch.Tensor, v: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand key/value heads to match a larger number of query heads.

        For use in Grouped Query Attention (GQA) or Multi-Query Attention (MQA),
        where the number of KV heads (H_kv) is smaller than the number of query heads (H_q).
        Each KV head is repeated `group_size` times so that:
            H_q = H_kv * group_size

        Args:
            k (torch.Tensor): Key tensor of shape (B, H_kv, T, D)
            v (torch.Tensor): Value tensor of shape (B, H_kv, T, D)
            group_size (int): Number of times each KV head is repeated

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Expanded keys of shape (B, H_kv * group_size, T, D)
                - Expanded values of shape (B, H_kv * group_size, T, D)

        Notes:
            - Uses `expand` (no memory copy) followed by `view`, so `.contiguous()`
              is required before reshaping.
            - This is more memory-efficient than `repeat`, but still materializes
              when `.contiguous()` is called.
        """
        B, H_kv, T, D = k.shape

        k = k[:, :, None, :, :]  # (B, H_kv, 1, T, D)
        v = v[:, :, None, :, :]

        # Broadcast along group dimension
        k = k.expand(B, H_kv, group_size, T, D)
        v = v.expand(B, H_kv, group_size, T, D)

        # Merge H_kv and group size -> H_q
        k = k.contiguous().view(B, H_kv * group_size, T, D)
        v = v.contiguous().view(B, H_kv * group_size, T, D)

        return k, v

    def blockwise_decode_attention(
        self,
        q: torch.Tensor,
        cache: BaseKVCache,
        k_cur: torch.Tensor,
        v_cur: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention over cached KV blocks + current token using
        block-wise streaming softmax.
        This avoids materializing full attention scores over long contexts
        by iteratively updating the softmax normalization across blocks.

        Args:
            q (torch.Tensor): Query tensor of shape (B, H_q, 1, D)
            cache: KV cache object exposing `get_blocks()` which returns:
                - blocks_k: Iterable of key blocks, each (B, H_kv, T_block, D)
                - blocks_v: Iterable of value blocks, same shape as keys
                - last_offset (int): Valid length of the final cached block
            k_cur (torch.Tensor): Current step keys, shape (B, H_kv, 1, D)
            v_cur (torch.Tensor): Current step values, shape (B, H_kv, 1, D)

        Returns:
            torch.Tensor: Attention output of shape (B, H_q, 1, D)

        Notes:
            - Uses Grouped Query Attention (GQA) via `expand_kv`
            - Applies numerically stable streaming softmax:
                  softmax(x) = exp(x - max) / sum(exp(x - max))
            - Avoids large intermediate tensors across full sequence length
            - Final division normalizes accumulated weighted values
        """
        blocks_k, blocks_v, last_offset = cache.get_blocks()

        # Treat current token as last block
        blocks_k = list(blocks_k) + [k_cur]
        blocks_v = list(blocks_v) + [v_cur]

        max_score = None  # Running max for stability
        sum_exp = None  # runnign averaging denominator
        out = None  # running weighted values

        for i, (k_block, v_block) in enumerate(zip(blocks_k, blocks_v)):

            # Apply last_offset only for last cached block
            if i == len(blocks_k) - 2:
                k_block = k_block[:, :, :last_offset]
                v_block = v_block[:, :, :last_offset]

            # Expand KV heads
            k_exp, v_exp = self.expand_kv(k_block, v_block, self.group_size)

            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            block_max = scores.max(dim=-1, keepdim=True).values

            if max_score is None:
                max_score = block_max
                exp_scores = torch.exp(scores - max_score)
                sum_exp = exp_scores.sum(dim=-1, keepdim=True)
                out = torch.matmul(exp_scores, v_exp)
                continue
            assert sum_exp is not None and out is not None and max_score is not None

            # Streaming softmax update
            new_max = torch.maximum(max_score, block_max)

            scale_old = torch.exp(max_score - new_max)
            exp_new = torch.exp(scores - new_max)

            sum_exp = scale_old * sum_exp + exp_new.sum(dim=-1, keepdim=True)
            out = scale_old * out + torch.matmul(exp_new, v_exp)

            max_score = new_max

        assert sum_exp is not None and out is not None
        return out / sum_exp
