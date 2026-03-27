import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.inference.cache import BaseKVCache, SparseSelector


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
        embed_dim: int,
        n_head: int,
        n_kv_head: int,
        block_size: int,
        use_flash_attention: bool,
        attn_droupout: float,
        residual_droupout: float,
    ):
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        super().__init__()
        self.embed_dim: int = embed_dim
        self.head_dim: int = embed_dim // n_head
        self.n_head: int = n_head
        self.n_kv_head: int = n_kv_head
        self.group_size: int = self.n_head // self.n_kv_head
        self.use_flash_attention: bool = use_flash_attention
        self.last_attn: float = 0.0

        # GQA projections
        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj: nn.Linear = nn.Linear(
            embed_dim, self.n_kv_head * self.head_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            embed_dim, self.n_kv_head * self.head_dim, bias=False
        )
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )
        self.scale: float = self.head_dim**-0.5

        self.attn_dropout: nn.Dropout = nn.Dropout(attn_droupout)
        self.resid_dropout: nn.Dropout = nn.Dropout(residual_droupout)

        self.sparse_selector: SparseSelector | None = None

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: BaseKVCache | None = None,
        use_cache: bool = False,
    ):
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
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # Reshape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Cache setup
        k_raw, v_raw = k, v
        cache: BaseKVCache | None = past_key_value if use_cache else None
        present = cache
        is_decode: bool = (T == 1) and use_cache

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

            if self.sparse_selector is not None:
                idx = self.sparse_selector.select(q, k, scores)
                D = k.size(-1)

                # k_exp = k.unsqueeze(2).expand(-1, -1, q.size(2), -1, -1)
                v_exp = v.unsqueeze(2).expand(-1, -1, q.size(2), -1, -1)

                # k_sel = torch.gather(
                #     k_exp,
                #     3,
                #     idx.unsqueeze(-1).expand(-1, -1, -1, -1, D),
                # )
                v_sel = torch.gather(
                    v_exp,
                    3,
                    idx.unsqueeze(-1).expand(-1, -1, -1, -1, D),
                )
                scores_sel = torch.gather(scores, -1, idx)

                probs = torch.softmax(scores_sel, dim=-1)
                probs = self.attn_dropout(probs)
                atten = torch.sum(probs.unsqueeze(-1) * v_sel, dim=-2)
            else:
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
        max_score = torch.full_like(
            q[..., :1], float("-inf")
        )  # Running max for stability
        sum_exp = torch.zeros_like(max_score)  # runnign averaging denominator
        out = torch.zeros_like(q)  # running weighted values

        # iterate cached KV
        for k_block, v_block in cache.iter_kv():
            # Expand KV heads
            k_exp, v_exp = self.expand_kv(k_block, v_block, self.group_size)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            max_score, sum_exp, out = self._streaming_softmax_update(
                max_score, sum_exp, out, scores, v_exp
            )

        # treat current token as final chunk
        k_exp, v_exp = self.expand_kv(k_cur, v_cur, self.group_size)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        max_score, sum_exp, out = self._streaming_softmax_update(
            max_score, sum_exp, out, scores, v_exp
        )

        return out / sum_exp

    def _streaming_softmax_update(
        self,
        max_score: torch.Tensor,
        sum_exp: torch.Tensor,
        out: torch.Tensor,
        scores: torch.Tensor,
        v_exp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform numerically stable streaming softmax update.

        Args:
            max_score: running max (B, H, 1, Tq)
            sum_exp: running sum of exp (B, H, 1, Tq)
            out: running output (B, H, Tq, D)
            scores: current attention scores (B, H, Tq, Tk)
            v_exp: expanded values (B, H, Tk, D)

        Returns:
            updated (max_score, sum_exp, out)
        """
        block_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(max_score, block_max)
        scale_old = torch.exp(max_score - new_max)
        exp_new = torch.exp(scores - new_max).to(v_exp.dtype)
        sum_exp = scale_old * sum_exp + exp_new.sum(dim=-1, keepdim=True)
        out = scale_old * out + torch.matmul(exp_new, v_exp)
        return new_max, sum_exp, out

    def set_sparse_selector(self, selector: SparseSelector):
        self.sparse_selector = selector
