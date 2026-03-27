import torch
import pytest

from ai_playground.models.attention.multi_head_attention import MultiHeadAttention


def get_attn():
    return MultiHeadAttention(
        embed_dim=32,
        n_head=4,
        n_kv_head=2,
        block_size=16,
        use_flash_attention=False,
        attn_droupout=0.0,
        residual_droupout=0.0,
    )


class FixedTopKSelector:
    def __init__(self, k):
        self.k = k

    def select(self, q, k_tensor, scores):
        _, idx = torch.topk(scores, self.k, dim=-1)
        return idx


class InstrumentedSelector(FixedTopKSelector):
    def __init__(self, k):
        super().__init__(k)
        self.called = False

    def select(self, q, k_tensor, scores):
        self.called = True
        idx = super().select(q, k_tensor, scores)
        assert idx.shape[-1] == self.k
        return idx


def test_sparse_attention_output_shape():
    torch.manual_seed(0)

    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    attn = get_attn()
    attn.set_sparse_selector(FixedTopKSelector(k=4))
    out, _ = attn(x, use_cache=False)

    assert out.shape == (B, T, C)


def test_sparse_selector_called():
    torch.manual_seed(0)

    B, T, C = 1, 8, 32
    x = torch.randn(B, T, C)
    selector = InstrumentedSelector(k=2)
    attn = get_attn()
    attn.set_sparse_selector(selector)
    out, _ = attn(x, use_cache=False)

    assert selector.called, "Sparse selector was not called"


def test_sparse_topk_k_dimension():
    torch.manual_seed(0)

    B, T, C = 1, 8, 32
    x = torch.randn(B, T, C)
    k_val = 3
    selector = InstrumentedSelector(k=k_val)
    attn = get_attn()
    attn.set_sparse_selector(selector)
    out, _ = attn(x, use_cache=False)


def test_sparse_vs_dense_outputs_different():
    torch.manual_seed(0)

    B, T, C = 1, 8, 32
    x = torch.randn(B, T, C)
    attn_dense = get_attn()
    attn_sparse = get_attn()
    attn_sparse.set_sparse_selector(FixedTopKSelector(k=2))
    out_dense, _ = attn_dense(x, use_cache=False)
    out_sparse, _ = attn_sparse(x, use_cache=False)

    assert not torch.allclose(
        out_dense, out_sparse
    ), "Sparse attention behaving same as dense"


def test_sparse_topk_matches_manual_subset():
    torch.manual_seed(0)

    B, T, C = 1, 6, 32
    x = torch.randn(B, T, C)

    k_val = 3

    attn = get_attn()
    attn.set_sparse_selector(FixedTopKSelector(k=k_val))

    out_sparse, _ = attn(x, use_cache=False)

    with torch.no_grad():
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)

        B, T, _ = q.shape

        q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = k.view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        v = v.view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)

        k, v = attn.expand_kv(k, v, attn.group_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        key_len = k.size(2)
        scores = scores.masked_fill(~attn.mask[:T, :key_len], float("-inf"))

        _, idx = torch.topk(scores, k_val, dim=-1)

        D = k.size(-1)

        k_exp = k.unsqueeze(2).expand(-1, -1, T, -1, -1)
        v_exp = v.unsqueeze(2).expand(-1, -1, T, -1, -1)

        k_sel = torch.gather(k_exp, 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        v_sel = torch.gather(v_exp, 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, D))

        scores_sel = torch.gather(scores, -1, idx)

        probs = torch.softmax(scores_sel, dim=-1)

        out_manual = torch.sum(probs.unsqueeze(-1) * v_sel, dim=-2)

        out_manual = out_manual.transpose(1, 2).contiguous().view(B, T, C)
        out_manual = attn.out_proj(out_manual)

    assert torch.allclose(
        out_sparse, out_manual, atol=1e-5
    ), "Sparse attention mismatch with manual computation"
