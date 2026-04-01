import torch
import pytest

from ai_playground.models.attention import SelfAttention, MultiHeadAttention
from ai_playground.models.attention.sparse_selector import TopKSelector, StrideSelector


@pytest.fixture
def mha():
    return MultiHeadAttention(
        embed_dim=64,
        n_head=8,
        n_kv_head=4,
        block_size=32,
        use_flash_attention=False,
        attn_droupout=0.0,
        residual_droupout=0.0,
    )


def make_qkv(attn, B=1, T=4):
    H_kv = attn.n_kv_head
    group_size = attn.group_size
    H_q = H_kv * group_size
    D = attn.head_dim

    q = torch.randn(B, H_q, 1, D)
    k = torch.randn(B, H_kv, T, D)
    v = torch.randn(B, H_kv, T, D)

    return q, k, v


# Self Attention tests
@pytest.mark.parametrize("B,T,C,head_dim", [(2, 8, 32, 16), (1, 4, 8, 8)])
def test_self_attention_shape(B, T, C, head_dim):
    attn = SelfAttention(C, head_dim, block_size=T)
    x = torch.randn(B, T, C)
    out = attn(x)
    assert out.shape == (B, T, head_dim)


def test_self_attention_causal_mask():
    attn = SelfAttention(8, 8, block_size=4)
    x = torch.randn(1, 4, 8)

    with torch.no_grad():
        k = attn.key(x)
        q = attn.query(x)

        scores = (q @ k.transpose(-2, -1)) * attn.scale
        scores = scores.masked_fill(~attn.mask[:4, :4], float("-inf"))
        mask = attn.mask[:4, :4]

        assert torch.isinf(scores[0][~mask]).all()
        assert torch.isfinite(scores[0][mask]).all()


# MHA
@pytest.mark.parametrize("B,T,C", [(2, 8, 64), (2, 1, 64)])
def test_mha_output_shape(mha, B, T, C):
    x = torch.randn(B, T, C)
    out, _ = mha(x)
    assert out.shape == (B, T, C)


def test_expand_kv(mha):
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)

    k_exp, v_exp = mha.expand_kv(k, v, group_size=4)

    assert k_exp.shape[1] == 8
    for i in range(2):
        for g in range(4):
            idx = i * 4 + g
            assert torch.allclose(k_exp[:, idx], k[:, i])


# KV Cache
class DummyKVCache:
    def __init__(self):
        self.k, self.v = [], []

    def append(self, k, v):
        self.k.append(k)
        self.v.append(v)

    def get_kv(self):
        return torch.cat(self.k, dim=2), torch.cat(self.v, dim=2)

    def iter_kv(self):
        return zip(self.k, self.v)

    def supports_blocks(self):
        return False


def test_cache_growth(mha):
    cache = DummyKVCache()

    mha(torch.randn(2, 5, 64), past_key_value=cache, use_cache=True)

    for _ in range(3):
        mha(torch.randn(2, 1, 64), past_key_value=cache, use_cache=True)

    k, _ = cache.get_kv()
    assert k.shape[2] == 8


def test_no_cache_mode(mha):
    out, present = mha(torch.randn(2, 8, 64), use_cache=False)
    assert present is None


# Flash Attention
@pytest.mark.parametrize("T", [4, 8])
def test_flash_vs_standard_close(T):
    torch.manual_seed(42)

    x = torch.randn(2, T, 64)

    attn_std = MultiHeadAttention(64, 8, 4, 32, False, 0.0, 0.0)
    attn_flash = MultiHeadAttention(64, 8, 4, 32, True, 0.0, 0.0)

    attn_flash.load_state_dict(attn_std.state_dict())

    out_std, _ = attn_std(x)
    out_flash, _ = attn_flash(x)

    assert torch.allclose(out_std, out_flash, atol=1e-4)


# Sparse Selectors
@pytest.mark.parametrize("topk,num_blocks", [(1, 5), (2, 4), (10, 3)])
def test_topk_selection(mha, topk, num_blocks):
    mha.sparse_selector = TopKSelector(topk=topk)

    q, k, v = make_qkv(mha)
    blocks = [(k, v) for _ in range(num_blocks)]

    selected = mha._select_blocks(q, blocks)

    assert len(selected) == min(topk, num_blocks)


def test_topk_ordering(mha):
    mha.sparse_selector = TopKSelector(topk=2)

    B, T = 1, 4
    H_kv = mha.n_kv_head
    D = mha.head_dim
    group_size = mha.group_size

    H_q = H_kv * group_size
    q = torch.ones(B, H_q, 1, D)

    blocks = []
    for i in range(4):
        k = torch.ones(B, H_kv, T, D) * i
        v = torch.ones(B, H_kv, T, D)
        blocks.append((k, v))

    selected = mha._select_blocks(q, blocks)

    assert selected[0][0].mean() >= selected[1][0].mean()


def test_topk_stability(mha):
    mha.sparse_selector = TopKSelector(topk=2)

    q, k, v = make_qkv(mha)
    blocks = [(k, v), (k, v), (k, v)]

    selected = mha._select_blocks(q, blocks)

    assert selected[0] == blocks[0]
    assert selected[1] == blocks[1]


def test_stride_selector(mha):
    mha.sparse_selector = StrideSelector(stride=2)

    blocks = list(range(6))
    selected = mha._select_blocks(None, blocks)

    assert selected == [0, 2, 4]


def test_selector_fallback(mha):
    mha.sparse_selector = None
    blocks = [1, 2, 3]

    assert mha._select_blocks(None, blocks) == blocks


# Raw Score Optimization
def test_raw_scores_equivalence(mha):
    mha.group_size = 4

    q = torch.randn(2, 8, 1, 8)
    k = torch.randn(2, 2, 5, 8)

    raw = mha._compute_raw_scores(q, k)

    k_exp, _ = mha.expand_kv(k, k, 4)
    exp = torch.matmul(q, k_exp.transpose(-2, -1)) * mha.scale

    assert torch.allclose(raw, exp, atol=1e-5)


def test_raw_scores_properties(mha):
    mha.group_size = 2

    q, k, _ = make_qkv(mha)

    s1 = mha._compute_raw_scores(q, k)
    s2 = mha._compute_raw_scores(q, k)

    assert s1.shape[-1] == k.shape[2]
    assert torch.allclose(s1, s2)


def test_raw_scores_assert(mha):
    mha.group_size = 2

    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 2, 5, 8)

    with pytest.raises(AssertionError):
        mha._compute_raw_scores(q, k)


def test_raw_scores_grad(mha):
    mha.group_size = 2

    q = torch.randn(1, 4, 1, 8, requires_grad=True)
    k = torch.randn(1, 2, 5, 8, requires_grad=True)

    loss = mha._compute_raw_scores(q, k).sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
