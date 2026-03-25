import pytest
import torch

from ai_playground.models.transformer.attention import MultiHeadAttention
from ai_playground.inference.cache import KVCache, PagedKVCache


@pytest.fixture
def seed():
    torch.manual_seed(42)


@pytest.fixture
def shape():
    return 1, 8, 64  # B, T, C


@pytest.fixture
def attn(shape):
    B, T, C = shape
    return MultiHeadAttention(
        embed_dim=C,
        n_head=4,
        n_kv_head=2,
        block_size=16,
        use_flash_attention=False,
        attn_droupout=0.0,
        residual_droupout=0.0,
    ).eval()


@pytest.fixture
def x(shape):
    B, T, C = shape
    return torch.randn(B, T, C)


def test_kv_cache_correctness(attn, x, shape):
    B, T, C = shape
    head_dim = C // attn.n_head

    with torch.no_grad():
        k_full = attn.k_proj(x).view(B, T, attn.n_kv_head, head_dim).transpose(1, 2)
        v_full = attn.v_proj(x).view(B, T, attn.n_kv_head, head_dim).transpose(1, 2)

    cache = KVCache(B, attn.n_kv_head, head_dim, T, x.device, x.dtype)

    for t in range(T):
        xt = x[:, t : t + 1]
        k = attn.k_proj(xt).view(B, 1, attn.n_kv_head, head_dim).transpose(1, 2)
        v = attn.v_proj(xt).view(B, 1, attn.n_kv_head, head_dim).transpose(1, 2)
        cache.append(k, v)

    k_cache, v_cache = cache.get_kv()

    assert torch.allclose(k_full, k_cache, atol=1e-5)
    assert torch.allclose(v_full, v_cache, atol=1e-5)


def test_full_vs_cache_prefill(attn, x):
    out_full, _ = attn(x, use_cache=False)

    cache = KVCache(
        x.size(0), attn.n_kv_head, attn.head_dim, x.size(1), x.device, x.dtype
    )

    out_cache, _ = attn(x, past_key_value=cache, use_cache=True)

    assert torch.allclose(out_full, out_cache, atol=1e-5)


def test_decode_matches_full(attn, x):
    out_full, _ = attn(x, use_cache=False)

    cache = KVCache(
        B=x.size(0),
        H=attn.n_kv_head,
        head_dim=attn.head_dim,
        max_len=1024,
        device=x.device,
        dtype=x.dtype,
    )

    # prefill
    attn(x[:, :-1], past_key_value=cache, use_cache=True)

    # decode
    out_decode, _ = attn(x[:, -1:], past_key_value=cache, use_cache=True)

    out_full_last = out_full[:, -1:]

    assert torch.allclose(out_full_last, out_decode, atol=1e-5)


def test_attention_scores(attn, x):
    B, T, _ = x.shape

    q = attn.q_proj(x[:, -1:])
    k = attn.k_proj(x)

    q = q.view(B, 1, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k.view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)

    k_exp, _ = attn.expand_kv(k, k, attn.group_size)
    scores_full = torch.matmul(q, k_exp.transpose(-2, -1)) * attn.scale

    cache = KVCache(B, attn.n_kv_head, attn.head_dim, T, x.device, x.dtype)

    for t in range(T):
        xt = x[:, t : t + 1]
        k_t = attn.k_proj(xt).view(B, 1, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        cache.append(k_t, k_t)

    k_cache, _ = cache.get_kv()
    k_exp2, _ = attn.expand_kv(k_cache, k_cache, attn.group_size)

    scores_cache = torch.matmul(q, k_exp2.transpose(-2, -1)) * attn.scale

    assert torch.allclose(scores_full, scores_cache, atol=1e-5)


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_paged_matches_contiguous(attn, x, block_size):
    B, T, _ = x.shape

    cache1 = KVCache(B, attn.n_kv_head, attn.head_dim, T, x.device, x.dtype)
    cache2 = PagedKVCache(
        B, attn.n_kv_head, attn.head_dim, block_size, x.device, x.dtype
    )

    for t in range(T):
        xt = x[:, t : t + 1]
        k = attn.k_proj(xt).view(B, 1, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(xt).view(B, 1, attn.n_kv_head, attn.head_dim).transpose(1, 2)

        cache1.append(k, v)
        cache2.append(k, v)

    k1, v1 = cache1.get_kv()
    k2, v2 = cache2.get_kv()

    assert torch.allclose(k1, k2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


def test_blockwise_decode(attn, x):
    out_full, _ = attn(x, use_cache=False)

    cache = PagedKVCache(
        B=x.size(0),
        H=attn.n_kv_head,
        head_dim=attn.head_dim,
        block_size=128,
        device=x.device,
        dtype=x.dtype,
    )

    attn(x[:, :-1], past_key_value=cache, use_cache=True)
    out_decode, _ = attn(x[:, -1:], past_key_value=cache, use_cache=True)

    assert torch.allclose(out_full[:, -1:], out_decode, atol=1e-5)
