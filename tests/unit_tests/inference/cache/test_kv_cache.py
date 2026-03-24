import torch
import pytest

from ai_playground.inference.cache.kv_cache import KVCache


def make_cache(max_len=10):
    return KVCache(
        B=1,
        H=2,
        head_dim=3,
        max_len=max_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_init():
    cache = make_cache()

    assert cache.idx == 0
    assert len(cache) == 0
    assert cache.max_len == 10


def test_single_append():
    cache = make_cache()

    k = torch.randn(1, 2, 4, 3)
    v = torch.randn(1, 2, 4, 3)

    cache.append(k, v)

    k_out, v_out = cache.get_kv()

    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert len(cache) == 4


def test_multiple_appends():
    cache = make_cache()

    k1 = torch.randn(1, 2, 3, 3)
    v1 = torch.randn(1, 2, 3, 3)

    k2 = torch.randn(1, 2, 5, 3)
    v2 = torch.randn(1, 2, 5, 3)

    cache.append(k1, v1)
    cache.append(k2, v2)

    k_out, v_out = cache.get_kv()

    k_expected = torch.cat([k1, k2], dim=2)
    v_expected = torch.cat([v1, v2], dim=2)

    assert torch.allclose(k_out, k_expected)
    assert torch.allclose(v_out, v_expected)
    assert len(cache) == 8


def test_exact_capacity():
    cache = make_cache(max_len=6)

    k = torch.randn(1, 2, 6, 3)
    v = torch.randn(1, 2, 6, 3)

    cache.append(k, v)

    assert len(cache) == 6

    k_out, v_out = cache.get_kv()
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_overflow_raises():
    cache = make_cache(max_len=5)

    k = torch.randn(1, 2, 4, 3)
    v = torch.randn(1, 2, 4, 3)

    cache.append(k, v)

    k2 = torch.randn(1, 2, 2, 3)
    v2 = torch.randn(1, 2, 2, 3)

    with pytest.raises(RuntimeError):
        cache.append(k2, v2)


def test_iter_kv():
    cache = make_cache()

    k = torch.randn(1, 2, 5, 3)
    v = torch.randn(1, 2, 5, 3)

    cache.append(k, v)

    chunks = list(cache.iter_kv())

    assert len(chunks) == 1

    k_out, v_out = chunks[0]

    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_reset():
    cache = make_cache()

    k = torch.randn(1, 2, 4, 3)
    v = torch.randn(1, 2, 4, 3)

    cache.append(k, v)
    cache.reset()

    assert len(cache) == 0
    assert cache.idx == 0

    # Ensure overwrite works after reset
    k2 = torch.randn(1, 2, 3, 3)
    v2 = torch.randn(1, 2, 3, 3)

    cache.append(k2, v2)

    k_out, v_out = cache.get_kv()

    assert torch.allclose(k_out, k2)
    assert torch.allclose(v_out, v2)


def test_empty_cache():
    cache = make_cache()

    k, v = cache.get_kv()

    assert k.shape == (1, 2, 0, 3)
    assert v.shape == (1, 2, 0, 3)
    assert len(cache) == 0
