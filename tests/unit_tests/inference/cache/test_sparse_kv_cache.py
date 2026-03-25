import torch
import pytest
from ai_playground.inference.cache.sparse_kv_cache import SparseKVCache


def generate_random_kv(B, H, T, D, device="cpu", dtype=torch.float32):
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    return k, v


def generate_random_scores(T, device="cpu", dtype=torch.float32):
    return torch.randn(T, device=device, dtype=dtype)


def test_sparse_stride_storage():
    B, H, T, D = 1, 2, 16, 8
    stride = 2
    max_len = 8
    cache = SparseKVCache(
        B,
        H,
        D,
        max_len,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
        stride=stride,
    )

    k, v = generate_random_kv(B, H, T, D)
    cache.append(k, v)

    k_sparse, v_sparse = cache.get_kv()
    assert k_sparse.shape[2] == len(cache.positions) == 8
    assert cache.positions == list(range(0, T, stride))


def test_sparse_reset_clears_cache_stride():
    B, H, T, D = 1, 2, 8, 8
    cache = SparseKVCache(
        B, H, D, max_len=4, device="cpu", dtype=torch.float32, mode="stride", stride=2
    )
    k, v = generate_random_kv(B, H, T, D)
    cache.append(k, v)
    cache.reset()
    assert len(cache) == 0
    assert cache.positions == []


def test_sparse_overflow_raises_stride():
    B, H, T, D = 1, 2, 10, 8
    stride = 1
    max_len = 5
    cache = SparseKVCache(
        B,
        H,
        D,
        max_len,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
        stride=stride,
    )
    k, v = generate_random_kv(B, H, T, D)
    with pytest.raises(RuntimeError):
        cache.append(k, v)


def test_sparse_iter_kv_yields_correct_shapes_stride():
    B, H, T, D = 1, 2, 8, 8
    stride = 2
    max_len = 4
    cache = SparseKVCache(
        B,
        H,
        D,
        max_len,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
        stride=stride,
    )
    k, v = generate_random_kv(B, H, T, D)
    cache.append(k, v)
    for k_iter, v_iter in cache.iter_kv():
        assert k_iter.shape[2] == len(cache.positions)
        assert v_iter.shape == k_iter.shape


def test_sparse_topk_keeps_topk_tokens():
    B, H, T, D = 1, 2, 8, 8
    max_len = 4
    cache = SparseKVCache(
        B, H, D, max_len, device="cpu", dtype=torch.float32, mode="topk", topk=max_len
    )
    k, v = generate_random_kv(B, H, T, D)
    scores = generate_random_scores(T)
    cache.append(k, v, score=scores)

    k_sparse, v_sparse = cache.get_kv()
    assert k_sparse.shape[2] <= max_len
    # All stored scores must be among the top-k
    topk_scores = sorted(scores.tolist(), reverse=True)[:max_len]
    stored_scores = [s for s in cache.scores]
    for s in stored_scores:
        assert s in topk_scores


def test_sparse_topk_replacement_behavior():
    B, H, T, D = 1, 2, 6, 8
    max_len = 3
    cache = SparseKVCache(
        B, H, D, max_len, device="cpu", dtype=torch.float32, mode="topk", topk=max_len
    )
    k, v = generate_random_kv(B, H, T, D)
    # increasing scores to force replacement
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cache.append(k, v, score=scores)

    k_sparse, v_sparse = cache.get_kv()
    # only top-3 scores remain
    expected_scores = [0.4, 0.5, 0.6]
    torch.testing.assert_close(
        torch.tensor(cache.scores), torch.tensor(expected_scores), rtol=1e-6, atol=1e-6
    )


def test_sparse_reset_clears_cache_topk():
    B, H, T, D = 1, 2, 6, 8
    cache = SparseKVCache(
        B, H, D, max_len=3, device="cpu", dtype=torch.float32, mode="topk", topk=3
    )
    k, v = generate_random_kv(B, H, T, D)
    scores = generate_random_scores(T)
    cache.append(k, v, score=scores)
    cache.reset()
    assert len(cache) == 0
    assert cache.positions == []
    assert cache.scores == []
