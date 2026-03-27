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

    # expected top-k indices
    topk_idx = torch.topk(scores, k=max_len).indices.tolist()
    topk_idx_set = set(topk_idx)

    # verify that each returned token comes from top-k
    for i in range(k_sparse.shape[2]):
        found_match = False
        for pos in topk_idx_set:
            if torch.allclose(k_sparse[:, :, i], k[:, :, pos]):
                found_match = True
                break
        assert found_match, "Found token not in top-k selection"


def test_sparse_topk_replacement_behavior():
    B, H, T, D = 1, 2, 6, 8
    max_len = 3

    cache = SparseKVCache(
        B, H, D, max_len, device="cpu", dtype=torch.float32, mode="topk", topk=max_len
    )

    k, v = generate_random_kv(B, H, T, D)

    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cache.append(k, v, score=scores)

    k_sparse, v_sparse = cache.get_kv()

    # expected top-k indices
    topk_idx = torch.topk(scores, k=max_len).indices.tolist()
    topk_idx_sorted = sorted(topk_idx)

    # verify KV correctness
    for i, pos in enumerate(topk_idx_sorted):
        torch.testing.assert_close(
            k_sparse[:, :, i],
            k[:, :, pos],
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


def test_sparse_topk_preserves_temporal_order():
    B, H, T, D = 1, 1, 6, 4
    max_len = 3

    cache = SparseKVCache(
        B, H, D, max_len, device="cpu", dtype=torch.float32, mode="topk", topk=max_len
    )

    k, v = generate_random_kv(B, H, T, D)
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

    cache.append(k, v, score=scores)

    k_sparse, _ = cache.get_kv()

    # expected top-k indices
    topk_idx = torch.topk(scores, k=max_len).indices.tolist()
    topk_idx_sorted = sorted(topk_idx)

    # Now verify each position matches correct token
    for i, pos in enumerate(topk_idx_sorted):
        torch.testing.assert_close(
            k_sparse[:, :, i],
            k[:, :, pos],
        )


def test_sparse_stride_kv_values_match_source():
    B, H, T, D = 1, 1, 8, 4
    stride = 2

    cache = SparseKVCache(
        B,
        H,
        D,
        max_len=4,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
        stride=stride,
    )

    k, v = generate_random_kv(B, H, T, D)
    cache.append(k, v)

    k_sparse, v_sparse = cache.get_kv()

    expected_indices = list(range(0, T, stride))

    for i, idx in enumerate(expected_indices):
        torch.testing.assert_close(k_sparse[:, :, i], k[:, :, idx])
        torch.testing.assert_close(v_sparse[:, :, i], v[:, :, idx])


def test_sparse_topk_keeps_correct_positions():
    B, H, T, D = 1, 1, 6, 4
    max_len = 3

    cache = SparseKVCache(
        B, H, D, max_len, device="cpu", dtype=torch.float32, mode="topk", topk=max_len
    )

    k, v = generate_random_kv(B, H, T, D)

    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    cache.append(k, v, score=scores)

    # expected top-k indices
    topk_idx = torch.topk(scores, k=max_len).indices.tolist()

    # cache stores positions
    assert sorted(cache.positions) == sorted(topk_idx)


def test_sparse_empty_cache_returns_empty():
    cache = SparseKVCache(
        B=1,
        H=1,
        head_dim=4,
        max_len=4,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
    )

    k, v = cache.get_kv()
    assert k.numel() == 0
    assert v.numel() == 0


def test_sparse_incremental_append_stride():
    B, H, T, D = 1, 1, 6, 4
    stride = 2

    cache = SparseKVCache(
        B,
        H,
        D,
        max_len=3,
        device="cpu",
        dtype=torch.float32,
        mode="stride",
        stride=stride,
    )

    k, v = generate_random_kv(B, H, T, D)

    for t in range(T):
        cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])

    k_sparse, _ = cache.get_kv()

    expected_len = len(range(0, T, stride))
    assert k_sparse.shape[2] == expected_len
