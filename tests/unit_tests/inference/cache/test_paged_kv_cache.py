import torch

from ai_playground.inference.cache.paged_kv_cache import PagedKVCache


def make_cache(block_size=4):
    return PagedKVCache(
        B=1,
        H=2,
        head_dim=3,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_init_state():
    cache = make_cache()

    assert len(cache.blocks_k) == 1
    assert len(cache.blocks_v) == 1
    assert cache.offset == 0
    assert len(cache) == 0


def test_single_append():
    cache = make_cache(block_size=10)

    k = torch.randn(1, 2, 5, 3)
    v = torch.randn(1, 2, 5, 3)

    cache.append(k, v)

    k_out, v_out = cache.get_kv()

    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert len(cache) == 5
    assert cache.offset == 5


def test_append_across_blocks():
    cache = make_cache(block_size=4)

    k = torch.randn(1, 2, 6, 3)
    v = torch.randn(1, 2, 6, 3)

    cache.append(k, v)

    k_out, v_out = cache.get_kv()

    assert k_out.shape == (1, 2, 6, 3)
    assert v_out.shape == (1, 2, 6, 3)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)

    assert len(cache.blocks_k) == 2
    assert len(cache) == 6


def test_exact_block_boundary():
    cache = make_cache(block_size=4)

    k = torch.randn(1, 2, 4, 3)
    v = torch.randn(1, 2, 4, 3)

    cache.append(k, v)

    assert len(cache.blocks_k) == 1
    assert cache.offset == 4

    # next append forces new block
    cache.append(torch.randn(1, 2, 1, 3), torch.randn(1, 2, 1, 3))

    assert len(cache.blocks_k) == 2
    assert len(cache) == 5


def test_multiple_appends():
    cache = make_cache(block_size=4)

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


def test_iter_kv_matches_get_kv():
    cache = make_cache(block_size=4)

    k = torch.randn(1, 2, 7, 3)
    v = torch.randn(1, 2, 7, 3)

    cache.append(k, v)

    k_chunks = []
    v_chunks = []

    for kc, vc in cache.iter_kv():
        k_chunks.append(kc)
        v_chunks.append(vc)

    k_out = torch.cat(k_chunks, dim=2)
    v_out = torch.cat(v_chunks, dim=2)

    k_full, v_full = cache.get_kv()

    assert torch.allclose(k_out, k_full)
    assert torch.allclose(v_out, v_full)


def test_get_blocks():
    cache = make_cache(block_size=4)

    k = torch.randn(1, 2, 5, 3)
    v = torch.randn(1, 2, 5, 3)

    cache.append(k, v)

    blocks_k, blocks_v, offset = cache.get_blocks()

    assert isinstance(blocks_k, list)
    assert isinstance(blocks_v, list)
    assert len(blocks_k) == len(blocks_v)
    assert isinstance(offset, int)
    assert offset <= cache.block_size


def test_supports_blocks():
    cache = make_cache()
    assert cache.supports_blocks() is True


def test_reset():
    cache = make_cache(block_size=4)

    k = torch.randn(1, 2, 6, 3)
    v = torch.randn(1, 2, 6, 3)

    cache.append(k, v)
    cache.reset()

    assert len(cache) == 0
    assert len(cache.blocks_k) == 1
    assert cache.offset == 0


def test_empty_cache():
    cache = make_cache()

    k, v = cache.get_kv()

    assert k.shape == (1, 2, 0, 3)
    assert v.shape == (1, 2, 0, 3)
    assert len(cache) == 0
