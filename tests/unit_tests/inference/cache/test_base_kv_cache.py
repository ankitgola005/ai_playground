import pytest
import torch

from ai_playground.inference.cache import BaseKVCache


class DummyKVCache(BaseKVCache):
    def __init__(self):
        self.k = []
        self.v = []

    def append(self, k, v):
        self.k.append(k)
        self.v.append(v)

    def get_kv(self):
        return torch.cat(self.k, dim=2), torch.cat(self.v, dim=2)

    def __len__(self):
        return sum(t.shape[2] for t in self.k)


def test_base_class_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseKVCache()


def test_append_and_get_kv():
    cache = DummyKVCache()

    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)

    k2 = torch.randn(1, 2, 2, 4)
    v2 = torch.randn(1, 2, 2, 4)

    cache.append(k1, v1)
    cache.append(k2, v2)

    k, v = cache.get_kv()

    assert k.shape == (1, 2, 5, 4)
    assert v.shape == (1, 2, 5, 4)


def test_len():
    cache = DummyKVCache()

    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)

    k2 = torch.randn(1, 2, 2, 4)
    v2 = torch.randn(1, 2, 2, 4)

    cache.append(k1, v1)
    cache.append(k2, v2)

    assert len(cache) == 5


def test_iter_kv_not_implemented():
    cache = DummyKVCache()

    with pytest.raises(NotImplementedError):
        list(cache.iter_kv())


def test_get_blocks_not_implemented():
    cache = DummyKVCache()

    with pytest.raises(NotImplementedError):
        cache.get_blocks()


def test_supports_blocks_default_false():
    cache = DummyKVCache()
    assert cache.supports_blocks() is False
