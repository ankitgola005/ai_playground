import torch
import pytest

from ai_playground.models.attention import SelfAttention, MultiHeadAttention


class DummyKVCache:
    def __init__(self):
        self.k = []
        self.v = []

    def append(self, k, v):
        self.k.append(k)
        self.v.append(v)

    def get_kv(self):
        return torch.cat(self.k, dim=2), torch.cat(self.v, dim=2)

    def iter_kv(self):
        for k, v in zip(self.k, self.v):
            yield k, v

    def supports_blocks(self):
        return False


def test_self_attention_shape():
    B, T, C = 2, 8, 32
    head_dim = 16

    attn = SelfAttention(C, head_dim, block_size=16)
    x = torch.randn(B, T, C)

    out = attn(x)

    assert out.shape == (B, T, head_dim)


def test_self_attention_causal_mask():
    B, T, C = 1, 4, 8
    head_dim = 8

    attn = SelfAttention(C, head_dim, block_size=4)
    x = torch.randn(B, T, C)

    with torch.no_grad():
        k = attn.key(x)
        q = attn.query(x)

        scores = (q @ k.transpose(-2, -1)) * attn.scale
        scores = scores.masked_fill(~attn.mask[:T, :T], float("-inf"))
        mask = attn.mask[:T, :T]

        # Check that all masked (future) positions are -inf
        assert torch.isinf(scores[0][~mask]).all()

        # Unmasked positions must be finite
        assert torch.isfinite(scores[0][mask]).all()


def get_mha(use_flash=False):
    return MultiHeadAttention(
        embed_dim=64,
        n_head=8,
        n_kv_head=4,
        block_size=32,
        use_flash_attention=use_flash,
        attn_droupout=0.0,
        residual_droupout=0.0,
    )


def test_mha_output_shape():
    B, T, C = 2, 8, 64
    attn = get_mha()

    x = torch.randn(B, T, C)
    out, _ = attn(x)

    assert out.shape == (B, T, C)


def test_expand_kv():
    B, H_kv, T, D = 2, 2, 5, 8
    group_size = 4

    attn = get_mha()

    k = torch.randn(B, H_kv, T, D)
    v = torch.randn(B, H_kv, T, D)

    k_exp, v_exp = attn.expand_kv(k, v, group_size)

    assert k_exp.shape == (B, H_kv * group_size, T, D)
    assert v_exp.shape == (B, H_kv * group_size, T, D)

    # Check correctness of repetition
    for i in range(H_kv):
        for g in range(group_size):
            idx = i * group_size + g
            assert torch.allclose(k_exp[:, idx], k[:, i])
            assert torch.allclose(v_exp[:, idx], v[:, i])


def test_prefill_cache():
    B, T, C = 2, 6, 64
    attn = get_mha()

    x = torch.randn(B, T, C)
    cache = DummyKVCache()

    out, present = attn(x, past_key_value=cache, use_cache=True)

    k, v = cache.get_kv()

    assert k.shape[2] == T
    assert v.shape[2] == T
    assert present is cache


def test_decode_cache_growth():
    B, C = 2, 64
    attn = get_mha()

    cache = DummyKVCache()

    # Prefill
    x = torch.randn(B, 5, C)
    attn(x, past_key_value=cache, use_cache=True)

    # Decode step-by-step
    for _ in range(3):
        x = torch.randn(B, 1, C)
        attn(x, past_key_value=cache, use_cache=True)

    k, v = cache.get_kv()

    assert k.shape[2] == 8  # 5 + 3
    assert v.shape[2] == 8


def test_no_cache_mode():
    B, T, C = 2, 8, 64
    attn = get_mha()

    x = torch.randn(B, T, C)

    out, present = attn(x, use_cache=False)

    assert out.shape == (B, T, C)
    assert present is None


@pytest.mark.parametrize("T", [4, 8])
def test_flash_vs_standard_close(T):
    B, C = 2, 64

    torch.manual_seed(42)
    x = torch.randn(B, T, C)

    attn_std = get_mha(use_flash=False)
    attn_flash = get_mha(use_flash=True)

    # Copy weights to ensure identical params
    attn_flash.load_state_dict(attn_std.state_dict())

    out_std, _ = attn_std(x)
    out_flash, _ = attn_flash(x)

    assert torch.allclose(out_std, out_flash, atol=1e-4)


def test_deterministic_no_dropout():
    B, T, C = 2, 8, 64
    attn = get_mha()

    x = torch.randn(B, T, C)

    out1, _ = attn(x)
    out2, _ = attn(x)

    assert torch.allclose(out1, out2)


def test_single_token_forward():
    B, T, C = 2, 1, 64
    attn = get_mha()

    x = torch.randn(B, T, C)

    out, _ = attn(x)

    assert out.shape == (B, 1, C)
