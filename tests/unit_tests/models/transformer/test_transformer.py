import torch

from ai_playground.models.transformer.transformer import TransformerBlock, FFN


def make_block(
    embed_dim=64,
    n_head=8,
    n_kv_head=4,
    hidden_dim=128,
    block_size=32,
):
    return TransformerBlock(
        embed_dim=embed_dim,
        n_head=n_head,
        n_kv_head=n_kv_head,
        block_size=block_size,
        hidden_dim=hidden_dim,
        use_flash_attention=False,
        num_experts=0,
        attn_dropout=0.0,
        residual_dropout=0.0,
        ffn_dropout=0.0,
        moe_dropout=0.0,
    ).eval()


def test_ffn_shape():
    B, T, C = 2, 8, 64
    ffn = FFN(embed_dim=C, hidden_dim=128, dropout=0.0)

    x = torch.randn(B, T, C)
    out = ffn(x)

    assert out.shape == (B, T, C)


def test_ffn_deterministic():
    B, T, C = 2, 8, 64
    ffn = FFN(embed_dim=C, hidden_dim=128, dropout=0.0).eval()

    x = torch.randn(B, T, C)

    out1 = ffn(x)
    out2 = ffn(x)

    assert torch.allclose(out1, out2)


def test_block_output_shape():
    B, T, C = 2, 8, 64
    block = make_block(C)

    x = torch.randn(B, T, C)
    out, _, _ = block(x, use_cache=False)

    assert out.shape == (B, T, C)


def test_block_residual_connection_changes_output():
    """
    Ensures block is not identity.
    """
    B, T, C = 2, 8, 64
    block = make_block(C)

    x = torch.randn(B, T, C)
    out, _, _ = block(x)

    assert not torch.allclose(out, x)


def test_block_deterministic_no_dropout():
    B, T, C = 2, 8, 64
    block = make_block(C)

    x = torch.randn(B, T, C)

    out1, _, _ = block(x)
    out2, _, _ = block(x)

    assert torch.allclose(out1, out2)


def test_block_layernorm_effect():
    """
    Check that LayerNorm is actually applied.
    """
    B, T, C = 2, 8, 64
    block = make_block(C)

    x = torch.randn(B, T, C)

    normed = block.linear1(x)

    # Mean should be ~0, variance ~1 (rough check)
    mean = normed.mean(dim=-1)
    var = normed.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_block_runs_with_cache_flag():
    B, T, C = 2, 8, 64
    block = make_block(C)

    x = torch.randn(B, T, C)
    out, _, _ = block(x, past_key_values=None, use_cache=True)

    assert out.shape == (B, T, C)


def test_single_token():
    B, T, C = 2, 1, 64
    block = make_block(C)

    x = torch.randn(B, T, C)
    out, _, _ = block(x)

    assert out.shape == (B, 1, C)


def test_small_dimensions():
    B, T, C = 1, 2, 16
    block = make_block(embed_dim=C, n_head=4, n_kv_head=2, hidden_dim=32)

    x = torch.randn(B, T, C)
    out, _, _ = block(x)

    assert out.shape == (B, T, C)
