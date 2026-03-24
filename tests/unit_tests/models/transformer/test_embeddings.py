import torch
import pytest

from ai_playground.models.transformer import (
    TokenEmbedding,
    PositionEmbedding,
    EmbeddingWrapper,
)


def test_token_embedding_shape():
    emb = TokenEmbedding(vocab_size=20, embed_dim=8)
    x = torch.randint(0, 20, (2, 5))  # (B=2, T=5)
    out = emb(x)

    assert out.shape == (2, 5, 8)


def test_token_embedding_consistency():
    emb = TokenEmbedding(vocab_size=10, embed_dim=4)
    x = torch.tensor([[1, 1, 1]])  # same token repeated
    out = emb(x)

    # all embeddings should be identical along T
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 1], out[:, 2])


def test_position_embedding_shape():
    emb = PositionEmbedding(block_size=10, embed_dim=8)
    x = torch.randint(0, 20, (2, 5))
    out = emb(x)

    assert out.shape == (2, 5, 8)


def test_position_embedding_same_positions_across_batch():
    emb = PositionEmbedding(block_size=10, embed_dim=6)
    x = torch.randint(0, 20, (2, 4))
    out = emb(x)

    # same positions across batch → same embeddings
    assert torch.allclose(out[0, 0], out[1, 0])
    assert torch.allclose(out[0, 1], out[1, 1])


def test_position_embedding_different_positions():
    emb = PositionEmbedding(block_size=10, embed_dim=6)
    x = torch.randint(0, 20, (1, 5))
    out = emb(x)

    # different positions → embeddings should differ
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_embedding_wrapper_shape():
    emb = EmbeddingWrapper(vocab_size=20, block_size=10, embed_dim=8)
    x = torch.randint(0, 20, (2, 5))
    out = emb(x)

    assert out.shape == (2, 5, 8)


def test_embedding_wrapper_addition():
    emb = EmbeddingWrapper(vocab_size=20, block_size=10, embed_dim=8)
    x = torch.randint(0, 20, (2, 5))
    tok = emb.token_embedding(x)
    pos = emb.position_embedding(x)
    out = emb(x)

    assert torch.allclose(out, tok + pos)


def test_device_consistency():
    emb = EmbeddingWrapper(vocab_size=20, block_size=10, embed_dim=8)
    x = torch.randint(0, 20, (2, 5))
    out = emb(x)

    assert out.device == x.device


def test_position_embedding_max_len():
    block_size = 6
    emb = PositionEmbedding(block_size=block_size, embed_dim=4)
    x = torch.randint(0, 10, (1, block_size))
    out = emb(x)

    assert out.shape == (1, block_size, 4)


def test_position_embedding_overflow():
    block_size = 5
    emb = PositionEmbedding(block_size=block_size, embed_dim=4)
    x = torch.randint(0, 10, (1, 6))  # T > block_size

    with pytest.raises(IndexError):
        emb(x)
