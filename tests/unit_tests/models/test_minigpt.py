import torch
import pytest
from ai_playground.models.miniGPT import MiniGPT
from ai_playground.configs import ModelConfig


@pytest.fixture
def make_model():
    def _make_model(vocab_size=100, block_size=16):
        config = ModelConfig(
            model_name="minigpt",
            compile=False,
            model_kwargs={
                "n_embed": 64,
                "n_head": 8,
                "n_kv_head": 4,
                "hidden_dim": 128,
                "n_layer": 2,
                "use_flash_attention": False,
                "num_experts": 0,
                "attn_dropout": 0.0,
                "residual_dropout": 0.0,
                "ffn_dropout": 0.0,
                "moe_dropout": 0.0,
                "use_kv_cache": True,
                "kv_cache_max_len": 32,
                "use_paged_kv_cache": False,
                "paged_kv_cache_block_size": 8,
            },
        )
        return MiniGPT(config, vocab_size=vocab_size, block_size=block_size).eval()

    return _make_model


@pytest.mark.parametrize("B,T", [(2, 1), (2, 8)])
@pytest.mark.parametrize("vocab_size", [10, 100])
def test_forward_shapes(make_model, B, T, vocab_size):
    model = make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T))
    out = model(idx)

    assert out["logits"].shape == (B, T, vocab_size)
    assert out["loss"] is None
    assert out["kv"] is None


def test_forward_with_loss(make_model):
    B, T, vocab_size = 2, 8, 100
    model = make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    out = model(idx, targets=targets)

    assert out["logits"].shape == (B, T, vocab_size)
    assert out["loss"] is not None
    assert out["loss"].dim() == 0


def test_deterministic_no_dropout(make_model):
    B, T, vocab_size = 2, 8, 100
    model = make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T))
    out1 = model(idx)
    out2 = model(idx)
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6)


def test_forward_with_cache_flag(make_model):
    B, T, vocab_size = 2, 8, 100
    model = make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T))
    out = model(idx, use_cache=True)
    assert out["logits"].shape == (B, T, vocab_size)
    assert isinstance(out["kv"], list)
    assert len(out["kv"]) == model.model_config.model_kwargs["n_layer"]


def test_init_kv_cache(make_model):
    B, vocab_size = 2, 100
    model = make_model(vocab_size=vocab_size)
    caches = model.init_kv_cache(B, device="cpu")
    assert isinstance(caches, list)
    assert len(caches) == model.model_config.model_kwargs["n_layer"]


def test_not_identity(make_model):
    B, T, vocab_size = 2, 8, 100
    model = make_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T))
    out = model(idx)
    preds = torch.argmax(out["logits"], dim=-1)
    assert not torch.equal(preds, idx)
