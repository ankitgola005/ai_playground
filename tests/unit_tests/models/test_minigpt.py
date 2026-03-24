import torch

from ai_playground.models.miniGPT import MiniGPT
from ai_playground.configs import ModelConfig


def make_model(vocab_size=100, block_size=16):
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
            "attn_dropout": 0.0,
            "residual_dropout": 0.0,
            "ffn_dropout": 0.0,
            "use_kv_cache": True,
            "kv_cache_max_len": 32,
            "use_paged_kv_cache": False,
            "paged_kv_cache_block_size": 8,
        },
    )

    return MiniGPT(config, vocab_size=vocab_size, block_size=block_size).eval()


def test_forward_shape():
    B, T = 2, 8
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    logits, loss, cache = model(idx)

    assert logits.shape == (B, T, 100)
    assert loss is None
    assert cache is None


def test_forward_with_loss():
    B, T = 2, 8
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    logits, loss, _ = model(idx, targets=targets)

    assert logits.shape == (B, T, 100)
    assert loss is not None
    assert loss.dim() == 0  # scalar loss


def test_deterministic_no_dropout():
    B, T = 2, 8
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    out1, _, _ = model(idx)
    out2, _, _ = model(idx)

    assert torch.allclose(out1, out2)


def test_forward_with_cache_flag():
    B, T = 2, 8
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    logits, loss, cache = model(idx, use_cache=True)

    assert logits.shape == (B, T, 100)
    assert isinstance(cache, list)
    assert len(cache) == model.model_config.model_kwargs["n_layer"]


def test_init_kv_cache():
    B = 2
    model = make_model()
    caches = model.init_kv_cache(B, device="cpu")

    assert isinstance(caches, list)
    assert len(caches) == model.model_config.model_kwargs["n_layer"]


def test_single_token():
    B, T = 2, 1
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    logits, _, _ = model(idx)

    assert logits.shape == (B, 1, 100)


def test_small_vocab():
    B, T = 2, 8
    model = make_model(vocab_size=10)
    idx = torch.randint(0, 10, (B, T))
    logits, _, _ = model(idx)

    assert logits.shape == (B, T, 10)


def test_not_identity():
    B, T = 2, 8
    model = make_model()
    idx = torch.randint(0, 100, (B, T))
    logits, _, _ = model(idx)

    # Collapse logits to predictions
    preds = torch.argmax(logits, dim=-1)

    # Very unlikely that model behaves like identity mapping
    assert not torch.equal(preds, idx)
