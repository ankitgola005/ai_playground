import torch
import pytest

from ai_playground.configs import ModelConfig
from ai_playground.inference.generator import Generator
from ai_playground.models.miniGPT import MiniGPT
from ai_playground.data.char_tokenizer import CharTokenizer
from ai_playground.utils.data import get_dataset_path


@pytest.fixture
def tokenizer():
    dataset_path = get_dataset_path("shakespeare")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    return tokenizer


@pytest.fixture
def make_model(tokenizer):
    def _make_model(
        vocab_size: int = tokenizer.vocab_size,
        block_size: int = 16,
        device: str = "cpu",
    ):
        config = ModelConfig(
            model_name="minigpt",
            compile=True,
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
                "kv_cache_max_len": 2048,
                "use_paged_kv_cache": False,
                "paged_kv_cache_block_size": 8,
            },
        )
        model = MiniGPT(config, vocab_size=vocab_size, block_size=block_size)
        model = model.to(device)
        model.eval()
        return model

    return _make_model


def test_generator_runs(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))
    outputs, stats = generator.generate(
        ["hello world"],
        max_tokens=5,
        use_cache=False,
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)

    assert "prefill_time" in stats
    assert "decode_time" in stats


def test_generator_cache_toggle(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))
    out_no_cache, _ = generator.generate(["test"], max_tokens=5, use_cache=False)
    out_cache, _ = generator.generate(["test"], max_tokens=5, use_cache=True)

    assert len(out_no_cache) == len(out_cache)


def test_generator_batch(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))
    outputs, _ = generator.generate(
        ["hello", "world"],
        max_tokens=5,
    )

    assert len(outputs) == 2


def test_sample_shape(make_model):
    model = make_model()
    generator = Generator(model, tokenizer=None, device=torch.device("cpu"))
    logits = torch.randn(2, 3, 10)
    next_token = generator.sample(logits)

    assert next_token.shape == (2, 1)


def test_output_length_increases(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))
    prompt = "hello"
    max_tokens = 5
    outputs, _ = generator.generate([prompt], max_tokens=max_tokens)

    # crude check: length increases
    assert len(outputs[0]) >= len(prompt)
