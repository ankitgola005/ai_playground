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


def test_generator_stops_on_eos(make_model, tokenizer):
    model = make_model()

    # Force model to always output EOS
    class DummyModel(model.__class__):
        def forward(self, idx, **kwargs):
            B, T = idx.shape
            vocab_size = self.head.out_features
            logits = torch.zeros(B, T, vocab_size)
            logits[..., tokenizer.eos_token_id] = 100
            return logits, None, kwargs.get("past_key_values")

    dummy_model = DummyModel(model.model_config, tokenizer.vocab_size, 16)
    generator = Generator(dummy_model, tokenizer, device=torch.device("cpu"))

    outputs, _ = generator.generate(["hello"], max_tokens=50)

    # Should stop early (not reach max_tokens)
    assert len(outputs[0]) < len("hello") + 50


def test_cache_vs_no_cache_consistency(make_model, tokenizer):
    torch.manual_seed(0)
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))
    out_no_cache, _ = generator.generate(
        ["hello world"], max_tokens=10, use_cache=False
    )

    torch.manual_seed(0)
    out_cache, _ = generator.generate(["hello world"], max_tokens=10, use_cache=True)
    assert len(out_no_cache[0]) == len(out_cache[0])


def test_variable_length_prompts(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    outputs, _ = generator.generate(
        ["hi", "this is a much longer prompt"],
        max_tokens=5,
    )

    assert len(outputs) == 2


def test_empty_prompt(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    outputs, _ = generator.generate([""], max_tokens=5)

    assert len(outputs) == 1


def test_zero_max_tokens(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    prompt = "hello"
    outputs, _ = generator.generate([prompt], max_tokens=0)

    assert outputs[0].startswith(prompt)


def test_kv_cache_progression(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    _, stats = generator.generate(["hello"], max_tokens=5, use_cache=True)

    # Just ensure decode ran
    assert stats["decode_time"] > 0


def test_deterministic_generation(make_model, tokenizer):
    torch.manual_seed(42)

    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    out1, _ = generator.generate(["hello"], max_tokens=5, temperature=0.0)

    torch.manual_seed(42)
    out2, _ = generator.generate(["hello"], max_tokens=5, temperature=0.0)

    assert out1 == out2


def test_sampling_changes_output(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    out1, _ = generator.generate(["hello"], max_tokens=10, temperature=1.0)
    out2, _ = generator.generate(["hello"], max_tokens=10, temperature=1.0)

    assert out1 != out2


def test_top_k_runs(make_model, tokenizer):
    model = make_model()
    generator = Generator(model, tokenizer, device=torch.device("cpu"))

    outputs, _ = generator.generate(["hello"], max_tokens=5, topk=5)

    assert len(outputs) == 1
