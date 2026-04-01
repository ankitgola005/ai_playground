import torch
import pytest
import tempfile
import os

from ai_playground.inference.generator import Generator
from ai_playground.models.miniGPT import MiniGPT
from ai_playground.data.char_tokenizer import CharTokenizer
from ai_playground.utils.data import get_dataset_path


class DummyConfig:
    """Minimal config for generator tests"""

    def __init__(self, ckpt_path=None):
        self.ckpt_dir = ckpt_path
        self.model_kwargs = {
            "n_embed": 64,
            "n_head": 8,
            "n_kv_head": 4,
            "hidden_dim": 128,
            "n_layer": 2,
            "use_flash_attention": False,
            "num_experts": 0.0,
            "attn_dropout": 0.0,
            "residual_dropout": 0.0,
            "ffn_dropout": 0.0,
            "moe_dropout": 0.0,
            "use_kv_cache": True,
            "kv_cache_max_len": 2048,
            "use_paged_kv_cache": False,
            "paged_kv_cache_block_size": 8,
        }


@pytest.fixture
def tokenizer():
    dataset_path = get_dataset_path("shakespeare")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    return CharTokenizer(text)


@pytest.fixture
def make_model(tokenizer):
    def _make_model(
        vocab_size: int | None = None, block_size: int = 16, device: str = "cpu"
    ):
        vocab_size = vocab_size or tokenizer.vocab_size
        model = MiniGPT(DummyConfig(), vocab_size=vocab_size, block_size=block_size)
        model = model.to(device)
        model.eval()
        return model

    return _make_model


def get_generator(model, tokenizer, ckpt_path=None, device="cpu"):
    config = DummyConfig(ckpt_path=ckpt_path)
    return Generator(config, model, tokenizer, device=torch.device(device))  # type: ignore


# ---------------------------
# Parametrized generator tests
# ---------------------------


@pytest.mark.parametrize(
    "prompts,max_tokens,use_cache",
    [
        (["hello world"], 5, False),
        (["test"], 0, False),
        (["hi", "long prompt"], 3, True),
    ],
)
def test_generate_outputs(make_model, tokenizer, prompts, max_tokens, use_cache):
    generator = get_generator(make_model(), tokenizer)
    outputs, stats = generator.generate(
        prompts, max_tokens=max_tokens, use_cache=use_cache
    )
    assert isinstance(outputs, list)
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert isinstance(out, str)
    assert "prefill_time" in stats
    assert "decode_time" in stats


@pytest.mark.parametrize(
    "prompt,max_tokens,temperature",
    [
        ("hello", 5, 0.0),
        ("world", 10, 0.0),
    ],
)
def test_deterministic_generation(
    make_model, tokenizer, prompt, max_tokens, temperature
):
    torch.manual_seed(42)
    generator = get_generator(make_model(), tokenizer)
    out1, _ = generator.generate(
        [prompt], max_tokens=max_tokens, temperature=temperature
    )
    torch.manual_seed(42)
    out2, _ = generator.generate(
        [prompt], max_tokens=max_tokens, temperature=temperature
    )
    assert out1 == out2


def test_generator_with_dummy_ckpt(make_model, tokenizer):
    # Create a temporary dummy checkpoint file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        ckpt_path = tmp.name
    try:
        generator = get_generator(make_model(), tokenizer, ckpt_path=ckpt_path)
        outputs, stats = generator.generate(["hello"], max_tokens=3)
        assert len(outputs) == 1
    finally:
        os.remove(ckpt_path)
