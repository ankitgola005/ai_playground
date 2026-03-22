import pytest
from unittest.mock import patch, MagicMock
from ai_playground.utils.model import build_model
from ai_playground.configs.config import ModelConfig


@pytest.mark.parametrize(
    "model_name, patch_path",
    [
        ("minigpt", "ai_playground.models.miniGPT.MiniGPT"),
        ("bigram", "ai_playground.models.bigram.BiGram"),
        ("mnist", "ai_playground.models.mnist.MNIST"),
    ],
)
def test_build_model_known(model_name, patch_path):
    config = ModelConfig(model_name=model_name, compile=True, model_kwargs={})
    with patch(patch_path, new=MagicMock) as mock_class:
        model_class = build_model(config)
        assert model_class == mock_class


def test_build_model_unknown():
    config = ModelConfig(model_name="unknown", compile=True, model_kwargs={})
    with pytest.raises(NotImplementedError):
        build_model(config)
