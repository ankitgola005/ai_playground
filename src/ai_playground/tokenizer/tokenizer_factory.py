from pathlib import Path
from ai_playground.tokenizer import CharTokenizer, BPETokenizer
from ai_playground.configs.config import TokenizerConfig


_TOKENIZER_CACHE: dict[tuple[str, str | None, str | None], object] = {}
TOKENIZER_DIR = Path(__file__).resolve().parent / "pretrained"


def build_tokenizer(tokenizer_config: TokenizerConfig, dataset: str | None = None):
    """
    Factory function to build a tokenizer.

    Tokenizers built from the same config and dataset are cached within the
    current process to avoid repeated expensive char tokenizer builds.
    """
    cache_key = (
        tokenizer_config.name,
        tokenizer_config.filename,
        dataset,
    )

    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    tokenizer_path = resolve_tokenizer_path(tokenizer_config)

    if tokenizer_config.name == "char":
        if tokenizer_path:
            tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            if dataset is None:
                raise ValueError(
                    "Dataset must be provided for char tokenizer build mode"
                )

            from ai_playground.utils.data import get_dataset_path

            paths = get_dataset_path(dataset)
            text_parts = []
            with open(paths["train"], "r", encoding="utf-8") as f:
                text_parts.append(f.read())

            if paths.get("val") is not None:
                with open(paths["val"], "r", encoding="utf-8") as f:
                    text_parts.append(f.read())

            text = "\n".join(text_parts)

            tokenizer = CharTokenizer()
            tokenizer.build_from_text(text)

    elif tokenizer_config.name == "bpe":
        if tokenizer_path is None:
            raise ValueError("BPE tokenizer requires a tokenizer file.")
        tokenizer = BPETokenizer.load(tokenizer_path)

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_config.name}")

    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def resolve_tokenizer_path(tokenizer_config: TokenizerConfig) -> str | None:
    if tokenizer_config.filename is None:
        return None

    tokenizer_path = TOKENIZER_DIR / tokenizer_config.filename

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    return str(tokenizer_path)
