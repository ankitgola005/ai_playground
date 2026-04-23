from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm
from ai_playground.data import dataset
from ai_playground.tokenizer.base_tokenizer import BaseTokenizer
from ai_playground.tokenizer.tokenizer_factory import build_tokenizer

if TYPE_CHECKING:
    from ai_playground.configs.config import DataConfig
    from typing import Iterable, Iterator, TypeVar, Dict, Tuple

    T = TypeVar("T")


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"


def get_dataset_path(dataset: str) -> Dict[str, Path | None]:
    """
    Returns train/val file paths for a given dataset name.

    Args:
        dataset: Dataset identifier.

    Returns:
        Dict with keys:
            - "train": Path to training file
            - "val": Path to validation file (or None if split is needed)

    Raises:
        NotImplementedError: If dataset is not supported.
    """
    if dataset == "tinyshakespeare":
        path = DATA_DIR / "text_datasets/tiny_shakespeare/tiny_shakespeare.txt"
        return {
            "train": path,
            "val": None,  # Split later from full dataset
        }

    elif dataset == "tinystories":
        base = DATA_DIR / "text_datasets/tinystories/"
        return {
            "train": base / "train.txt",
            "val": base / "val.txt",
        }

    elif dataset == "tinystoriesV2":
        base = DATA_DIR / "text_datasets/tinystoriesV2"
        return {
            "train": base / "train.txt",
            "val": base / "val.txt",
        }

    else:
        raise NotImplementedError(f"Dataset '{dataset}' is currently not supported.")


def create_infinite_loader(dl: Iterable[T]) -> Iterator[T]:
    """
    Creates an infinite iterator over a dataloader.
    Repeats dataset indefinitely by cycling over batches.

    Args:
        dl: Any iterable dataloader.

    Yields:
        Batches from dataloader in an infinite loop.
    """
    while True:
        for batch in dl:
            yield batch


def split_by_eos(
    encoded: torch.Tensor,
    split_ratio: float,
    eos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits token sequence at nearest EOS boundary before split point.

    Args:
        encoded: 1D tensor of token ids.
        split_ratio: Fraction for train split (e.g. 0.9).
        eos_token_id: End-of-sequence token id.

    Returns:
        (train_tokens, val_tokens)
    """
    target = int(len(encoded) * split_ratio)
    eos_positions = (encoded == eos_token_id).nonzero(as_tuple=True)[0]

    if eos_positions.numel() == 0:
        split_idx = target
    else:
        split_idx = eos_positions[(eos_positions <= target)].max().item() + 1

    return encoded[:split_idx], encoded[split_idx:]


def _compute_dataset_hash(paths: Dict[str, Path | None]) -> str:
    """
    Computes SHA256 hash over dataset files.
    Only train/val files are included. Used for cache invalidation.

    Args:
        paths: Dictionary containing dataset file paths.

    Returns:
        Hex digest string representing dataset content hash.
    """
    digest = sha256()
    for key in ("train", "val"):
        path = paths.get(key)
        if path is None:
            continue
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                digest.update(chunk)
    return digest.hexdigest()


def _get_cache_paths(
    data_config: "DataConfig",
    tokenizer: BaseTokenizer,
) -> Tuple[Path, Path, str]:
    """
    Computes cache paths and ID for dataset tokenization.

    Args:
        data_config: Data configuration.
        tokenizer: Tokenizer instance.

    Returns:
        train_tokens_path: Path for train tokens cache.
        val_tokens_path: Path for val tokens cache.
        cache_id: Unique cache identifier.
    """
    paths = get_dataset_path(data_config.dataset)
    cache_dir = DATA_DIR / "processed" / data_config.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = _compute_dataset_hash(paths)
    tokenizer_id = getattr(tokenizer, "filename", None) or type(tokenizer).__name__
    cache_id = sha256(
        f"{dataset_hash}|{type(tokenizer).__name__}|{tokenizer_id}".encode("utf-8")
    ).hexdigest()

    train_tokens_path = cache_dir / f"train_tokens_{cache_id}.pt"
    val_tokens_path = cache_dir / f"val_tokens_{cache_id}.pt"

    return train_tokens_path, val_tokens_path, cache_id


def _is_cache_valid(
    train_tokens_path: Path,
    val_tokens_path: Path,
    tokenizer: BaseTokenizer,
) -> bool:
    """
    Checks if cached token tensors are valid for the given tokenizer.

    Args:
        train_tokens_path: Path to train tokens.
        val_tokens_path: Path to val tokens.
        tokenizer: Tokenizer instance.

    Returns:
        True if cache is valid, False otherwise.
    """
    if not (train_tokens_path.exists() and val_tokens_path.exists()):
        return False

    try:
        train_tokens = torch.load(train_tokens_path)
        val_tokens = torch.load(val_tokens_path)
        return (
            train_tokens.numel() > 0
            and train_tokens.max().item() < tokenizer.vocab_size
            and val_tokens.numel() > 0
            and val_tokens.max().item() < tokenizer.vocab_size
        )
    except Exception:
        return False


def _process_dataset(
    data_config: "DataConfig",
    tokenizer: BaseTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes dataset files into tokenized tensors.

    Args:
        data_config: Data configuration.
        tokenizer: Tokenizer instance.

    Returns:
        train_tokens: Tokenized train data.
        val_tokens: Tokenized val data.
    """
    paths = get_dataset_path(data_config.dataset)

    def count_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def process_file(path):
        total_lines = count_lines(path)
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(
                f, total=total_lines, desc=f"Processing {path.name}", unit="lines"
            ):
                tokens.extend(tokenizer.encode(line.strip()) + [tokenizer.eos_token_id])
        return torch.tensor(tokens, dtype=torch.long)

    if paths["val"] is not None:
        train_tokens = process_file(paths["train"])
        val_tokens = process_file(paths["val"])
    else:
        full_tokens = process_file(paths["train"])
        train_tokens, val_tokens = split_by_eos(
            full_tokens,
            split_ratio=data_config.tokenizer.split,
            eos_token_id=tokenizer.eos_token_id,
        )

    return train_tokens, val_tokens


def _save_tokens(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    train_tokens_path: Path,
    val_tokens_path: Path,
) -> None:
    """
    Saves tokenized tensors to cache files.

    Args:
        train_tokens: Train token tensor.
        val_tokens: Val token tensor.
        train_tokens_path: Path to save train tokens.
        val_tokens_path: Path to save val tokens.
    """
    torch.save(train_tokens, train_tokens_path)
    torch.save(val_tokens, val_tokens_path)


def maybe_cache_dataset(
    data_config: "DataConfig",
    tokenizer: BaseTokenizer,
) -> Tuple[Path, Path]:
    """
    Loads or builds tokenized dataset with caching support.
    Cache is invalidated when:
        - Dataset file hash changes
        - Tokenizer config changes

    Args:
        data_config: Data configuration.
        tokenizer: Pre-built tokenizer instance.

    Returns:
        train_tokens_path: Path to cached train tensor.
        val_tokens_path: Path to cached val tensor.
    """
    train_tokens_path, val_tokens_path, _ = _get_cache_paths(data_config, tokenizer)

    if _is_cache_valid(train_tokens_path, val_tokens_path, tokenizer):
        return train_tokens_path, val_tokens_path

    print("Processing dataset ...")
    train_tokens, val_tokens = _process_dataset(data_config, tokenizer)
    _save_tokens(train_tokens, val_tokens, train_tokens_path, val_tokens_path)

    return train_tokens_path, val_tokens_path


def build_data_pipeline(
    data_config: "DataConfig",
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[BaseTokenizer, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Builds full training and validation data pipeline.

    Steps:
        1. Build or load cached tokenizer
        2. Load or build cached tokenized dataset
        3. Load tensors from disk
        4. Validate dataset size vs block size
        5. Create PyTorch dataloaders

    Returns:
        tokenizer:
            Tokenizer used for encoding.
        train_loader:
            Training dataloader.
        val_loader:
            Validation dataloader.
    """
    tokenizer = build_tokenizer(data_config.tokenizer, data_config.dataset)
    train_path, val_path = maybe_cache_dataset(data_config, tokenizer)

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    if len(train_data) <= data_config.block_size:
        raise ValueError("Train dataset too small")

    if (
        val_data is not None
        and len(val_data) > 0
        and len(val_data) <= data_config.block_size
    ):
        raise ValueError("Val dataset too small")

    train_loader = dataset.build_dataloader(
        data_config=data_config,
        encoded_data=train_data,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    val_loader = dataset.build_dataloader(
        data_config=data_config,
        encoded_data=val_data,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return tokenizer, train_loader, val_loader
