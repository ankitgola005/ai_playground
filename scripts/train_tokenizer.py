import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from ai_playground.utils.data import get_dataset_path


def load_all_text(dataset: str):
    paths = get_dataset_path(dataset)

    texts = []

    for split in ["train", "val"]:
        path = paths.get(split)
        if path is None:
            continue

        print(f"Loading: {path}")

        with open(path, "r", encoding="utf-8") as f:
            texts.extend(f.readlines())

    return texts


# BPE
def train_bpe(lines, save_path: Path, vocab_size: int):
    print(f"Training BPE (vocab_size={vocab_size})...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<eos>"],
    )

    tokenizer.train_from_iterator(lines, trainer)

    tokenizer.save(str(save_path))

    print(f"BPE tokenizer saved to: {save_path}")


# Char tokenizer
def train_char(lines, save_path: Path):
    print("Building char tokenizer...")

    vocab = set()

    for line in lines:
        vocab.update(list(line.strip()))

    vocab = sorted(vocab)

    stoi = {ch: i for i, ch in enumerate(vocab)}
    eos_token = "<eos>"
    eos_id = len(stoi)

    stoi[eos_token] = eos_id

    config = {
        "type": "char",
        "stoi": stoi,
        "itos": {i: ch for ch, i in stoi.items()},
        "eos_token": eos_token,
        "eos_token_id": eos_id,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Char tokenizer saved to: {save_path}")
    print(f"Vocab size: {len(stoi)}")


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer")

    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=["bpe", "char"],
    )

    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer.json",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8192,
        help="Only used for BPE",
    )

    args = parser.parse_args()

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 🔹 load all text (train + val if exists)
    lines = load_all_text(args.dataset)

    # 🔹 train tokenizer
    if args.tokenizer == "bpe":
        train_bpe(lines, save_path, args.vocab_size)

    elif args.tokenizer == "char":
        train_char(lines, save_path)

    else:
        raise ValueError("Unsupported tokenizer")


if __name__ == "__main__":
    main()
