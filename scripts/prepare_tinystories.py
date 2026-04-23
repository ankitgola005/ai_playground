import argparse
from pathlib import Path

from datasets import load_dataset


def save_split(dataset, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + " <eos>\n")

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download + prepare TinyStories")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/tinystories",
        help="Where to save processed dataset",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("Downloading TinyStories from HuggingFace...")

    train_ds = load_dataset("noanabeshima/TinyStoriesV2", split="train")
    val_ds = load_dataset("noanabeshima/TinyStoriesV2", split="validation")

    print("Converting train split...")
    save_split(train_ds, output_dir / "train.txt")

    print("Converting validation split...")
    save_split(val_ds, output_dir / "val.txt")

    print("\nDone. Dataset ready.")


if __name__ == "__main__":
    main()
