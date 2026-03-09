import argparse
from pathlib import Path

from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.trainer.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a YAML config")
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        required=True,
        help="Name of the YAML config file in the configs/ directory (e.g., minigpt.yaml)",
    )
    return parser.parse_args()


def run_training(config: ConfigProtocol):
    tokenizer, train_loader, val_loader = build_data_pipeline(config)
    model = build_model(config)(tokenizer.vocab_size, config)
    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    trainer.fit(model, train_loader, val_loader)


def main():
    # args = parse_args()
    # # Construct full path in configs/ folder relative to project root
    # project_root = Path(__file__).parent.resolve()
    # cfg_path = project_root / "configs" / args.cfg
    config: ConfigProtocol = load_yaml_config("/home/kitkat/Desktop/ishtudy/ml_system/ai_playground/configs/gpt_config.yaml")  # type: ignore
    strategy = get_strategy(config.distributed)
    strategy.launch(run_training, config)


if __name__ == "__main__":
    main()
