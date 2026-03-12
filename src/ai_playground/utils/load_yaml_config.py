import yaml
from pathlib import Path
import ai_playground


CONFIG_BASE_PATH = yaml_path = Path(ai_playground.__file__).parent / "configs"


class ConfigNode:
    """
    Simple object wrapper around dict, allows attribute access.
    Nested dicts are converted recursively to ConfigNode.
    """

    def __init__(self, data: dict):
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"


def load_yaml_config(filename: str) -> "ConfigNode":
    """
    Load a YAML file from ai_playground/configs by filename and return a top-level ConfigNode.
    Args:
        filename: Name of the config file relative to ai_playground/configs, e.g. "gpt_config.yaml"
    Returns:
        ConfigNode: top-level config object
    """
    path = CONFIG_BASE_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return ConfigNode(cfg_dict)


def config_to_dict(obj):
    """Recursively convert a Protocol-based config to a dict."""
    if hasattr(obj, "__dict__"):  # objects with attributes
        return {k: config_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [config_to_dict(v) for v in obj]
    else:
        return obj
