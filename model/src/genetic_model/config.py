from pathlib import Path

import yaml

# model/, so paths in configs work no matter where a command is run from
MODEL_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = MODEL_DIR / "configs" / "tiny.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to model/."""
    path = Path(path)
    return path if path.is_absolute() else MODEL_DIR / path


def model_source(name: str) -> str:
    """A local model directory (relative to model/) if it exists, else a Hugging Face Hub id."""
    local = resolve(name)
    return str(local) if local.is_dir() else name
