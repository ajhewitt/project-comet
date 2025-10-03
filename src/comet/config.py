from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_prereg(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_dir() -> Path:
    env = os.getenv("COMET_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[2]
    return (here / "data").resolve()
