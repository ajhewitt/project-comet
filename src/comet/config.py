from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - optional dependency in CI
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def load_prereg(path: str | os.PathLike) -> dict:
    if yaml is None:
        msg = (
            "PyYAML is required to load preregistration files; install the optional "
            "'config' dependencies to enable this feature"
        )
        raise ModuleNotFoundError(msg)
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_dir() -> Path:
    env = os.getenv("COMET_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[2]
    return (here / "data").resolve()
