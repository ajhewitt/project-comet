from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import get_data_dir


def _git_info() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out = ""
        return out

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": "1" if _run(["git", "status", "--porcelain"]) else "0",
    }


def _read_yaml(p: str | Path | None) -> dict:
    if not p:
        return {}
    pth = Path(p)
    if not pth.exists():
        return {}
    with pth.open() as f:
        return yaml.safe_load(f) or {}


def _cfg_hash(prereg: dict, paths: dict) -> str:
    blob = json.dumps({"prereg": prereg, "paths": paths}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def run_pipeline(
    prereg_path: str | Path | None = None,
    paths_path: str | Path | None = None,
    ordering: str = "both",
    check_only: bool = False,
) -> dict:
    """Top-level runner: reads configs, inspects maps, writes metadata stub."""
    now = datetime.now(timezone.utc).isoformat()

    prereg = _read_yaml(prereg_path)
    paths = _read_yaml(paths_path)
    cfg_hash = _cfg_hash(prereg, paths)

    # Optional guardrail: refuse to run on dirty trees unless explicitly allowed
    gi = _git_info()
    if gi["dirty"] == "1" and not bool(int(os.getenv("COMET_ALLOW_DIRTY", "0"))):
        raise RuntimeError(
            "Refusing to run on a dirty git tree. Set COMET_ALLOW_DIRTY=1 to override."
        )

    data_dir = get_data_dir()
    lensing = data_dir / "COM_CompMap_Lensing_2048_R1.10.fits"
    cmb = data_dir / "COM_CompMap_CMB-smica_2048_R1.20.fits"

    payload: dict = {
        "timestamp_utc": now,
        "ordering": ordering,
        "config": {
            "prereg_path": str(prereg_path) if prereg_path else None,
            "paths_path": str(paths_path) if paths_path else None,
            "hash_sha256": cfg_hash,
        },
        "git": gi,
        "data": {
            "data_dir": str(data_dir),
            "lensing": {"present": lensing.exists(), "path": str(lensing)},
            "cmb": {"present": cmb.exists(), "path": str(cmb)},
        },
        "results": {
            "nbins": 0,
            "z": 0.0,
            "notes": "check_only" if check_only else "stub",
        },
    }

    return payload
