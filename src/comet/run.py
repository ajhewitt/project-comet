from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import healpy as hp
import numpy as np
import yaml

from .config import get_data_dir


def _git_info() -> Dict[str, str]:
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


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_yaml(p: Optional[str | Path]) -> Dict[str, Any]:
    if not p:
        return {}
    pth = Path(p)
    if not pth.exists():
        return {}
    with pth.open() as f:
        return yaml.safe_load(f) or {}


def run_pipeline(
    prereg_path: Optional[str | Path] = None,
    paths_path: Optional[str | Path] = None,
    ordering: str = "both",
    check_only: bool = False,
) -> Dict[str, Any]:
    """Top-level runner: reads configs, inspects maps, writes metadata stub."""
    now = datetime.now(timezone.utc).isoformat()

    prereg = _read_yaml(prereg_path)
    paths = _read_yaml(paths_path)

    data_dir = get_data_dir()
    lensing = data_dir / "COM_CompMap_Lensing_2048_R1.10.fits"
    cmb = data_dir / "COM_CompMap_CMB-smica_2048_R1.20.fits"

    payload: Dict[str, Any] = {
        "timestamp_utc": now,
        "ordering": ordering,
        "config": {
            "prereg_path": str(prereg_path) if prereg_path else None,
            "paths_path": str(paths_path) if paths_path else None,
        },
        "git": _git_info(),
        "data": {
            "data_dir": str(data_dir),
            "lensing": {"present": lensing.exists(), "path": str(lensing)},
            "cmb": {"present": cmb.exists(), "path": str(cmb)},
        },
        "results": {
            "nbins": 0,
            "z": 0.0,
            "notes": "stub" if not check_only else "check_only",
        },
    }

    return payload
