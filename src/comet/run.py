from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import healpy as hp
import numpy as np
import yaml

from .config import get_data_dir


@dataclass
class MapInfo:
    path: Path
    nside: int
    npix: int
    f_sky: float


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


def _map_info(p: Path) -> Optional[MapInfo]:
    try:
        m = hp.read_map(p.as_posix())
    except Exception:
        return None
    if m.ndim != 1:
        m = np.asarray(m)
        if m.ndim == 2 and m.shape[0] >= 1:
            m = m[0]
        else:
            return None
    nside = hp.get_nside(m)
    f_sky = float(np.isfinite(m).mean())
    return MapInfo(path=p, nside=nside, npix=m.size, f_sky=f_sky)


def run_pipeline(
    prereg_path: Optional[str | Path] = None,
    paths_path: Optional[str | Path] = None,
    ordering: str = "both",
    check_only: bool = False,
) -> Dict[str, Any]:
    """
    Minimal runner that validates inputs, inspects local Planck maps if present,
    computes stable metadata, and returns a structured payload.
    This is a drop-in upgrade over the previous stub.
    """
    now = datetime.now(timezone.utc).isoformat()

    prereg = _read_yaml(prereg_path)
    paths = _read_yaml(paths_path)

    data_dir = get_data_dir()
    lensing = data_dir / "COM_CompMap_Lensing_2048_R1.10.fits"
    cmb = data_dir / "COM_CompMap_CMB-smica_2048_R1.20.fits"

    lensing_info = _map_info(lensing) if lensing.exists() else None
    cmb_info = _map_info(cmb) if cmb.exists() else None

    # Config hash based on prereg + paths content for traceability
    cfg_blob = json.dumps({"prereg": prereg, "paths": paths}, sort_keys=True).encode()
    cfg_hash = hashlib.sha256(cfg_blob).hexdigest()

    payload: Dict[str, Any] = {
        "timestamp_utc": now,
        "ordering": ordering,
        "config": {
            "prereg_path": str(prereg_path) if prereg_path else None,
            "paths_path": str(paths_path) if paths_path else None,
            "hash_sha256": cfg_hash,
        },
        "git": _git_info(),
        "data": {
            "data_dir": str(data_dir),
            "lensing": {
                "path": str(lensing) if lensing.exists() else None,
                "present": lensing.exists(),
                "sha256": _sha256_file(lensing) if lensing.exists() else None,
                "nside": getattr(lensing_info, "nside", None),
                "npix": getattr(lensing_info, "npix", None),
                "f_sky": getattr(lensing_info, "f_sky", None),
            },
            "cmb": {
                "path": str(cmb) if cmb.exists() else None,
                "present": cmb.exists(),
                "sha256": _sha256_file(cmb) if cmb.exists() else None,
                "nside": getattr(cmb_info, "nside", None),
                "npix": getattr(cmb_info, "npix", None),
                "f_sky": getattr(cmb_info, "f_sky", None),
            },
        },
        # Placeholders until real math is wired
        "results": {
            "nbins": 0,
            "z": 0.0,
            "notes": "check_only" if check_only else "stub",
        },
    }

    return payload
