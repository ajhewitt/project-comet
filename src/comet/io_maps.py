from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np


def read_fits_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    m = hp.read_map(path.as_posix(), verbose=False)
    if m.ndim != 1:
        m = np.array(m)
        if m.ndim == 2 and m.shape[0] >= 1:
            m = m[0]
        else:
            raise ValueError(f"Unexpected FITS shape for {path}: {m.shape}")
    return m


def get_nside(m: np.ndarray) -> int:
    return hp.get_nside(m)


def map_info(m: np.ndarray) -> dict:
    return {
        "nside": get_nside(m),
        "npix": m.size,
        "f_sky": float(np.isfinite(m).mean()),
    }
