from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt


@dataclass
class MapBundle:
    name: str
    map: np.ndarray
    mask: np.ndarray
    nside: int


def read_map(path: Path, quick_nside: int | None = None) -> np.ndarray:
    m = hp.read_map(path.as_posix())
    if m.ndim != 1:
        arr = np.asarray(m)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            m = arr[0]
        else:
            raise ValueError(f"Unexpected FITS shape for {path}: {arr.shape}")
    if quick_nside is not None and hp.get_nside(m) != quick_nside:
        # Power = -2 for temperature-like scalar field when downgrading
        m = hp.ud_grade(m, nside_out=quick_nside, power=-2)
    return m


def build_mask(m: np.ndarray, threshold_sigma: float = 10.0) -> np.ndarray:
    x = np.asarray(m, dtype=float)
    finite = np.isfinite(x)
    x = np.where(finite, x, 0.0)
    rms = float(np.sqrt(np.mean(x**2)))
    mask = finite & (np.abs(x) < threshold_sigma * (rms + 1e-30))
    return mask.astype(float)


def nm_bins_from_params(nside: int, lmax: int | None = None, nlb: int = 50) -> nmt.NmtBin:
    """
    Construct simple linear-ℓ binning based on map nside using NaMaster's helper.
    Uses NmtBin.from_nside_linear, which chooses lmax ~ 3*nside-1 internally.
    """
    if nlb <= 0:
        raise ValueError("nlb must be positive")
    # from_nside_linear ignores lmax; it sets binning according to nside and nlb.
    return nmt.NmtBin.from_nside_linear(nside=nside, nlb=nlb)


def nm_field_from_scalar(m: np.ndarray, mask: np.ndarray) -> nmt.NmtField:
    return nmt.NmtField(mask, [m])


def nm_bandpowers(f1: nmt.NmtField, f2: nmt.NmtField, b: nmt.NmtBin) -> np.ndarray:
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, b)
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = w.decouple_cell(cl_coupled)
    # Return the scalar spectrum (index 0)
    return cl_decoupled[0]


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), arr)


def summary_line(msg: str) -> None:
    print(json.dumps({"msg": msg}))
