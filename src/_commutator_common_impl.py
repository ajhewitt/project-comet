from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import pymaster as nmt

from comet.config import load_prereg
from comet.masking import build_mask as _build_mask
from comet.masking import effective_f_sky as _effective_f_sky

__all__ = [
    "MapBundle",
    "build_mask",
    "effective_f_sky",
    "load_bins_from_prereg",
    "nm_bandpowers",
    "nm_bins_from_config",
    "nm_bins_from_params",
    "nm_field_from_scalar",
    "read_map",
    "save_json",
    "save_npy",
    "summary_line",
]


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


def build_mask(
    m: np.ndarray,
    threshold_sigma: float = 10.0,
    apod_arcmin: float | None = None,
) -> np.ndarray:
    return _build_mask(m, threshold_sigma=threshold_sigma, apod_arcmin=apod_arcmin)


def effective_f_sky(mask: np.ndarray) -> float:
    return _effective_f_sky(mask)


def nm_bins_from_params(
    nside: int,
    lmax: int | None = None,
    nlb: int = 50,
    lmin: int | None = None,
) -> nmt.NmtBin:
    """
    Construct simple linear-ℓ binning based on map nside using NaMaster's helper.
    Uses NmtBin.from_nside_linear, which chooses lmax ~ 3*nside-1 internally.
    """
    if nlb <= 0:
        raise ValueError("nlb must be positive")
    kwargs: dict[str, Any] = {"nside": nside, "nlb": nlb}
    lmin_val: int | None = None
    if lmin is not None:
        if lmin < 0:
            raise ValueError("lmin must be non-negative")
        lmin_val = int(lmin)
        kwargs["lmin"] = lmin_val
    lmax_val: int | None = None
    if lmax is not None:
        if lmax <= 0:
            raise ValueError("lmax must be positive")
        lmax_val = int(lmax)
        kwargs["lmax"] = lmax_val

    try:
        return nmt.NmtBin.from_nside_linear(**kwargs)
    except TypeError:
        if lmin_val is None:
            raise
        # Older NaMaster releases do not accept lmin in from_nside_linear; fall back to
        # constructing explicit edges that honour the requested lower bound.
        from_edges = getattr(nmt.NmtBin, "from_edges", None)
        if from_edges is None:
            raise RuntimeError("NaMaster does not expose NmtBin.from_edges; cannot enforce lmin")
        if lmax_val is None:
            lmax_val = 3 * nside - 1
        ell_min = np.arange(lmin_val, lmax_val + 1, nlb, dtype=int)
        ell_max = np.minimum(ell_min + nlb - 1, lmax_val)
        return from_edges(ell_min, ell_max)


def nm_bins_from_config(nside: int, bins_cfg: Mapping[str, Any]) -> nmt.NmtBin:
    """Create NaMaster bins from a preregistration ``bins`` configuration block."""

    if "nlb" not in bins_cfg:
        raise KeyError("bins configuration requires 'nlb'")

    nlb = int(bins_cfg["nlb"])
    lmin = bins_cfg.get("lmin")
    lmax = bins_cfg.get("lmax")
    return nm_bins_from_params(
        nside=nside,
        nlb=nlb,
        lmin=None if lmin is None else int(lmin),
        lmax=None if lmax is None else int(lmax),
    )


def load_bins_from_prereg(prereg_path: Path, nside: int) -> tuple[nmt.NmtBin, Mapping[str, Any]]:
    """Load the prereg YAML and construct NaMaster bins from its ``ells.bins`` block."""

    prereg = load_prereg(prereg_path)
    bins_cfg = prereg.get("ells", {}).get("bins")
    if bins_cfg is None:
        raise KeyError("prereg configuration is missing 'ells.bins'")
    bins = nm_bins_from_config(nside=nside, bins_cfg=bins_cfg)
    return bins, bins_cfg


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
