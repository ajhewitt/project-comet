"""Theory C_ell loading utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True, frozen=True)
class TheoryCls:
    """Container for a set of temperature/lensing theory spectra."""

    ell: np.ndarray
    cl_tt: np.ndarray
    cl_kk: np.ndarray
    cl_tk: np.ndarray

    def as_synalm_array(self) -> list[np.ndarray]:
        """Return spectra ordered for :func:`healpy.synalm`."""

        return [self.cl_tt, self.cl_tk, self.cl_kk]

    @property
    def lmax(self) -> int:
        if self.ell.size == 0:
            return 0
        return int(self.ell[-1])

    def truncate(self, lmax: int) -> TheoryCls:
        if lmax < 0:
            raise ValueError("lmax must be non-negative")
        if self.ell.size == 0:
            return self
        mask = self.ell <= lmax
        if not np.any(mask):
            raise ValueError("Requested lmax is below the smallest ell in the spectra")
        return TheoryCls(
            ell=self.ell[mask],
            cl_tt=self.cl_tt[mask],
            cl_kk=self.cl_kk[mask],
            cl_tk=self.cl_tk[mask],
        )


def _coerce_array(name: str, values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return arr


def _validate_monotonic(ell: np.ndarray) -> None:
    if ell.size == 0:
        raise ValueError("ell grid cannot be empty")
    diffs = np.diff(ell)
    if np.any(diffs <= 0):
        raise ValueError("ell grid must be strictly increasing")


def _validate_lengths(ell: np.ndarray, *arrays: tuple[str, np.ndarray]) -> None:
    expected = ell.size
    for name, arr in arrays:
        if arr.size != expected:
            raise ValueError(f"{name} has length {arr.size}, expected {expected}")


def load_theory(path: Path) -> TheoryCls:
    """Load temperature/lensing theory C_ell from ``.npz`` or column text."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        keys = {key.lower(): key for key in data.keys()}
        try:
            ell = _coerce_array("ell", data[keys.get("ell", "ell")])
            cl_tt = _coerce_array("cl_tt", data[keys.get("cl_tt", "cl_tt")])
            cl_kk = _coerce_array("cl_kk", data[keys.get("cl_kk", "cl_kk")])
            cl_tk = _coerce_array("cl_tk", data[keys.get("cl_tk", "cl_tk")])
        finally:
            data.close()
    elif suffix in {".txt", ".dat"}:
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError("Text theory files must have at least four columns")
        ell = _coerce_array("ell", raw[:, 0])
        cl_tt = _coerce_array("cl_tt", raw[:, 1])
        cl_kk = _coerce_array("cl_kk", raw[:, 2])
        cl_tk = _coerce_array("cl_tk", raw[:, 3])
    else:
        raise ValueError(f"Unsupported theory format: {path.suffix}")

    _validate_monotonic(ell)
    _validate_lengths(ell, ("cl_tt", cl_tt), ("cl_kk", cl_kk), ("cl_tk", cl_tk))

    for name, arr in {"cl_tt": cl_tt, "cl_kk": cl_kk}.items():
        if np.any(arr < 0):
            raise ValueError(f"{name} contains negative power")

    return TheoryCls(ell=ell, cl_tt=cl_tt, cl_kk=cl_kk, cl_tk=cl_tk)


__all__ = ["TheoryCls", "load_theory"]
