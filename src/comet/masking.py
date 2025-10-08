"""Mask construction and apodization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

_NMT_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - optional dependency in CI
    import pymaster as _nmt
except ModuleNotFoundError as exc:  # pragma: no cover - handled lazily in helpers
    _nmt = None
    _NMT_IMPORT_ERROR = exc
except Exception as exc:  # pragma: no cover - propagate unexpected import failures
    _nmt = None
    _NMT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _NMT_IMPORT_ERROR = None

def _require_nmt() -> Any:  # pragma: no cover - exercised only with dependency installed
    if _nmt is None:
        if isinstance(_NMT_IMPORT_ERROR, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "pymaster (NaMaster) is required for mask apodization. "
                "Install it from conda-forge as 'namaster'."
            ) from _NMT_IMPORT_ERROR
        raise RuntimeError("Failed to import pymaster") from _NMT_IMPORT_ERROR
    return _nmt


def _as_float_array(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Mask must be 1-D, got shape {arr.shape}")
    return arr


def threshold_mask(m: np.ndarray, threshold_sigma: float = 10.0) -> np.ndarray:
    """Return a binary mask after applying an RMS-based threshold."""

    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")

    x = np.asarray(m, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Input map must be 1-D, got shape {x.shape}")

    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Input map has no finite pixels to mask")

    cleaned = np.where(finite, x, 0.0)
    rms = float(np.sqrt(np.mean(cleaned**2)))
    if not np.isfinite(rms) or rms == 0:
        rms = 1.0

    mask = finite & (np.abs(cleaned) < threshold_sigma * (rms + 1e-30))
    return mask.astype(float)


def apodize_mask(mask: np.ndarray, apod_arcmin: float | None = None) -> np.ndarray:
    """Apply NaMaster C1 apodization with the requested radius (in arcminutes)."""

    arr = np.ascontiguousarray(_as_float_array(mask), dtype=float)
    if apod_arcmin is None or apod_arcmin <= 0:
        return arr

    if not np.any(arr > 0.0):
        return arr

    nmt = _require_nmt()
    aporad_deg = float(apod_arcmin) / 60.0
    apodized = nmt.mask_apodization(arr, aporad_deg, apotype="C1")
    apodized = np.clip(np.ascontiguousarray(apodized, dtype=float), 0.0, 1.0)
    apodized = apodized * (arr > 0)
    return apodized


def build_mask(
    m: np.ndarray,
    threshold_sigma: float = 10.0,
    apod_arcmin: float | None = None,
) -> np.ndarray:
    """Construct a float mask with thresholding and optional apodization."""

    mask = threshold_mask(m, threshold_sigma=threshold_sigma)
    mask = apodize_mask(mask, apod_arcmin=apod_arcmin)
    return mask.astype(float)


def effective_f_sky(mask: np.ndarray) -> float:
    """Compute the sky fraction covered by the (possibly apodized) mask."""

    arr = _as_float_array(mask)
    return float(np.mean(arr))


__all__ = [
    "apodize_mask",
    "build_mask",
    "effective_f_sky",
    "threshold_mask",
]
