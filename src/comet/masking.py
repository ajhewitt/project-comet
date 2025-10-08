"""Mask construction and apodization utilities."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency in CI
    import pymaster as nmt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pymaster (NaMaster) is required for mask apodization. "
        "Install it from conda-forge as 'namaster'."
    ) from exc


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

    arr = _as_float_array(mask)
    if apod_arcmin is None or apod_arcmin <= 0:
        return arr

    aporad_deg = float(apod_arcmin) / 60.0
    apodized = nmt.mask_apodization(arr, aporad_deg, apotype="C1")
    apodized = np.clip(apodized, 0.0, 1.0)
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
