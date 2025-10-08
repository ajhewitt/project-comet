"""Mask construction and apodization utilities."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

_NMT_IMPORT_ERROR: Exception | None = None
_NMT_MODULE: Any | None = None
_HP_IMPORT_ERROR: Exception | None = None
_HP_MODULE: Any | None = None


def _load_namaster() -> Any | None:  # pragma: no cover - simple loader
    global _NMT_MODULE, _NMT_IMPORT_ERROR
    if _NMT_MODULE is not None or _NMT_IMPORT_ERROR is not None:
        return _NMT_MODULE

    import importlib

    try:
        _NMT_MODULE = importlib.import_module("pymaster")
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced via helper
        _NMT_IMPORT_ERROR = exc
        _NMT_MODULE = None
    except Exception as exc:  # pragma: no cover - surfaced via helper
        _NMT_IMPORT_ERROR = exc
        _NMT_MODULE = None
    else:  # pragma: no cover
        _NMT_IMPORT_ERROR = None

    return _NMT_MODULE


def _load_healpy() -> Any | None:  # pragma: no cover - simple loader
    global _HP_MODULE, _HP_IMPORT_ERROR
    if _HP_MODULE is not None or _HP_IMPORT_ERROR is not None:
        return _HP_MODULE

    import importlib

    try:
        _HP_MODULE = importlib.import_module("healpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced via helper
        _HP_IMPORT_ERROR = exc
        _HP_MODULE = None
    except Exception as exc:  # pragma: no cover - surfaced via helper
        _HP_IMPORT_ERROR = exc
        _HP_MODULE = None
    else:  # pragma: no cover
        _HP_IMPORT_ERROR = None

    return _HP_MODULE


def _require_nmt() -> Any:  # pragma: no cover - exercised only with dependency installed
    module = _load_namaster()
    if module is None:
        error = globals().get("_NMT_IMPORT_ERROR")
        if isinstance(error, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "pymaster (NaMaster) is required for mask apodization. "
                "Install it from conda-forge as 'namaster'."
            ) from error
        raise RuntimeError("Failed to import pymaster") from error
    return module


def _require_healpy() -> Any:  # pragma: no cover - exercised through tests
    module = _load_healpy()
    if module is None:
        error = globals().get("_HP_IMPORT_ERROR")
        if isinstance(error, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "healpy is required for mask apodization fallback."
            ) from error
        raise RuntimeError("Failed to import healpy") from error
    return module


def _should_use_namaster() -> bool:
    flag = os.environ.get("COMET_USE_NAMASTER_APODIZATION")
    if flag is None:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _healpy_apodization(mask: np.ndarray, apod_arcmin: float) -> np.ndarray:
    hp = _require_healpy()
    npix = mask.size
    hp.npix2nside(npix)
    fwhm_rad = float(apod_arcmin) * math.pi / (60.0 * 180.0)
    fwhm_rad *= math.sqrt(8.0 * math.log(2.0))
    smoothed = hp.smoothing(mask, fwhm=fwhm_rad, verbose=False)
    apodized = np.clip(np.asarray(smoothed, dtype=float), 0.0, 1.0)
    apodized = np.ascontiguousarray(apodized * (mask > 0.0), dtype=float)
    return apodized


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
    """Apply apodization with the requested radius (in arcminutes).

    NaMaster-based apodization can be re-enabled by setting the
    ``COMET_USE_NAMASTER_APODIZATION`` environment variable to a truthy value.
    Otherwise a healpy-based smoothing fallback is used to avoid native
    crashes in environments where NaMaster is unstable.
    """

    arr = np.ascontiguousarray(_as_float_array(mask), dtype=float)
    if apod_arcmin is None or apod_arcmin <= 0:
        return arr

    if not np.any(arr > 0.0):
        return arr

    if _should_use_namaster():  # pragma: no cover - requires dependency at runtime
        nmt = _require_nmt()
        aporad_deg = float(apod_arcmin) / 60.0
        apodized = nmt.mask_apodization(arr, aporad_deg, apotype="C1")
        apodized = np.clip(np.ascontiguousarray(apodized, dtype=float), 0.0, 1.0)
        apodized = apodized * (arr > 0.0)
        return apodized

    return _healpy_apodization(arr, float(apod_arcmin))


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
