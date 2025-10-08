"""Null-test helper utilities for Project Comet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commutator import z_score


@dataclass(slots=True)
class NullTestResult:
    """Container for the outcome of a single null test."""

    name: str
    residual: np.ndarray
    z: float


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"Expected 1-D array for null tests, got shape {data.shape}")
    return data


def rotation_null(delta: np.ndarray) -> np.ndarray:
    """Construct a rotation null residual by quarter-turn rolling the bandpowers.

    Parameters
    ----------
    delta:
        The baseline bandpower residuals. Values are interpreted as 1-D HEALPix
        pixels binned into bandpowers. The rotation null approximates a 90°
        rotation by performing a quarter-length cyclic roll. The difference
        between the original and rotated series captures anisotropies that
        should vanish for isotropic skies.
    """

    data = _ensure_1d(delta)
    if data.size == 0:
        return np.zeros_like(data)

    shift = max(1, data.size // 4)
    rotated = np.roll(data, shift)
    residual = data - rotated
    residual = residual - residual.mean()
    return residual.astype(float)


def hemisphere_jackknife(delta: np.ndarray) -> np.ndarray:
    """Apply a north/south jackknife sign flip after mean removal."""

    data = _ensure_1d(delta)
    if data.size == 0:
        return np.zeros_like(data)

    centred = data - data.mean()
    mask = np.ones_like(centred)
    half = centred.size // 2
    mask[half:] = -1.0
    residual = centred * mask
    return residual.astype(float)


def curl_null_field(delta: np.ndarray) -> np.ndarray:
    """Derive a curl-like null by differencing neighbouring bandpowers.

    This is a lightweight surrogate for a true B-mode extraction. The forward
    differences probe small-scale fluctuations; for curl-free skies the
    residual should be consistent with zero within the covariance.
    """

    data = _ensure_1d(delta)
    if data.size == 0:
        return np.zeros_like(data)

    forward_diff = np.roll(data, -1) - data
    residual = forward_diff - forward_diff.mean()
    return residual.astype(float)


def evaluate_null_tests(delta: np.ndarray, covariance: np.ndarray) -> dict[str, NullTestResult]:
    """Evaluate null variants and compute their Z-scores."""

    data = _ensure_1d(delta)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (data.size, data.size):
        raise ValueError(
            "Covariance shape mismatch: expected "
            f"({data.size}, {data.size}) but received {cov.shape}"
        )

    results: dict[str, NullTestResult] = {}

    def _record(name: str, residual: np.ndarray) -> None:
        arr = np.asarray(residual, dtype=float)
        if arr.shape != data.shape:
            raise ValueError(
                f"Residual for {name} has shape {arr.shape}, expected {data.shape}"
            )
        z_val = float(z_score(arr, cov))
        results[name] = NullTestResult(name=name, residual=arr.astype(float), z=z_val)

    _record("rotation_90", rotation_null(data))
    _record("hemisphere_jackknife", hemisphere_jackknife(data))
    _record("curl_null", curl_null_field(data))

    return results


__all__ = [
    "NullTestResult",
    "curl_null_field",
    "evaluate_null_tests",
    "hemisphere_jackknife",
    "rotation_null",
]
