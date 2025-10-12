from __future__ import annotations

import numpy as np


def ordering_a_then_b(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return x_ell * y_ell


def ordering_b_then_a(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return y_ell * x_ell


def commutator(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return ordering_a_then_b(x_ell, y_ell) - ordering_b_then_a(x_ell, y_ell)


def z_score(delta: np.ndarray, cov: np.ndarray) -> float:
    """Return the Mahalanobis distance associated with ``delta``.

    Parameters
    ----------
    delta
        Residual vector to be tested against the covariance.
    cov
        Covariance matrix describing the uncertainty of ``delta``.

    Raises
    ------
    ValueError
        If the covariance shape mismatches the residual, contains non-positive
        variances, or cannot be pseudo-inverted into a positive semi-definite
        form.
    """

    resid = np.asarray(delta, dtype=float)
    covariance = np.asarray(cov, dtype=float)

    if resid.ndim != 1:
        raise ValueError(
            f"Expected 1-D residual array for z-score calculation, got {resid.shape}"
        )

    if covariance.shape != (resid.size, resid.size):
        raise ValueError(
            "Covariance shape mismatch: expected "
            f"({resid.size}, {resid.size}) but received {covariance.shape}"
        )

    diag = np.diag(covariance)
    if np.any(diag <= 0):
        raise ValueError("Covariance diagonal non-positive; cannot compute z-scores")

    try:
        cinv = np.linalg.pinv(covariance, hermitian=True)
    except TypeError:
        # ``hermitian`` keyword was added in NumPy 1.17; fall back gracefully if
        # an older version is in use.  This preserves backwards compatibility for
        # callers running in minimal environments.
        cinv = np.linalg.pinv(covariance)

    val = float(resid @ cinv @ resid)
    if val < 0:
        raise ValueError(
            "Covariance pseudo-inverse produced a negative quadratic form"
        )

    return float(np.sqrt(val))
