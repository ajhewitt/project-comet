import numpy as np


def ordering_a_then_b(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return x_ell * y_ell


def ordering_b_then_a(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return y_ell * x_ell


def commutator(x_ell: np.ndarray, y_ell: np.ndarray) -> np.ndarray:
    return ordering_a_then_b(x_ell, y_ell) - ordering_b_then_a(x_ell, y_ell)


def z_score(delta: np.ndarray, cov: np.ndarray) -> float:
    cinv = np.linalg.pinv(cov)
    val = float(delta @ cinv @ delta)
    return val**0.5
