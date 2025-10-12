from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="commutator tests require numpy")

import numpy as np

from comet.commutator import commutator, z_score


def test_commutator_zero_for_identical_ops():
    x = np.arange(1, 10, dtype=float)
    y = np.arange(1, 10, dtype=float)[::-1]
    d = commutator(x, y)
    assert np.allclose(d, 0)


def test_z_score_finite():
    d = np.zeros(5)
    cov = np.eye(5)
    z = z_score(d, cov)
    assert np.isfinite(z) and z == 0


def test_z_score_requires_positive_variances():
    d = np.zeros(3)
    cov = np.eye(3)
    cov[0, 0] = -1.0
    with pytest.raises(ValueError, match="Covariance diagonal non-positive"):
        z_score(d, cov)
