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
