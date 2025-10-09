import pytest

pytest.importorskip("numpy", reason="simulation tests require numpy")
pytest.importorskip("healpy", reason="simulation tests require healpy")
pytest.importorskip("pymaster", reason="simulation tests require NaMaster")

import numpy as np

from comet.simulations import (
    SimulationGeometry,
    estimate_delta_covariance,
    resolve_simulation_bandlimits,
)
from comet.theory import TheoryCls
from commutator_common import nm_bins_from_params


@pytest.mark.slow
@pytest.mark.unit
def test_simulated_covariance_positive_definite():
    rng = np.random.default_rng(123)
    nside = 16
    ell = np.arange(2, 3 * nside)
    cl_tt = 1e-10 / (ell + 1)
    cl_kk = 5e-11 / (ell + 1)
    cl_tk = 2e-11 / (ell + 1)
    theory = TheoryCls(ell=ell, cl_tt=cl_tt, cl_kk=cl_kk, cl_tk=cl_tk)

    mask = np.ones(12 * nside**2)
    mask[::2] = 0.75
    mask[mask.size // 3 :] *= 0.5

    bins = nm_bins_from_params(nside=nside, nlb=8)
    sim_lmax, field_lmax = resolve_simulation_bandlimits(
        bins,
        requested_lmax=int(ell.max()),
        theory_lmax=int(ell.max()),
        nside=nside,
    )
    geom = SimulationGeometry(
        mask=mask,
        bins=bins,
        nside=nside,
        lmax=sim_lmax,
        field_lmax=field_lmax,
    )

    cov = estimate_delta_covariance(theory, geom, nsims=24, rng=rng)
    assert cov.shape[0] == cov.shape[1]
    assert np.allclose(cov, cov.T)
    diag = np.diag(cov)
    assert np.all(diag > 0)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)
