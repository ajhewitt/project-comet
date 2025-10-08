import pytest

pytest.importorskip("numpy", reason="healpix round-trip tests require numpy")
pytest.importorskip("healpy", reason="healpix round-trip tests require healpy")

import healpy as hp
import numpy as np


@pytest.mark.unit
def test_map_roundtrip_small_nside():
    rng = np.random.default_rng(42)
    nside = 16
    npix = hp.nside2npix(nside)

    # zero-mean random map for stability
    m = rng.normal(size=npix).astype(float)
    m -= m.mean()

    # consistent lmax; don't use deprecated/unsupported args
    lmax = 3 * nside - 1
    alm = hp.map2alm(m, lmax=lmax)
    m2 = hp.alm2map(alm, nside=nside, lmax=lmax)

    corr = np.corrcoef(m, m2)[0, 1]
    assert corr >= 0.85
