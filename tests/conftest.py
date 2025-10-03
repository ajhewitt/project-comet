from __future__ import annotations

import healpy as hp
import numpy as np
import pytest


@pytest.fixture
def tiny_maps():
    nside = 32
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(42)
    cmb = rng.normal(0, 1e-5, size=npix)
    phi = rng.normal(0, 1e-4, size=npix)
    return {"cmb": cmb, "phi": phi, "nside": nside}
