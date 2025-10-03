from __future__ import annotations

import os
from pathlib import Path

import pytest

from comet.io_maps import map_info, read_fits_map


@pytest.mark.skipif(os.getenv("COMET_DATA_DIR") is None, reason="no local data")
def test_read_local_maps():
    dd = Path(os.environ["COMET_DATA_DIR"])
    m1 = read_fits_map(dd / "COM_CompMap_Lensing_2048_R1.10.fits")
    m2 = read_fits_map(dd / "COM_CompMap_CMB-smica_2048_R1.20.fits")
    for m in (m1, m2):
        info = map_info(m)
        assert info["nside"] in (1024, 2048)
        assert 0.5 < info["f_sky"] <= 1.0
