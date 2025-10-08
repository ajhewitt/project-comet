from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="mask tests require numpy")
pytest.importorskip("healpy", reason="mask tests require healpy")
pytest.importorskip("pymaster", reason="mask tests require pymaster")

import healpy as hp
import numpy as np
from commutator_common import nm_bins_from_config

from comet.config import load_prereg
from comet.masking import build_mask, effective_f_sky


def test_build_mask_f_sky_within_expected_range():
    nside = 32
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    m = np.cos(theta)

    mask = build_mask(m, threshold_sigma=1.0, apod_arcmin=30.0)
    f_sky = effective_f_sky(mask)

    assert 0.5 < f_sky < 0.9
    assert np.all(mask >= 0.0)
    assert np.all(mask <= 1.0)


def test_bins_follow_prereg_configuration():
    prereg = load_prereg(Path("config/prereg.yaml"))
    bins_cfg = prereg["ells"]["bins"]
    nside = 256
    bins = nm_bins_from_config(nside=nside, bins_cfg=bins_cfg)

    expected_bins = math.ceil((bins_cfg["lmax"] - bins_cfg["lmin"] + 1) / bins_cfg["nlb"])
    assert bins.get_n_bands() == expected_bins
