import pytest

pytest.importorskip("numpy", reason="window tests require numpy")
pytest.importorskip("healpy", reason="window tests require healpy")

import numpy as np

from comet.namaster_utils import (
    WindowConfig,
    apply_window_corrections,
    gaussian_beam_window,
    parse_window_config,
    pixel_window,
)


def test_parse_window_config_handles_mapping():
    raw = {
        "pixel": {"apply": True},
        "beam": {
            "deconvolve": True,
            "fwhm_arcmin": {"cmb": 5.0},
        },
    }
    cfg = parse_window_config(raw)
    assert isinstance(cfg, WindowConfig)
    assert cfg.apply_pixel_window is True
    assert cfg.deconvolve_beam is True
    assert cfg.beam_for("cmb") == pytest.approx(5.0, rel=1e-6)
    assert cfg.beam_for("phi") is None


def test_beam_deconvolution_changes_high_ell_tail():
    ells = np.linspace(10, 500, num=32, dtype=float)
    cl = np.ones_like(ells)
    beam = gaussian_beam_window(30.0, lmax=int(ells.max()))

    corrected = apply_window_corrections(cl, ells, beam_windows=(beam, beam))
    baseline = apply_window_corrections(cl, ells)

    # NaMaster-style Gaussian beams suppress high-ℓ power; deconvolution restores it.
    assert corrected[-1] > baseline[-1]
    assert corrected[-1] > corrected[0]


def test_pixel_window_evaluation_is_finite():
    nside = 64
    lmax = 3 * nside - 1
    window = pixel_window(nside, lmax=lmax)
    ells = np.linspace(0, lmax, num=16)
    response = apply_window_corrections(
        np.ones_like(ells),
        ells,
        pixel_windows=(window, window),
    )

    assert np.all(np.isfinite(response))
    assert response[0] == pytest.approx(1.0)
