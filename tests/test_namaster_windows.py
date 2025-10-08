from __future__ import annotations

import math
from pathlib import Path

import pytest

from comet.namaster_utils import (
    WindowConfig,
    apply_window_corrections,
    gaussian_beam_window,
    parse_window_config,
)


def test_parse_window_config_handles_mapping() -> None:
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


def test_window_config_to_metadata_sorts_keys() -> None:
    cfg = WindowConfig(
        apply_pixel_window=True,
        deconvolve_beam=True,
        beam_fwhm_arcmin={"cmb": 5.0, "default": 7.5},
    )
    metadata = cfg.to_metadata()
    assert metadata["apply_pixel_window"] is True
    assert metadata["deconvolve_beam"] is True
    assert list(metadata["beam_fwhm_arcmin"].keys()) == ["cmb", "default"]


def test_apply_window_corrections_combines_all_windows() -> None:
    ells = [float(i) for i in range(5)]
    cl = [1.0 for _ in ells]
    pixel_windows = (
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [1.0, 0.95, 0.9, 0.85, 0.8],
    )
    beam_windows = (
        [1.0, 0.97, 0.94, 0.91, 0.88],
        [1.0, 0.96, 0.92, 0.88, 0.84],
    )

    corrected = apply_window_corrections(
        cl,
        ells,
        pixel_windows=pixel_windows,
        beam_windows=beam_windows,
    )
    corrected_list = corrected.tolist() if hasattr(corrected, "tolist") else list(corrected)

    total_response = [
        max(
            1e-12,
            pixel_windows[0][i]
            * pixel_windows[1][i]
            * beam_windows[0][i]
            * beam_windows[1][i],
        )
        for i in range(len(ells))
    ]
    expected = [cl[i] / total_response[i] for i in range(len(ells))]
    assert all(abs(a - b) < 1e-12 for a, b in zip(corrected_list, expected))


def test_beam_deconvolution_changes_high_ell_tail() -> None:
    lmax = 64
    ells = [float(i) for i in range(lmax + 1)]
    cl = [1.0 for _ in ells]
    beam = gaussian_beam_window(30.0, lmax=lmax)

    corrected = apply_window_corrections(cl, ells, beam_windows=(beam, beam))
    baseline = apply_window_corrections(cl, ells)

    corrected_list = corrected.tolist() if hasattr(corrected, "tolist") else list(corrected)
    baseline_list = baseline.tolist() if hasattr(baseline, "tolist") else list(baseline)

    assert corrected_list[-1] > baseline_list[-1]
    assert corrected_list[-1] > corrected_list[0]


def test_gaussian_beam_window_matches_reference_formula() -> None:
    lmax = 16
    fwhm = 12.0
    beam = gaussian_beam_window(fwhm, lmax=lmax)
    sigma = math.radians(fwhm / 60.0) / math.sqrt(8.0 * math.log(2.0))
    expected = [math.exp(-0.5 * ell * (ell + 1.0) * sigma**2) for ell in range(lmax + 1)]
    beam_list = beam.tolist() if hasattr(beam, "tolist") else list(beam)
    assert all(abs(a - b) < 1e-12 for a, b in zip(beam_list, expected))


def test_load_windows_from_prereg_uses_prereg_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from commutator_common import load_windows_from_prereg

    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(
        "windows: {pixel: yes, beam: {deconvolve: true, fwhm_arcmin: {default: 5}}}\n"
    )

    fake_prereg = {
        "windows": {
            "pixel": True,
            "beam": {"deconvolve": True, "fwhm_arcmin": {"default": 5}},
        }
    }
    monkeypatch.setattr("_commutator_common_impl.load_prereg", lambda path: fake_prereg)

    cfg = load_windows_from_prereg(prereg)
    assert cfg.apply_pixel_window is True
    assert cfg.deconvolve_beam is True
    assert cfg.beam_for("cmb") == pytest.approx(5.0, rel=1e-6)
