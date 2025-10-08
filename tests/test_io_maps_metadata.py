from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="metadata ingestion tests require numpy")
pytest.importorskip("healpy", reason="metadata ingestion tests require healpy")

import healpy as hp
import numpy as np

from comet.io_maps import read_fits_map_with_meta


def _write_mock_map(path: Path, *, unit: str, coord: str = "G", ordering: str = "RING") -> None:
    nside = 8
    data = np.arange(hp.nside2npix(nside), dtype=float)
    extra_header = [
        ("FWHM", 5.0, "Effective beam FWHM (arcmin)"),
        ("PIXWIN", "PLANCK_PIXEL_WINDOW", "Pixel window reference"),
    ]
    nest = ordering.upper() != "RING"
    hp.write_map(
        path.as_posix(),
        data,
        nest=nest,
        coord=coord,
        column_units=unit,
        overwrite=True,
        extra_header=extra_header,
    )


def test_read_fits_map_with_meta_extracts_metadata(tmp_path: Path) -> None:
    path = tmp_path / "smica.fits"
    _write_mock_map(path, unit="K_CMB")

    result = read_fits_map_with_meta(path)

    assert result.metadata.unit == "K_CMB"
    assert result.metadata.ordering == "RING"
    assert result.metadata.coord_system == "G"
    assert result.metadata.pixel_window == "PLANCK_PIXEL_WINDOW"
    assert result.metadata.beam_fwhm_arcmin == 5.0

    meta_path = result.write_metadata_json(tmp_path / "metadata.json")
    payload = json.loads(meta_path.read_text())
    assert payload["unit"] == "K_CMB"
    assert payload["nside"] == 8
    assert payload["npix"] == result.pixels.size


def test_check_data_script_writes_metadata_and_summary(tmp_path: Path) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts_dir / "check_data.py"

    temp_map = tmp_path / "cmb.fits"
    lens_map = tmp_path / "kappa.fits"
    _write_mock_map(temp_map, unit="K_CMB")
    _write_mock_map(lens_map, unit="DIMENSIONLESS")

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        "\n".join(
            [
                f"temperature_map: {temp_map.as_posix()}",
                f"lensing_map: {lens_map.as_posix()}",
            ]
        )
    )

    metadata_dir = tmp_path / "metadata"
    summary_path = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            "--paths",
            paths_yaml.as_posix(),
            "--metadata-dir",
            metadata_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    temp_meta = metadata_dir / "cmb.metadata.json"
    lens_meta = metadata_dir / "kappa.metadata.json"
    assert temp_meta.exists()
    assert lens_meta.exists()

    summary = json.loads(summary_path.read_text())
    assert {entry["unit"] for entry in summary["maps"]} == {"K_CMB", "DIMENSIONLESS"}


def test_check_data_script_fails_for_unexpected_units(tmp_path: Path) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts_dir / "check_data.py"

    temp_map = tmp_path / "cmb.fits"
    _write_mock_map(temp_map, unit="MJy/sr")

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(f"temperature_map: {temp_map.as_posix()}\n")

    proc = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            "--paths",
            paths_yaml.as_posix(),
            "--metadata-dir",
            (tmp_path / "metadata").as_posix(),
            "--summary",
            (tmp_path / "summary.json").as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert "expected units" in proc.stderr
