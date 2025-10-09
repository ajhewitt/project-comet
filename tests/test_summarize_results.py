from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="summary CLI tests require numpy")

import numpy as np


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_results_creates_figures(tmp_path, monkeypatch):
    module = _load_script("summarize_results", Path("scripts/summarize_results.py"))

    delta = np.array([0.1, -0.2, 0.3], dtype=float)
    delta_path = tmp_path / "delta.npy"
    np.save(delta_path, delta)

    cov = np.diag([0.04, 0.09, 0.01])
    cov_path = tmp_path / "cov.npy"
    np.save(cov_path, cov)

    ell = np.array([10.0, 20.0, 30.0])
    cl_data = np.array([0.5, 0.6, 0.7])
    cl_theory = np.array([0.45, 0.55, 0.65])
    sigma = np.array([0.1, 0.2, 0.15])
    delta_cross = cl_data - cl_theory
    z = delta_cross / sigma
    cross_path = tmp_path / "cross.npz"
    np.savez(
        cross_path,
        ell=ell,
        cl_data=cl_data,
        cl_theory=cl_theory,
        sigma=sigma,
        delta=delta_cross,
        z=z,
    )

    outdir = tmp_path / "figures"
    summary_path = tmp_path / "summary.json"
    messages: list[str] = []
    monkeypatch.setattr(module, "summary_line", lambda msg: messages.append(msg), raising=False)

    module.main(
        [
            "--delta",
            str(delta_path),
            "--cov",
            str(cov_path),
            "--cross",
            str(cross_path),
            "--outdir",
            str(outdir),
            "--summary",
            str(summary_path),
        ]
    )

    for name in [
        "delta_ell.png",
        "null_hist.png",
        "cross_spectrum.png",
        "cross_z_scores.png",
    ]:
        assert (outdir / name).exists()

    payload = json.loads(summary_path.read_text())
    assert "summary" in payload and payload["summary"]
    assert any('"valid_bins": 3' in msg for msg in messages)


def test_summarize_results_warns_when_no_covariance(tmp_path, monkeypatch):
    module = _load_script("summarize_results", Path("scripts/summarize_results.py"))

    delta = np.array([0.0, 0.0], dtype=float)
    delta_path = tmp_path / "delta.npy"
    np.save(delta_path, delta)

    ell = np.array([10.0, 20.0], dtype=float)
    cl_data = np.array([1.0, 2.0], dtype=float)
    cl_theory = np.array([0.5, 1.5], dtype=float)
    cross_path = tmp_path / "cross.npz"
    np.savez(
        cross_path,
        ell=ell,
        cl_data=cl_data,
        cl_theory=cl_theory,
        sigma=np.zeros_like(cl_data),
        delta=cl_data - cl_theory,
        z=np.zeros_like(cl_data),
    )

    outdir = tmp_path / "figures"
    summary_path = tmp_path / "summary.json"
    messages: list[str] = []
    monkeypatch.setattr(module, "summary_line", lambda msg: messages.append(msg), raising=False)

    module.main(
        [
            "--delta",
            str(delta_path),
            "--cov",
            str(tmp_path / "missing_cov.npy"),
            "--cross",
            str(cross_path),
            "--outdir",
            str(outdir),
            "--summary",
            str(summary_path),
        ]
    )

    payload = json.loads(summary_path.read_text())
    assert any("z-scores unavailable" in line for line in payload["summary"])
    assert any('"valid_bins": 0' in msg for msg in messages)
