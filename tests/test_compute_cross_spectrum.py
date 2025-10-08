from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="cross-spectrum CLI tests require numpy")

import numpy as np


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_cross_spectrum_outputs(tmp_path, monkeypatch):
    module = _load_script_module(
        "compute_cross_spectrum", Path("scripts/compute_cross_spectrum.py")
    )

    cl_a = np.array([1.0, 2.0, 3.0], dtype=float)
    cl_b = np.array([1.2, 1.8, 3.2], dtype=float)
    order_a = tmp_path / "order_a.npz"
    order_b = tmp_path / "order_b.npz"
    np.savez(order_a, cl=cl_a, nside=256)
    np.savez(order_b, cl=cl_b, nside=256)

    diag = np.array([0.25, 0.16, 0.09], dtype=float)
    cov = tmp_path / "cov.npy"
    np.save(cov, np.diag(diag))

    ell = np.arange(0, 40, dtype=float)
    theory_path = tmp_path / "theory.npz"
    np.savez(
        theory_path,
        ell=ell,
        cl_tt=0.2 * ell,
        cl_kk=0.1 * ell,
        cl_tk=0.05 * ell,
    )

    messages: list[str] = []
    monkeypatch.setattr(module, "summary_line", lambda msg: messages.append(msg), raising=False)

    out = tmp_path / "cross.npz"
    summary = tmp_path / "summary.json"
    module.main(
        [
            "--order-a",
            str(order_a),
            "--order-b",
            str(order_b),
            "--theory",
            str(theory_path),
            "--prereg",
            str(tmp_path / "missing_prereg.yaml"),
            "--lmin",
            "10",
            "--nlb",
            "5",
            "--cov",
            str(cov),
            "--out",
            str(out),
            "--summary",
            str(summary),
        ]
    )

    assert out.exists()
    data = np.load(out)
    np.testing.assert_allclose(data["ell"], np.array([12.0, 17.0, 22.0]))
    np.testing.assert_allclose(data["cl_data"], np.array([1.1, 1.9, 3.1]))
    np.testing.assert_allclose(data["cl_theory"], np.array([0.6, 0.85, 1.1]))
    np.testing.assert_allclose(data["sigma"], np.sqrt(diag))
    np.testing.assert_allclose(data["delta"], np.array([0.5, 1.05, 2.0]))
    np.testing.assert_allclose(data["z"], np.array([1.0, 2.625, 6.66666667]), rtol=1e-6)

    summary_payload = json.loads(summary.read_text())
    assert summary_payload["nbins"] == 3
    assert messages and "cross-spectrum" in messages[-1]
