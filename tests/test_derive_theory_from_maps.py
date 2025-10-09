"""Tests for :mod:`scripts.derive_theory_from_maps`."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types
import importlib.machinery

import pytest

np = pytest.importorskip("numpy")

if "healpy" not in sys.modules:
    stub = types.ModuleType("healpy")
    stub.__spec__ = importlib.machinery.ModuleSpec("healpy", loader=None)
    stub.read_map = None
    stub.anafast = None
    sys.modules["healpy"] = stub

derive = importlib.import_module("scripts.derive_theory_from_maps")


def test_compute_spectra(monkeypatch):
    cmb_path = Path("cmb.fits")
    kappa_path = Path("kappa.fits")
    lmax = 3

    cmb_map = np.array([1.0, 2.0, 3.0])
    kappa_map = np.array([4.0, 5.0, 6.0])

    def fake_read_map(path, field):
        assert field == 0
        if path == cmb_path:
            return cmb_map
        if path == kappa_path:
            return kappa_map
        raise AssertionError(f"unexpected path {path}")

    def fake_anafast(*arrays, lmax):
        assert lmax == expected_lmax[0]
        if len(arrays) == 1 and arrays[0] is cmb_map:
            return np.array([0.1, 0.2, 0.3, 0.4])
        if len(arrays) == 1 and arrays[0] is kappa_map:
            return np.array([1.1, 1.2, 1.3, 1.4])
        if len(arrays) == 2 and arrays[0] is cmb_map and arrays[1] is kappa_map:
            return np.array([2.1, 2.2, 2.3, 2.4])
        raise AssertionError("unexpected ana fast call")

    expected_lmax = (lmax,)

    monkeypatch.setattr(derive.hp, "read_map", fake_read_map)
    monkeypatch.setattr(derive.hp, "anafast", fake_anafast)

    ell, cl_tt, cl_kk, cl_tk = derive.compute_spectra(cmb_path, kappa_path, lmax)

    np.testing.assert_array_equal(ell, np.arange(4))
    np.testing.assert_array_equal(cl_tt, np.array([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_array_equal(cl_kk, np.array([1.1, 1.2, 1.3, 1.4]))
    np.testing.assert_array_equal(cl_tk, np.array([2.1, 2.2, 2.3, 2.4]))


def test_write_text(tmp_path):
    output = tmp_path / "theory.txt"
    ell = np.array([0, 1])
    cl_tt = np.array([0.1, 0.2])
    cl_kk = np.array([1.1, 1.2])
    cl_tk = np.array([2.1, 2.2])

    derive.write_text(output, ell, cl_tt, cl_kk, cl_tk)

    content = output.read_text().splitlines()
    assert content[0].startswith("#")
    loaded = np.loadtxt(output)
    np.testing.assert_array_equal(loaded[:, 0], ell)
    np.testing.assert_allclose(loaded[:, 1], cl_tt)
    np.testing.assert_allclose(loaded[:, 2], cl_kk)
    np.testing.assert_allclose(loaded[:, 3], cl_tk)


def test_write_npz(tmp_path):
    output = tmp_path / "theory.npz"
    ell = np.array([0, 1])
    cl_tt = np.array([0.1, 0.2])
    cl_kk = np.array([1.1, 1.2])
    cl_tk = np.array([2.1, 2.2])

    derive.write_npz(output, ell, cl_tt, cl_kk, cl_tk)

    loaded = np.load(output)
    np.testing.assert_array_equal(loaded["ell"], ell)
    np.testing.assert_array_equal(loaded["cl_tt"], cl_tt)
    np.testing.assert_array_equal(loaded["cl_kk"], cl_kk)
    np.testing.assert_array_equal(loaded["cl_tk"], cl_tk)


def test_main_invokes_helpers(monkeypatch, tmp_path):
    args = [
        "--cmb-map",
        "cmb.fits",
        "--kappa-map",
        "kappa.fits",
        "--lmax",
        "100",
        "--output-text",
        str(tmp_path / "out.txt"),
        "--output-npz",
        str(tmp_path / "out.npz"),
    ]

    expected_ell = np.array([0, 1])
    expected_tt = np.array([0.1, 0.2])
    expected_kk = np.array([1.1, 1.2])
    expected_tk = np.array([2.1, 2.2])

    def fake_compute(cmb_map, kappa_map, lmax):
        assert cmb_map == Path("cmb.fits")
        assert kappa_map == Path("kappa.fits")
        assert lmax == 100
        return expected_ell, expected_tt, expected_kk, expected_tk

    called = {"text": False, "npz": False}

    def fake_write_text(path, ell, cl_tt, cl_kk, cl_tk):
        called["text"] = True
        assert path == tmp_path / "out.txt"
        np.testing.assert_array_equal(ell, expected_ell)
        np.testing.assert_array_equal(cl_tt, expected_tt)
        np.testing.assert_array_equal(cl_kk, expected_kk)
        np.testing.assert_array_equal(cl_tk, expected_tk)

    def fake_write_npz(path, ell, cl_tt, cl_kk, cl_tk):
        called["npz"] = True
        assert path == tmp_path / "out.npz"
        np.testing.assert_array_equal(ell, expected_ell)
        np.testing.assert_array_equal(cl_tt, expected_tt)
        np.testing.assert_array_equal(cl_kk, expected_kk)
        np.testing.assert_array_equal(cl_tk, expected_tk)

    monkeypatch.setattr(derive, "compute_spectra", fake_compute)
    monkeypatch.setattr(derive, "write_text", fake_write_text)
    monkeypatch.setattr(derive, "write_npz", fake_write_npz)

    exit_code = derive.main(args)

    assert exit_code == 0
    assert called == {"text": True, "npz": True}
