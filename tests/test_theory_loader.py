from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="theory loader tests require numpy")

import numpy as np

from comet.theory import TheoryCls, load_theory


@pytest.mark.unit
def test_load_theory_npz(tmp_path: Path):
    ell = np.arange(2, 10)
    data = {
        "ell": ell,
        "cl_tt": np.full_like(ell, 1e-10, dtype=float),
        "cl_kk": np.full_like(ell, 2e-10, dtype=float),
        "cl_tk": np.full_like(ell, 1e-11, dtype=float),
    }
    path = tmp_path / "theory.npz"
    np.savez(path, **data)

    theory = load_theory(path)
    assert isinstance(theory, TheoryCls)
    assert np.allclose(theory.ell, ell)
    assert theory.lmax == int(ell[-1])


@pytest.mark.unit
def test_load_theory_text(tmp_path: Path):
    ell = np.arange(2, 6)
    matrix = np.column_stack([ell, ell * 1e-10, ell * 2e-10, ell * 3e-11])
    path = tmp_path / "theory.txt"
    np.savetxt(path, matrix)

    theory = load_theory(path)
    assert theory.ell.size == ell.size
    truncated = theory.truncate(4)
    assert truncated.ell.max() == 4


@pytest.mark.unit
def test_load_theory_negative_power_raises(tmp_path: Path):
    ell = np.arange(2, 5)
    np.savez(
        tmp_path / "bad.npz",
        ell=ell,
        cl_tt=np.array([1.0, -1.0, 0.5]),
        cl_kk=np.ones_like(ell),
        cl_tk=np.zeros_like(ell),
    )

    with pytest.raises(ValueError):
        load_theory(tmp_path / "bad.npz")


@pytest.mark.unit
def test_theory_truncate_requires_overlap(tmp_path: Path):
    ell = np.arange(5, 10)
    np.savez(
        tmp_path / "shifted.npz",
        ell=ell,
        cl_tt=np.ones_like(ell),
        cl_kk=np.ones_like(ell),
        cl_tk=np.ones_like(ell),
    )
    theory = load_theory(tmp_path / "shifted.npz")
    with pytest.raises(ValueError):
        theory.truncate(2)


@pytest.mark.unit
def test_as_synalm_array_pads_missing_low_ell():
    ell = np.arange(2, 5)
    theory = TheoryCls(
        ell=ell,
        cl_tt=np.full_like(ell, 1e-9, dtype=float),
        cl_kk=np.full_like(ell, 2e-9, dtype=float),
        cl_tk=np.full_like(ell, 5e-10, dtype=float),
    )

    autos_tt, autos_kk, cross = theory.as_synalm_array(lmax=6)

    assert autos_tt.shape == (7,)
    assert np.allclose(autos_tt[:2], 0.0)
    assert np.allclose(autos_tt[ell], theory.cl_tt)
    assert autos_kk.shape == (7,)
    assert cross.shape == (7,)
    assert np.allclose(cross[ell], theory.cl_tk)
