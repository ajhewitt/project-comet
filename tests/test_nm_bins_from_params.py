import types

import pytest

pytest.importorskip("numpy", reason="nm bin helper tests require numpy")

import numpy as np

from commutator_common import nm_bins_from_params


class _FakeNmtBin:
    def __init__(self, ell_min, ell_max):
        self.ell_min = np.asarray(ell_min)
        self.ell_max = np.asarray(ell_max)

    @classmethod
    def from_nside_linear(cls, *, nside, nlb, lmin=None):  # pragma: no cover - signature only
        # This stub mimics older NaMaster releases that rejected an ``lmax`` keyword.
        return cls([], [])

    @classmethod
    def from_edges(cls, ell_min, ell_max):
        return cls(ell_min, ell_max)


@pytest.fixture
def fake_nmt(monkeypatch):
    module = types.SimpleNamespace(NmtBin=_FakeNmtBin)
    monkeypatch.setattr("_commutator_common_impl._nmt", module, raising=False)
    monkeypatch.setattr("_commutator_common_impl.nmt", module, raising=False)
    return module


def test_nm_bins_from_params_falls_back_when_lmax_not_supported(fake_nmt):
    bins = nm_bins_from_params(nside=16, nlb=10, lmax=47)

    np.testing.assert_array_equal(bins.ell_min, np.array([0, 10, 20, 30, 40]))
    np.testing.assert_array_equal(bins.ell_max, np.array([9, 19, 29, 39, 47]))
