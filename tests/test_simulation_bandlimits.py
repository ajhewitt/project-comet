from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="simulation helpers require numpy")

from comet.simulations import resolve_simulation_bandlimits


def test_resolve_bandlimits_prefers_strictest_cap():
    bins = SimpleNamespace(lmax=200)

    sim_lmax, field_lmax = resolve_simulation_bandlimits(
        bins,
        requested_lmax=512,
        theory_lmax=1024,
        nside=256,
    )

    assert sim_lmax == 200
    assert field_lmax == 200


def test_resolve_bandlimits_uses_bin_edges_when_missing_attribute():
    class EdgeBins:
        def get_ell_list(self):
            return ([0, 10, 20], [49, 74, 99])

    bins = EdgeBins()

    sim_lmax, field_lmax = resolve_simulation_bandlimits(
        bins,
        requested_lmax=None,
        theory_lmax=400,
        nside=128,
    )

    assert sim_lmax == 99
    assert field_lmax == 99


def test_resolve_bandlimits_requires_positive_nside():
    bins = SimpleNamespace(lmax=10)

    with pytest.raises(ValueError):
        resolve_simulation_bandlimits(
            bins,
            requested_lmax=64,
            theory_lmax=64,
            nside=0,
        )
