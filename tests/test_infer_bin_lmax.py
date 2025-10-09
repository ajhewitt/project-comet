from types import SimpleNamespace

from commutator_common import infer_bin_lmax


def test_infer_bin_lmax_prefers_attribute():
    bins = SimpleNamespace(lmax=512)
    assert infer_bin_lmax(bins) == 512


def test_infer_bin_lmax_uses_metadata_before_edges():
    bins = SimpleNamespace()
    meta = {"lmax": 321}
    assert infer_bin_lmax(bins, bins_meta=meta) == 321


def test_infer_bin_lmax_from_edge_lists():
    class EdgeBins:
        def get_ell_list(self):
            return ([0, 10], [25, 49])

    bins = EdgeBins()
    assert infer_bin_lmax(bins) == 49


def test_infer_bin_lmax_falls_back_to_requested_values():
    bins = SimpleNamespace()
    assert infer_bin_lmax(bins, fallbacks=(None, 256)) == 256
