from __future__ import annotations

import numpy as np

try:
    import pymaster as nmt  # type: ignore
except Exception as e:  # pragma: no cover
    # Keep import errors obvious during local runs; tests can skip if not installed.
    raise RuntimeError(
        "pymaster (NaMaster) is required for namaster_utils. "
        "Install it from conda-forge as 'namaster'."
    ) from e


def make_bins(lmax: int, nlb: int) -> nmt.NmtBin:
    """
    Construct a simple linear-ℓ binning up to lmax with bin width nlb.
    Swap this out for explicit bin edges from prereg when ready.
    """
    if lmax <= 0 or nlb <= 0:
        raise ValueError("lmax and nlb must be positive")
    # NaMaster provides helpers for typical binning schemes.
    # For more control, use NmtBin.from_edges with explicit bounds.
    return nmt.NmtBin.from_lmax(lmax=lmax, nlb=nlb)


def field_from_map(m: np.ndarray, mask: np.ndarray | None = None) -> nmt.NmtField:
    """
    Build a spin-0 NaMaster field from a scalar map and an optional mask.
    """
    if mask is None:
        mask = np.isfinite(m).astype(float)
    return nmt.NmtField(mask, [m])


def bandpowers(f1: nmt.NmtField, f2: nmt.NmtField, b: nmt.NmtBin) -> np.ndarray:
    """
    Compute decoupled pseudo-C_ell bandpowers for two fields using a fresh workspace.
    """
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, b)
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = w.decouple_cell(cl_coupled)
    # cl_decoupled is shape (n_spectra, n_bins). For spin-0 autos, index 0.
    return cl_decoupled[0]
