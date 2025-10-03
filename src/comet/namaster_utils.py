from __future__ import annotations
import numpy as np
import pymaster as nmt

def make_bins(lmax: int, nlb: int) -> nmt.NmtBin:
    ells = np.arange(lmax + 1)
    return nmt.NmtBin.from_nside_linear(nside=lmax//2, nlb=nlb)  # replace with your choice

def field_from_map(m: np.ndarray, mask: np.ndarray | None = None) -> nmt.NmtField:
    if mask is None:
        mask = np.isfinite(m).astype(float)
    return nmt.NmtField(mask, [m])

def bandpowers(f1: nmt.NmtField, f2: nmt.NmtField, b: nmt.NmtBin) -> np.ndarray:
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f1, f2, b)
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = w.decouple_cell(cl_coupled)
    return cl_decoupled[0]
