#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np
from commutator_common import (
    build_mask,
    nm_bandpowers,
    nm_bins_from_params,
    nm_field_from_scalar,
    read_map,
    save_npy,
    summary_line,
)


def _standardize(deltas: np.ndarray) -> np.ndarray:
    # Optional helper if you ever want to sanity-check per-bin scaling
    mu = deltas.mean(axis=0)
    sig = deltas.std(axis=0, ddof=1)
    sig = np.where(sig <= 0, 1.0, sig)
    return (deltas - mu) / sig


def main():
    ap = argparse.ArgumentParser(description="Run null simulations to estimate Cov[Δ]")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cmb", default="COM_CompMap_CMB-smica_2048_R1.20.fits")
    ap.add_argument("--phi", default="COM_CompMap_Lensing_2048_R1.10.fits")
    ap.add_argument("--quick-nside", type=int, default=256)
    ap.add_argument("--nlb", type=int, default=50)
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--nsims", type=int, default=200)  # bumped for better conditioning
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-cov", default="artifacts/cov_delta.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Geometry from data (mask/binning only)
    d = Path(args.data_dir)
    cmb = read_map(d / args.cmb, quick_nside=args.quick_nside)
    phi = read_map(d / args.phi, quick_nside=args.quick_nside)
    nside = hp.get_nside(cmb)
    mask = (build_mask(cmb) * build_mask(phi)).astype(float)
    nb = nm_bins_from_params(nside=nside, lmax=args.lmax, nlb=args.nlb)

    deltas = []
    for _ in range(args.nsims):
        # White Gaussian placeholders; replace with theory Cl sims when ready
        m1 = rng.normal(0, 1e-5, size=cmb.size)
        m2 = rng.normal(0, 1e-5, size=phi.size)
        f1 = nm_field_from_scalar(m1, mask)
        f2 = nm_field_from_scalar(m2, mask)
        cl_ab = nm_bandpowers(f1, f2, nb)
        cl_ba = nm_bandpowers(f2, f1, nb)
        deltas.append(cl_ab - cl_ba)

    D = np.vstack(deltas)  # [nsims, nbins]
    # Use unbiased covariance (rowvar=False), ddof=1 by default
    C = np.cov(D, rowvar=False)
    # Tiny floor to the diagonal helps conditioning without biasing structure
    eps = 1e-10 * (np.trace(C) / max(C.shape[0], 1))
    C += eps * np.eye(C.shape[0])

    save_npy(C, Path(args.out_cov))
    summary_line(f"wrote {args.out_cov} with shape {C.shape}")


if __name__ == "__main__":
    main()
