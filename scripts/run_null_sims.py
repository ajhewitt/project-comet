#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from comet.simulations import (
    SimulationGeometry,
    estimate_delta_covariance,
    resolve_simulation_bandlimits,
)
from comet.theory import load_theory
from commutator_common import (
    build_mask,
    nm_bins_from_params,
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
    ap.add_argument("--theory", type=Path, required=True, help="Path to theory Cl file (.npz/.txt)")
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
    bins = nm_bins_from_params(nside=nside, lmax=args.lmax, nlb=args.nlb)

    theory = load_theory(args.theory)
    sim_lmax, field_lmax = resolve_simulation_bandlimits(
        bins,
        requested_lmax=args.lmax,
        theory_lmax=theory.lmax,
        nside=nside,
    )
    geom = SimulationGeometry(
        mask=mask,
        bins=bins,
        nside=nside,
        lmax=sim_lmax,
        field_lmax=field_lmax,
    )

    cov = estimate_delta_covariance(theory, geom, nsims=args.nsims, rng=rng)

    save_npy(cov, Path(args.out_cov))
    summary_line(
        f"wrote {args.out_cov} with shape {cov.shape} nside={nside}"
        f" lmax={sim_lmax} field_lmax={field_lmax} nsims={args.nsims}"
    )


if __name__ == "__main__":
    main()
