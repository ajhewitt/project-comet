#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from commutator_common import save_npy, summary_line


def ecliptic_latitude(nside: int) -> np.ndarray:
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Convert to ecliptic latitude beta from equatorial (assume obliquity 23.439281 deg)
    eps = np.deg2rad(23.439281)
    sin_beta = np.sin(theta) * np.sin(eps) * np.sin(phi) + np.cos(theta) * np.cos(eps)
    beta = np.arcsin(sin_beta)
    return beta


def main():
    ap = argparse.ArgumentParser(description="Build simple context template basis.")
    ap.add_argument("--nside", type=int, default=256)
    ap.add_argument("--out", default="artifacts/templates.npy")
    args = ap.parse_args()

    beta = ecliptic_latitude(args.nside)
    T0 = np.ones_like(beta)
    T1 = beta
    T2 = beta**2
    T = np.vstack([T0, T1, T2])  # [n_templates, npix]

    save_npy(T, Path(args.out))
    summary_line(f"wrote {args.out} with {T.shape[0]} templates at nside={args.nside}")


if __name__ == "__main__":
    main()
