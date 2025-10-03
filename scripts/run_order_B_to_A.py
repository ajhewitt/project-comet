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
    save_json,
    summary_line,
)


def main():
    ap = argparse.ArgumentParser(description="Run ordering B to A (phi -> CMB)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cmb", default="COM_CompMap_CMB-smica_2048_R1.20.fits")
    ap.add_argument("--phi", default="COM_CompMap_Lensing_2048_R1.10.fits")
    ap.add_argument("--quick-nside", type=int, default=256, help="Downsample before analysis")
    ap.add_argument("--nlb", type=int, default=50)
    ap.add_argument("--lmax", type=int, default=None)  # kept for interface parity, not used
    ap.add_argument("--out", default="artifacts/order_B_to_A.npz")
    args = ap.parse_args()

    d = Path(args.data_dir)
    cmb = read_map(d / args.cmb, quick_nside=args.quick_nside)
    phi = read_map(d / args.phi, quick_nside=args.quick_nside)
    nside = hp.get_nside(cmb)

    mask = (build_mask(cmb) * build_mask(phi)).astype(float)
    bins = nm_bins_from_params(nside=nside, lmax=args.lmax, nlb=args.nlb)

    f1 = nm_field_from_scalar(phi, mask)
    f2 = nm_field_from_scalar(cmb, mask)
    cl = nm_bandpowers(f1, f2, bins)

    np.savez(Path(args.out), cl=cl, nside=nside, nlb=args.nlb)
    save_json(
        {"order": "B_to_A", "nside": nside, "nbins": int(cl.size)},
        Path(args.out).with_suffix(".json"),
    )
    summary_line(f"wrote {args.out} with {cl.size} bins at nside={nside}")


if __name__ == "__main__":
    main()
