#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np
from commutator_common import build_mask, effective_f_sky, read_map, save_json, summary_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined mask with apodization")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cmb", default="COM_CompMap_CMB-smica_2048_R1.20.fits")
    parser.add_argument("--phi", default="COM_CompMap_Lensing_2048_R1.10.fits")
    parser.add_argument(
        "--quick-nside", type=int, default=256, help="Optional downgrade before masking"
    )
    parser.add_argument("--threshold-sigma", type=float, default=5.0)
    parser.add_argument("--apod-arcmin", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/mask.npy"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cmb = read_map(data_dir / args.cmb, quick_nside=args.quick_nside)
    phi = read_map(data_dir / args.phi, quick_nside=args.quick_nside)

    mask_cmb = build_mask(cmb, threshold_sigma=args.threshold_sigma, apod_arcmin=args.apod_arcmin)
    mask_phi = build_mask(phi, threshold_sigma=args.threshold_sigma, apod_arcmin=args.apod_arcmin)
    mask = np.clip(mask_cmb * mask_phi, 0.0, 1.0)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask)

    metadata = {
        "cmb": args.cmb,
        "phi": args.phi,
        "quick_nside": int(hp.get_nside(cmb)),
        "threshold_sigma": float(args.threshold_sigma),
        "apod_arcmin": float(args.apod_arcmin),
        "f_sky": effective_f_sky(mask),
        "path": out_path.as_posix(),
    }
    save_json(metadata, out_path.with_suffix(".json"))
    summary_line(
        "built mask {path} with f_sky={f_sky:.3f}".format(
            path=out_path.as_posix(), f_sky=metadata["f_sky"]
        )
    )


if __name__ == "__main__":
    main()
