#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from commutator_common import (
    WindowConfig,
    build_mask,
    load_bins_from_prereg,
    load_windows_from_prereg,
    nm_bandpowers,
    nm_bins_from_params,
    nm_field_from_scalar,
    infer_bin_lmax,
    read_map,
    save_json,
    summary_line,
)


def main():
    ap = argparse.ArgumentParser(description="Run ordering A to B (CMB -> phi)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cmb", default="COM_CompMap_CMB-smica_2048_R1.20.fits")
    ap.add_argument("--phi", default="COM_CompMap_Lensing_2048_R1.10.fits")
    ap.add_argument("--quick-nside", type=int, default=256, help="Downsample before analysis")
    ap.add_argument("--nlb", type=int, default=50)
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--lmin", type=int, default=None)
    ap.add_argument("--threshold-sigma", type=float, default=5.0)
    ap.add_argument("--apod-arcmin", type=float, default=30.0)
    ap.add_argument(
        "--prereg",
        type=Path,
        default=Path("config/prereg.yaml"),
        help="Path to prereg YAML for bin configuration",
    )
    ap.add_argument("--out", default="artifacts/order_A_to_B.npz")
    args = ap.parse_args()

    d = Path(args.data_dir)
    cmb = read_map(d / args.cmb, quick_nside=args.quick_nside)
    phi = read_map(d / args.phi, quick_nside=args.quick_nside)
    nside = hp.get_nside(cmb)

    mask = (
        build_mask(
            cmb,
            threshold_sigma=args.threshold_sigma,
            apod_arcmin=args.apod_arcmin,
        )
        * build_mask(
            phi,
            threshold_sigma=args.threshold_sigma,
            apod_arcmin=args.apod_arcmin,
        )
    ).astype(float)
    mask = np.clip(mask, 0.0, 1.0)

    bins = None
    bins_meta = None
    windows_cfg: WindowConfig | None = None
    try:
        bins, bins_meta = load_bins_from_prereg(args.prereg, nside=nside)
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - keep fallback path available
        summary_line(f"failed to load prereg bins: {exc}; falling back to CLI params")

    if bins is None:
        bins = nm_bins_from_params(
            nside=nside,
            lmax=args.lmax,
            lmin=args.lmin,
            nlb=args.nlb,
        )

    try:
        windows_cfg = load_windows_from_prereg(args.prereg)
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - keep CLI resilient
        summary_line(f"failed to load window config: {exc}; using defaults")
        windows_cfg = None

    fallback_lmax: list[int | None] = []
    if args.lmax is not None:
        fallback_lmax.append(int(args.lmax))
    fallback_lmax.append(3 * nside - 1)
    field_lmax = infer_bin_lmax(
        bins,
        bins_meta=bins_meta,
        fallbacks=fallback_lmax,
    )
    f1 = nm_field_from_scalar(cmb, mask, lmax=field_lmax)
    f2 = nm_field_from_scalar(phi, mask, lmax=field_lmax)
    cl = nm_bandpowers(
        f1,
        f2,
        bins,
        window_config=windows_cfg,
        field_names=("cmb", "phi"),
    )

    np.savez(Path(args.out), cl=cl, nside=nside, nlb=args.nlb)
    save_json(
        {
            "order": "A_to_B",
            "nside": nside,
            "nbins": int(cl.size),
            "mask_threshold_sigma": args.threshold_sigma,
            "mask_apod_arcmin": args.apod_arcmin,
            "bins_source": "prereg" if bins_meta is not None else "cli",
            "bins": bins_meta,
            "windows": windows_cfg.to_metadata() if windows_cfg is not None else None,
        },
        Path(args.out).with_suffix(".json"),
    )
    summary_line(f"wrote {args.out} with {cl.size} bins at nside={nside}")


if __name__ == "__main__":
    main()
