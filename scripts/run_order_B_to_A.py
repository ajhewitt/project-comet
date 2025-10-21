#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import healpy as hp
import numpy as np

from commutator_common import (
    WindowConfig,
    build_mask,
    infer_bin_lmax,
    load_bins_from_prereg,
    load_windows_from_prereg,
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
    ap.add_argument(
        "--disable-prereg",
        action="store_true",
        help="Ignore prereg metadata and use CLI binning/mask parameters",
    )
    ap.add_argument("--out", default="artifacts/order_B_to_A.npz")
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
    if not args.disable_prereg:
        try:
            bins, bins_meta = load_bins_from_prereg(args.prereg, nside=nside)
        except FileNotFoundError:
            pass
        except Exception as exc:  # pragma: no cover
            summary_line(f"failed to load prereg bins: {exc}; falling back to CLI params")

    if bins is None:
        bins = nm_bins_from_params(
            nside=nside,
            lmax=args.lmax,
            lmin=args.lmin,
            nlb=args.nlb,
        )

    def _infer_bin_limits() -> tuple[int | None, int | None]:
        lmin_val = getattr(bins, "lmin", None)
        lmax_val = getattr(bins, "lmax", None)
        ell_list = getattr(bins, "get_ell_list", None)
        if callable(ell_list):
            try:
                groups = list(ell_list())
            except Exception:
                groups = []
            for group in groups:
                arr = np.asarray(group, dtype=int)
                if arr.size:
                    lmin_val = int(arr[0])
                    break
            for group in reversed(groups):
                arr = np.asarray(group, dtype=int)
                if arr.size:
                    lmax_val = int(arr[-1])
                    break
        return (
            None if lmin_val is None else int(lmin_val),
            None if lmax_val is None else int(lmax_val),
        )

    if not args.disable_prereg:
        try:
            windows_cfg = load_windows_from_prereg(args.prereg)
        except FileNotFoundError:
            pass
        except Exception as exc:  # pragma: no cover
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
    f1 = nm_field_from_scalar(phi, mask, lmax=field_lmax)
    f2 = nm_field_from_scalar(cmb, mask, lmax=field_lmax)
    cl = nm_bandpowers(
        f1,
        f2,
        bins,
        window_config=windows_cfg,
        field_names=("phi", "cmb"),
    )

    lmin_used, lmax_used = _infer_bin_limits()
    payload: dict[str, object] = {"cl": cl, "nside": nside, "nlb": int(args.nlb)}
    if lmin_used is not None:
        payload["lmin"] = np.array(lmin_used, dtype=int)
    if lmax_used is not None:
        payload["lmax"] = np.array(lmax_used, dtype=int)
    np.savez(Path(args.out), **payload)

    bins_summary: dict[str, object] = {}
    if isinstance(bins_meta, Mapping):
        bins_summary.update({k: v for k, v in bins_meta.items()})
    if lmin_used is not None and "lmin" not in bins_summary:
        bins_summary["lmin"] = int(lmin_used)
    if lmax_used is not None and "lmax" not in bins_summary:
        bins_summary["lmax"] = int(lmax_used)
    if "nlb" not in bins_summary:
        bins_summary["nlb"] = int(args.nlb)
    bins_summary["nbins"] = int(cl.size)

    save_json(
        {
            "order": "B_to_A",
            "nside": nside,
            "nbins": int(cl.size),
            "mask_threshold_sigma": args.threshold_sigma,
            "mask_apod_arcmin": args.apod_arcmin,
            "bins_source": "prereg" if bins_meta is not None else "cli",
            "bins": bins_summary,
            "windows": windows_cfg.to_metadata() if windows_cfg is not None else None,
        },
        Path(args.out).with_suffix(".json"),
    )
    summary_line(f"wrote {args.out} with {cl.size} bins at nside={nside}")


if __name__ == "__main__":
    main()
