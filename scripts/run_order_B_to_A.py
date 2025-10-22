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
    bins_ell_groups: list[np.ndarray] = []
    ell_effective: np.ndarray | None = None
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

    if hasattr(bins, "get_effective_ells"):
        try:
            ell_effective = np.asarray(bins.get_effective_ells(), dtype=float)
        except Exception:
            ell_effective = None

    if hasattr(bins, "get_ell_list"):
        try:
            bins_ell_groups = [np.asarray(group, dtype=int) for group in bins.get_ell_list()]
        except Exception:
            bins_ell_groups = []

    def _infer_bin_limits() -> tuple[int | None, int | None]:
        lmin_val = getattr(bins, "lmin", None)
        lmax_val = getattr(bins, "lmax", None)
        for group in bins_ell_groups:
            arr = np.asarray(group, dtype=int)
            if arr.size:
                lmin_val = int(arr[0])
                break
        for group in reversed(bins_ell_groups):
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
    if lmin_used is None:
        if args.lmin is not None:
            lmin_used = int(args.lmin)
        elif isinstance(bins_meta, Mapping) and "lmin" in bins_meta:
            try:
                lmin_used = int(bins_meta["lmin"])
            except Exception:
                lmin_used = None
        if lmin_used is None:
            # Mirror the implicit NaMaster behaviour of starting at ℓ=0 so the
            # cross-spectrum step can fall back to linear binning even when
            # preregistration metadata is absent.
            lmin_used = 0
    bin_left_edges: np.ndarray | None = None
    bin_right_edges: np.ndarray | None = None
    if bins_ell_groups and len(bins_ell_groups) == cl.size:
        bin_left_edges = np.full(cl.size, -1, dtype=int)
        bin_right_edges = np.full(cl.size, -1, dtype=int)
        for idx, group in enumerate(bins_ell_groups):
            if group.size:
                bin_left_edges[idx] = int(group[0])
                bin_right_edges[idx] = int(group[-1]) + 1

    nlb_value = int(args.nlb)
    if isinstance(bins_meta, Mapping) and "nlb" in bins_meta:
        try:
            nlb_value = int(bins_meta["nlb"])
        except Exception:
            nlb_value = int(args.nlb)

    payload: dict[str, object] = {"cl": cl, "nside": nside, "nlb": np.array(nlb_value, dtype=int)}
    if lmin_used is not None:
        payload["lmin"] = np.array(lmin_used, dtype=int)
    if lmax_used is not None:
        payload["lmax"] = np.array(lmax_used, dtype=int)
    if ell_effective is not None and ell_effective.size == cl.size:
        payload["ell_effective"] = ell_effective.astype(float)
    if bin_left_edges is not None and bin_right_edges is not None:
        payload["ell_left_edges"] = bin_left_edges
        payload["ell_right_edges"] = bin_right_edges
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
    if (
        ell_effective is not None
        and ell_effective.size == cl.size
        and "ell_effective" not in bins_summary
    ):
        bins_summary["ell_effective"] = ell_effective.astype(float).tolist()
    if (
        bin_left_edges is not None
        and bin_right_edges is not None
        and "ell_left_edges" not in bins_summary
        and "ell_right_edges" not in bins_summary
    ):
        bins_summary["ell_left_edges"] = bin_left_edges.astype(int).tolist()
        bins_summary["ell_right_edges"] = bin_right_edges.astype(int).tolist()

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
