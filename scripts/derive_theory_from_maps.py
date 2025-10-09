#!/usr/bin/env python3
"""Generate a temperature/lensing theory table directly from Planck maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

DEFAULT_CMB = Path("data/COM_CompMap_CMB-smica_2048_R1.20.fits")
DEFAULT_KAPPA = Path("data/COM_CompMap_Lensing_2048_R1.10.fits")
DEFAULT_TXT = Path("data/theory/tk_planck2018.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmb-map",
        type=Path,
        default=DEFAULT_CMB,
        help=f"Path to the Planck temperature map (default: {DEFAULT_CMB})",
    )
    parser.add_argument(
        "--kappa-map",
        type=Path,
        default=DEFAULT_KAPPA,
        help=f"Path to the Planck lensing convergence map (default: {DEFAULT_KAPPA})",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=2048,
        help="Maximum multipole to include when computing spectra (default: 2048)",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=DEFAULT_TXT,
        help=f"Destination for the four-column ASCII table (default: {DEFAULT_TXT})",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="Optional destination for a NumPy archive containing ell, cl_tt, cl_kk, and cl_tk",
    )
    return parser


def compute_spectra(
    cmb_map: Path, kappa_map: Path, lmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cmb = hp.read_map(cmb_map, field=0)
    kappa = hp.read_map(kappa_map, field=0)

    cl_tt = hp.anafast(cmb, lmax=lmax)
    cl_kk = hp.anafast(kappa, lmax=lmax)
    cl_tk = hp.anafast(cmb, kappa, lmax=lmax)
    ell = np.arange(cl_tt.size)
    return ell, cl_tt, cl_kk, cl_tk


def write_text(
    path: Path, ell: np.ndarray, cl_tt: np.ndarray, cl_kk: np.ndarray, cl_tk: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack([ell, cl_tt, cl_kk, cl_tk])
    header = "ell ClTT Clkk ClTk"
    np.savetxt(path, table, header=header)


def write_npz(
    path: Path, ell: np.ndarray, cl_tt: np.ndarray, cl_kk: np.ndarray, cl_tk: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ell=ell, cl_tt=cl_tt, cl_kk=cl_kk, cl_tk=cl_tk)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ell, cl_tt, cl_kk, cl_tk = compute_spectra(args.cmb_map, args.kappa_map, args.lmax)
    write_text(args.output_text, ell, cl_tt, cl_kk, cl_tk)

    if args.output_npz is not None:
        write_npz(args.output_npz, ell, cl_tt, cl_kk, cl_tk)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
