#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import Any

import healpy as hp
import numpy as np
import yaml
from astropy.io import fits


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_map(path: str) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) > 1 and hasattr(hdul[1], "data"):
            data = hdul[1].data
            arr = np.asarray(data.field(0), dtype=np.float64)
        else:
            arr = np.asarray(hdul[0].data, dtype=np.float64)
    return arr


def map_lowell(m: np.ndarray, lmax_keep: int, nside: int) -> np.ndarray:
    alm = hp.map2alm(m)
    lmax = hp.Alm.getlmax(alm.size)
    ell = np.arange(lmax + 1)
    keep = (ell <= lmax_keep).astype(float)
    hp.almxfl(alm, keep, inplace=True)
    return hp.alm2map(alm, nside, verbose=False)


def run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    t_path = paths.get("temperature_map")
    if t_path is None:
        raise ValueError("paths.yaml missing 'temperature_map'")

    T = read_map(t_path)
    nside = hp.get_nside(T)

    # Same ISW projection step (for order parity).
    # Lensing reconstruction is assumed fixed (public phi).
    lkeep = int(cfg["ells"]["T"][1])
    T_isw = map_lowell(T, lmax_keep=lkeep, nside=nside)

    out = {"estimate": T_isw.astype(np.float32)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[B->A] wrote {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline order B->A")
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--paths", required=True, help="YAML with data paths")
    p.add_argument("--out", required=True, help="Output NPZ path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
