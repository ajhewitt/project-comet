#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import Any

import healpy as hp
import numpy as np
import yaml
from astropy.io import fits


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_map(path: str | pathlib.Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) > 1 and hasattr(hdul[1], "data"):
            data = hdul[1].data
            arr = np.asarray(data.field(0), dtype=np.float64)
        else:
            arr = np.asarray(hdul[0].data, dtype=np.float64)
    return arr


def apodize_mask(mask: np.ndarray, aporad_deg: float = 1.0) -> np.ndarray:
    # Smooth then clip, simple soft edge
    sigma_rad = np.deg2rad(aporad_deg) / np.sqrt(8.0 * np.log(2.0))
    ap = hp.smoothing(mask.astype(float), sigma=sigma_rad, verbose=False)
    return np.clip(ap, 0.0, 1.0)


def norm_vec(v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        n = np.linalg.norm(v)
    else:
        n = float(np.sqrt(np.sum((v * w) * v)))
    return v if n == 0 else v / n


def build(args: argparse.Namespace) -> int:
    paths = load_yaml(args.paths)

    mask_path = paths.get("common_tmask") or paths.get("mask") or paths.get("T_mask")
    if mask_path is None:
        raise ValueError("No mask path found in config (common_tmask / mask / T_mask).")

    exp_path = paths.get("exposure") or paths.get("exposure_map")
    scan_path = paths.get("scan") or paths.get("scan_map")
    if exp_path is None or scan_path is None:
        raise ValueError("Need 'exposure' and 'scan' entries in paths YAML.")

    mask = read_map(mask_path)
    exp_map = read_map(exp_path)
    scan_map = read_map(scan_path)

    if not (mask.size == exp_map.size == scan_map.size):
        raise ValueError("Input maps must have the same size.")

    mask_apo = apodize_mask(mask, aporad_deg=1.0)

    # Basis 1: demeaned exposure pattern
    valid = mask_apo > 0.5
    c1 = exp_map - float(np.mean(exp_map[valid]))
    c1 = norm_vec(c1, w=mask_apo)

    # Basis 2: demeaned scan pattern, orthogonalize to c1
    c2 = scan_map - float(np.mean(scan_map[valid]))
    w = mask_apo

    def dot(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * b * w))

    c2 = c2 - dot(c2, c1) * c1
    c2 = norm_vec(c2, w=mask_apo)

    out = {
        "mask": mask_apo.astype(np.float32),
        "c1": c1.astype(np.float32),
        "c2": c2.astype(np.float32),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[context] wrote {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build context template vectors")
    p.add_argument("--paths", required=True, help="YAML with data file paths")
    p.add_argument("--prereg", required=False, help="YAML prereg config (unused here)")
    p.add_argument("--out", required=True, help="Output .npz path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return build(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
