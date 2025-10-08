#!/usr/bin/env python3
"""Inspect and summarise theory C_ell inputs for simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comet.theory import load_theory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to theory spectra (.npz, .txt, .dat)")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional path to write a JSON summary of the theory file",
    )
    return parser


def summarise(path: Path) -> dict[str, object]:
    theory = load_theory(path)
    ell = theory.ell
    summary = {
        "path": str(path),
        "ell_min": int(ell.min()),
        "ell_max": int(ell.max()),
        "n_ell": int(ell.size),
        "tt_mean": float(np.mean(theory.cl_tt)),
        "kk_mean": float(np.mean(theory.cl_kk)),
        "tk_mean": float(np.mean(theory.cl_tk)),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = summarise(args.path)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
