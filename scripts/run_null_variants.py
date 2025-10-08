#!/usr/bin/env python3
"""Generate null-test residuals and Z-scores from existing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comet.nulls import evaluate_null_tests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta",
        default="artifacts/delta.npy",
        type=Path,
        help="Path to the baseline delta bandpowers (NumPy .npy)",
    )
    parser.add_argument(
        "--cov",
        default="artifacts/cov_delta.npy",
        type=Path,
        help="Path to the covariance matrix for the delta bandpowers",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("artifacts/nulls"),
        type=Path,
        help="Directory where null residuals will be written",
    )
    parser.add_argument(
        "--summary",
        default=Path("artifacts/null_tests.json"),
        type=Path,
        help="Path to the JSON summary report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.delta.exists():
        raise FileNotFoundError(f"Missing delta bandpowers at {args.delta}")
    if not args.cov.exists():
        raise FileNotFoundError(f"Missing covariance matrix at {args.cov}")

    delta = np.load(args.delta)
    cov = np.load(args.cov)

    results = evaluate_null_tests(delta, cov)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    for name, result in results.items():
        residual_path = args.out_dir / f"{name}_residual.npy"
        np.save(residual_path, result.residual.astype(np.float32))
        summary[name] = {"z": result.z, "residual": residual_path.name}

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "status": "ok",
                "summary": str(args.summary),
                "variants": {name: info["z"] for name, info in summary.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
