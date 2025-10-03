#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stable_z(delta: np.ndarray, C: np.ndarray) -> float | None:
    """
    Compute Z = sqrt(delta^T C^{-1} delta) robustly.

    Strategy:
      1) Add jitter scaled to the trace(C)/n and try Cholesky solves.
      2) If that fails up to a generous ceiling, fall back to pinv with rcond.
    """
    n = C.shape[0]
    if n == 0 or delta.size != n:
        return None

    tr = float(np.trace(C))
    base = 1e-10 * (tr / max(n, 1) if tr > 0 else 1.0)
    jitter_levels = [0.0, base, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    for eps in jitter_levels:
        try:
            Ci = C + eps * np.eye(n)
            L = np.linalg.cholesky(Ci)
            # Solve L y = delta
            y = np.linalg.solve(L, delta)
            # Solve L^T x = y
            x = np.linalg.solve(L.T, y)
            z2 = float(delta @ x)
            return float(np.sqrt(max(z2, 0.0)))
        except np.linalg.LinAlgError:
            continue

    # Fallback: Moore-Penrose pseudoinverse with a conservative rcond
    Ci = np.linalg.pinv(C, rcond=1e-6)
    z2 = float(delta @ (Ci @ delta))
    return float(np.sqrt(max(z2, 0.0)))


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), arr)


def main():
    ap = argparse.ArgumentParser(description="Compute commutator Δ and Z")
    ap.add_argument("--order-a", default="artifacts/order_A_to_B.npz")
    ap.add_argument("--order-b", default="artifacts/order_B_to_A.npz")
    ap.add_argument("--cov", default="artifacts/cov_delta.npy")
    ap.add_argument("--out-delta", default="artifacts/delta_ell.npy")
    ap.add_argument("--out-summary", default="artifacts/summary.json")
    args = ap.parse_args()

    A = np.load(args.order_a)["cl"]
    B = np.load(args.order_b)["cl"]
    if A.shape != B.shape:
        raise ValueError(f"Bandpower shapes differ: {A.shape} vs {B.shape}")

    delta = A - B
    _save_npy(delta, Path(args.out_delta))

    z = None
    cov_path = Path(args.cov)
    if cov_path.exists():
        C = np.load(cov_path)
        if C.shape[0] != C.shape[1] or C.shape[0] != delta.size:
            raise ValueError(f"Covariance shape {C.shape} incompatible with delta {delta.shape}")
        z = _stable_z(delta, C)

    summary = {
        "nbins": int(delta.size),
        "z": None if z is None else float(z),
        "inputs": {
            "order_a": args.order_a,
            "order_b": args.order_b,
            "cov": args.cov if cov_path.exists() else None,
        },
        "outputs": {
            "delta": args.out_delta,
        },
    }
    _save_json(summary, Path(args.out_summary))
    print(json.dumps({"msg": f"delta bins={delta.size} z={summary['z']}"}, sort_keys=True))


if __name__ == "__main__":
    main()
