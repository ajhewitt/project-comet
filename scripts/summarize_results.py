#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from commutator_common import summary_line


def main():
    ap = argparse.ArgumentParser(description="Plot Δ and (optional) null histogram")
    ap.add_argument("--delta", default="artifacts/delta_ell.npy")
    ap.add_argument("--summary", default="artifacts/summary.json")
    ap.add_argument("--cov", default="artifacts/cov_delta.npy")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    delta = np.load(args.delta)

    plt.figure()
    plt.plot(delta, marker="o")
    plt.xlabel("bin")
    plt.ylabel(r"$\Delta C_\ell$")
    plt.title("Commutator Δ bandpowers")
    p1 = out / "delta_ell.png"
    plt.savefig(p1.as_posix(), dpi=150, bbox_inches="tight")
    plt.close()

    if Path(args.cov).exists():
        C = np.load(args.cov)
        # Project Δ onto leading eigenvectors for a quick sanity
        w, _ = np.linalg.eigh(C + 1e-12 * np.eye(C.shape[0]))
        plt.figure()
        plt.hist(delta / (np.sqrt(np.maximum(np.diag(C), 1e-30))), bins=20)
        plt.xlabel("standardized Δ per bin")
        plt.ylabel("count")
        plt.title("Null histogram (per-bin)")
        p2 = out / "null_hist.png"
        plt.savefig(p2.as_posix(), dpi=150, bbox_inches="tight")
        plt.close()

    summary_line(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
