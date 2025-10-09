#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from commutator_common import summary_line


def _write_text_summary(path: Path, lines: list[str]) -> None:
    payload = {"summary": lines}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot Δ, null diagnostics, and cross-spectrum comparisons"
    )
    ap.add_argument("--delta", default="artifacts/delta_ell.npy")
    ap.add_argument("--summary", default="artifacts/summary.json")
    ap.add_argument("--cov", default="artifacts/cov_delta.npy")
    ap.add_argument("--cross", default="artifacts/cross_tk.npz")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    delta = np.load(args.delta)

    text_lines = []

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
        plt.figure()
        plt.hist(delta / (np.sqrt(np.maximum(np.diag(C), 1e-30))), bins=20)
        plt.xlabel("standardized Δ per bin")
        plt.ylabel("count")
        plt.title("Null histogram (per-bin)")
        p2 = out / "null_hist.png"
        plt.savefig(p2.as_posix(), dpi=150, bbox_inches="tight")
        plt.close()

        max_z = float(np.max(np.abs(delta / (np.sqrt(np.maximum(np.diag(C), 1e-30))))))
        text_lines.append(f"max|Δ|/σ = {max_z:.3f}")

    cross_path = Path(args.cross)
    if cross_path.exists():
        data = np.load(cross_path)
        ell = np.asarray(data["ell"], dtype=float)
        cl_data = np.asarray(data["cl_data"], dtype=float)
        cl_theory = np.asarray(data["cl_theory"], dtype=float)
        sigma = np.asarray(data.get("sigma", np.zeros_like(cl_data)), dtype=float)
        positive = sigma > 0
        delta_cross = np.asarray(data.get("delta", cl_data - cl_theory), dtype=float)
        z = np.asarray(data.get("z", np.zeros_like(delta_cross)), dtype=float)

        plt.figure()
        if sigma.size and np.any(positive):
            plt.errorbar(ell, cl_data, yerr=sigma, fmt="o", label="data")
        else:
            plt.plot(ell, cl_data, "o", label="data")
        plt.plot(ell, cl_theory, label="theory")
        plt.xlabel(r"$\ell_\mathrm{eff}$")
        plt.ylabel(r"$C_\ell^{T\kappa}$")
        plt.title("T×κ cross-spectrum")
        plt.legend()
        p3 = out / "cross_spectrum.png"
        plt.savefig(p3.as_posix(), dpi=150, bbox_inches="tight")
        plt.close()

        if z.size and np.any(positive):
            text_lines.append(
                "mean|z| (cross) = "
                f"{float(np.mean(np.abs(z[positive]))):.3f}; max|z| = {float(np.max(np.abs(z[positive]))):.3f}"
            )
            plt.figure()
            plt.bar(np.arange(z.size), z)
            plt.xlabel("bin")
            plt.ylabel("z-score")
            plt.title("Cross-spectrum per-bin significance")
            p4 = out / "cross_z_scores.png"
            plt.savefig(p4.as_posix(), dpi=150, bbox_inches="tight")
            plt.close()
        elif z.size:
            text_lines.append("z-scores unavailable; covariance diagonal non-positive")

        text_lines.append(f"⟨Δ_cross⟩ = {float(delta_cross.mean()):.3e}")

        if z.size and np.any(positive):
            payload = {
                "nbins": int(delta_cross.size),
                "valid_bins": int(np.count_nonzero(positive)),
                "mean_abs_z": float(np.mean(np.abs(z[positive]))),
                "max_abs_z": float(np.max(np.abs(z[positive]))),
            }
        else:
            payload = {
                "nbins": int(delta_cross.size),
                "valid_bins": 0,
                "mean_abs_z": 0.0,
                "max_abs_z": 0.0,
            }
        summary_line("cross-spectrum " + json.dumps(payload, sort_keys=True))

    if text_lines:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_summary(summary_path, text_lines)

    summary_line(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
