#!/usr/bin/env python

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comet.theory import load_theory
from commutator_common import load_bins_from_prereg, summary_line


@dataclass(slots=True)
class _BinInfo:
    ell_effective: np.ndarray
    ell_lists: list[np.ndarray]


def _load_bin_info(
    prereg: Path,
    *,
    nside: int,
    nbins: int,
    fallback_lmin: int | None,
    fallback_nlb: int | None,
) -> _BinInfo:
    try:
        bins, bins_meta = load_bins_from_prereg(prereg, nside=nside)
    except FileNotFoundError:
        bins = None
    except Exception as exc:  # pragma: no cover - defensive fallback for CLI use
        summary_line(f"failed to load prereg bins: {exc}; falling back to CLI parameters")
        bins = None

    if bins is not None:
        ell_eff = np.asarray(bins.get_effective_ells(), dtype=float)
        ell_lists: list[np.ndarray] = []
        if hasattr(bins, "get_ell_list"):
            try:
                for group in bins.get_ell_list():
                    ell_lists.append(np.asarray(group, dtype=int))
            except Exception:  # pragma: no cover - guard against API mismatches
                ell_lists = []
        if ell_eff.size == nbins:
            return _BinInfo(ell_effective=ell_eff, ell_lists=ell_lists)

        summary_line(
            "prereg bin definition does not match input spectra; "
            "falling back to CLI-provided binning"
        )

    if fallback_nlb is None:
        raise ValueError("Prereg bins unavailable; please provide --nlb for fallback binning")
    if fallback_lmin is None:
        raise ValueError("Prereg bins unavailable; please provide --lmin for fallback binning")

    lmin = int(fallback_lmin)
    step = int(fallback_nlb)
    if step <= 0:
        raise ValueError("nlb must be positive")
    edges = lmin + step * np.arange(nbins + 1)
    ell_lists = [np.arange(int(edges[i]), int(edges[i + 1])) for i in range(nbins)]
    ell_eff = np.asarray(
        [
            0.5 * (float(ells[0]) + float(ells[-1]))
            if ells.size
            else float(lmin + step * i + 0.5 * step)
            for i, ells in enumerate(ell_lists)
        ],
        dtype=float,
    )
    return _BinInfo(ell_effective=ell_eff, ell_lists=ell_lists)


def _expand_spectrum(ell: Sequence[int], values: Sequence[float], lmax: int) -> np.ndarray:
    full = np.zeros(int(lmax) + 1, dtype=float)
    ell_arr = np.asarray(ell, dtype=int)
    mask = ell_arr <= int(lmax)
    full[ell_arr[mask]] = np.asarray(values, dtype=float)[mask]
    return full


def _bin_theory(
    info: _BinInfo,
    theory_ell: np.ndarray,
    theory_tk: np.ndarray,
) -> np.ndarray:
    if info.ell_lists:
        ell_max = max((int(group.max()) for group in info.ell_lists if group.size), default=0)
    else:
        ell_max = int(np.ceil(float(info.ell_effective.max()))) if info.ell_effective.size else 0
    full = _expand_spectrum(theory_ell, theory_tk, ell_max)

    # Prefer NaMaster-provided binning if available.
    if info.ell_lists:
        binned = []
        for group in info.ell_lists:
            if group.size == 0:
                binned.append(0.0)
                continue
            samples = full[group]
            binned.append(float(samples.mean()))
        return np.asarray(binned, dtype=float)

    if info.ell_effective.size:
        return np.interp(
            info.ell_effective,
            theory_ell,
            theory_tk,
            left=0.0,
            right=0.0,
        )
    return np.zeros(0, dtype=float)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the T×κ cross-spectrum science product")
    ap.add_argument("--order-a", default="artifacts/order_A_to_B.npz")
    ap.add_argument("--order-b", default="artifacts/order_B_to_A.npz")
    ap.add_argument("--theory", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, default=Path("config/prereg.yaml"))
    ap.add_argument("--cov", type=Path, default=Path("artifacts/cov_delta.npy"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/cross_tk.npz"))
    ap.add_argument("--summary", type=Path, default=Path("artifacts/cross_summary.json"))
    ap.add_argument(
        "--lmin", type=int, default=None, help="Fallback ℓ_min when prereg bins are missing"
    )
    ap.add_argument(
        "--nlb", type=int, default=None, help="Fallback bin width when prereg bins are missing"
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    order_a = np.load(args.order_a)
    order_b = np.load(args.order_b)
    cl_a = np.asarray(order_a["cl"], dtype=float)
    cl_b = np.asarray(order_b["cl"], dtype=float)

    if cl_a.shape != cl_b.shape:
        raise ValueError(f"Bandpower shapes differ: {cl_a.shape} vs {cl_b.shape}")

    nbins = cl_a.size
    nside = int(order_a.get("nside", order_b.get("nside", 0)))

    def _extract_scalar(value: np.ndarray | None) -> int | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.size == 0:
            return None
        try:
            return int(arr.ravel()[0])
        except Exception:
            return None

    fallback_nlb = args.nlb
    if fallback_nlb is None:
        fallback_nlb = _extract_scalar(order_a.get("nlb")) or _extract_scalar(order_b.get("nlb"))

    info = _load_bin_info(
        args.prereg,
        nside=nside if nside > 0 else 256,
        nbins=nbins,
        fallback_lmin=args.lmin,
        fallback_nlb=fallback_nlb,
    )

    ell_eff = info.ell_effective
    cl_data = 0.5 * (cl_a + cl_b)

    theory = load_theory(args.theory)
    theory_bins = _bin_theory(info, theory.ell.astype(int), theory.cl_tk)
    if theory_bins.size != nbins:
        if theory_bins.size == 0:
            raise ValueError("Theory binning produced an empty spectrum")
        theory_bins = np.interp(
            ell_eff,
            theory.ell,
            theory.cl_tk,
            left=0.0,
            right=0.0,
        )

    delta = cl_data - theory_bins

    cov_path = Path(args.cov)
    sigma = None
    if cov_path.exists():
        cov = np.load(cov_path)
        if cov.shape[0] != cov.shape[1] or cov.shape[0] != nbins:
            raise ValueError(
                f"Covariance shape {cov.shape} incompatible with bandpowers of length {nbins}"
            )
        diag = np.asarray(np.diag(cov), dtype=float)
        sigma = np.sqrt(np.clip(diag, a_min=0.0, a_max=None))
    else:
        sigma = np.zeros(nbins, dtype=float)

    z = np.zeros_like(delta)
    if sigma is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.divide(delta, sigma, out=np.zeros_like(delta), where=sigma > 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        ell=ell_eff,
        cl_data=cl_data,
        cl_theory=theory_bins,
        delta=delta,
        sigma=sigma,
        z=z,
        order_a=str(args.order_a),
        order_b=str(args.order_b),
        theory=args.theory.as_posix(),
        cov=args.cov.as_posix() if cov_path.exists() else None,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "nbins": int(nbins),
        "mean_abs_z": float(np.mean(np.abs(z))) if z.size else 0.0,
        "max_abs_z": float(np.max(np.abs(z))) if z.size else 0.0,
        "inputs": {
            "order_a": str(args.order_a),
            "order_b": str(args.order_b),
            "theory": args.theory.as_posix(),
            "cov": args.cov.as_posix() if cov_path.exists() else None,
        },
        "outputs": {
            "cross": args.out.as_posix(),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))

    summary_line(
        "cross-spectrum "
        f"bins={nbins} mean|z|={summary['mean_abs_z']:.3f} max|z|={summary['max_abs_z']:.3f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
