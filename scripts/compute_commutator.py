#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def compute_stat(a: np.ndarray, b: np.ndarray) -> float:
    # Placeholder statistic: signed normalized difference
    num = float(np.sum(a - b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12)
    return num / den


def run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    # If you later need observed context, load it here and actually use it.

    A = load_npz(args.A)
    B = load_npz(args.B)

    a_map = A.get("estimate") or A.get("map") or A[list(A.keys())[0]]
    b_map = B.get("estimate") or B.get("map") or B[list(B.keys())[0]]

    if a_map.shape != b_map.shape:
        raise ValueError("A and B maps must have same shape")

    sign_pred = int(cfg.get("context_sign_prediction", +1))

    delta = a_map - b_map
    s_gamma = float(sign_pred * np.sign(delta.mean()) * abs(delta.mean()))
    commutator = compute_stat(a_map, b_map)

    result = {
        "S_gamma": s_gamma,
        "commutator": commutator,
        "shape": list(a_map.shape),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[commutator] wrote {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute commutator summary")
    p.add_argument("--A", required=True, help="NPZ from order A")
    p.add_argument("--B", required=True, help="NPZ from order B")
    p.add_argument("--context", required=True, help="Context NPZ (currently unused)")
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--out", required=True, help="Output JSON path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
