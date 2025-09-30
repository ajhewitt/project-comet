# scripts/build_context_template.py
import argparse
import numpy as np
import healpy as hp
from src.common import load_yaml, apodize
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="config/paths.yaml")
    ap.add_argument("--prereg", required=True, help="config/prereg.yaml")
    ap.add_argument("--out", required=True, help="output npz file")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    prereg = load_yaml(args.prereg)

    # Load required maps
    exp_map = hp.read_map(paths["exposure_map"], verbose=False)
    scan_map = hp.read_map(paths["scan_map"], verbose=False)
    mask = hp.read_map(paths["T_mask"], verbose=False)
    mask_apo = apodize(mask, aporad_deg=1.0)

    nside = hp.get_nside(exp_map)

    # Very simple basis: demean, normalize, mask-weighted
    def norm_vec(v):
        v = v - np.mean(v[mask > 0.5])
        v = v * mask_apo
        s = np.sqrt(np.sum(v**2))
        return v / s if s > 0 else v

    c1 = norm_vec(exp_map.copy())
    c2 = norm_vec(scan_map.copy())

    # Orthogonalize c2 to c1 (Gram-Schmidt under mask)
    w = mask_apo
    dot = lambda a,b: np.sum(a*b*w)
    c2 = c2 - dot(c2,c1)*c1
    c2 = norm_vec(c2)

    # Leading context mode = c1 by default
    np.savez(args.out, c=c1, basis=np.vstack([c1, c2]))
    print(f"[context] wrote {args.out} with modes shape {(2, c1.size)}")

if __name__ == "__main__":
    main()
