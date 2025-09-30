# scripts/run_order_A_to_B.py
import argparse, json, numpy as np, healpy as hp, pymaster as nmt
from src.common import load_yaml, load_maps, apodize, build_bins, map_lowell, compute_cross_cls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.prereg)
    T, T_mask, phi, phi_mask = load_maps(paths)

    nside = hp.get_nside(T)
    # ISW proxy: low-ell filtered T (ℓ<=64 by default)
    lkeep = int(cfg["ells"]["T"][1])
    T_isw = map_lowell(T, lmax_keep=lkeep, nside=nside)

    # Build fields with apodized masks
    T_mask_apo = apodize(T_mask, aporad_deg=1.0)
    phi_mask_apo = apodize(phi_mask, aporad_deg=1.0)
    f_phi = nmt.NmtField(phi_mask_apo, [phi])
    f_Tisw = nmt.NmtField(T_mask_apo, [T_isw])

    bins = build_bins(nside, cfg["ells"]["bins"])
    ells, cl = compute_cross_cls(f_phi, f_Tisw, bins)

    np.savez(args.out, ells=ells, cl=cl)
    print(f"[A->B] wrote {args.out}")

if __name__ == "__main__":
    main()
