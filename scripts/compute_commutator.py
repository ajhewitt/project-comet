# scripts/compute_commutator.py
import argparse, json, numpy as np
import healpy as hp
from src.common import load_yaml

def load_npz(p):
    d = np.load(p)
    return d["ells"], d["cl"]

def project_context(context_npz, mask_fits):
    ctx = np.load(context_npz)
    c = ctx["c"]
    # sign convention: positive if the masked sum is positive
    m = (hp.read_map(mask_fits, verbose=False) > 0.5).astype(float)
    sgn = np.sign((c*m).sum())
    return int(sgn) if sgn != 0 else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="npz from A->B")
    ap.add_argument("--B", required=True, help="npz from B->A")
    ap.add_argument("--context", required=True, help="context_template.npz")
    ap.add_argument("--prereg", required=True, help="config/prereg.yaml")
    ap.add_argument("--out", required=True, help="output JSON")
    args = ap.parse_args()

    ellsA, clA = load_npz(args.A)
    ellsB, clB = load_npz(args.B)
    if not np.all(ellsA == ellsB):
        raise RuntimeError("Bin mismatch between A and B outputs")

    cfg = load_yaml(args.prereg)
    # Simple analytic weights: inverse variance proxy ~ 1/(|clA|+|clB|+eps)
    eps = 1e-18
    wL = 1.0 / (np.abs(clA) + np.abs(clB) + eps)
    wL /= wL.sum()

    delta = np.sum(wL * (clA - clB))

    # Signed projection using context
    sign_pred = cfg.get("context_sign_prediction", +1)
    sign_obs  = project_context(args.context, cfg["masks"]["T_mask"])
    S_gamma = float(sign_pred * np.sign(delta) * np.abs(delta))

    out = {
        "ells": ellsA.tolist(),
        "clA_minus_clB": (clA - clB).tolist(),
        "Delta_comm": float(delta),
        "S_gamma_proxy": S_gamma,
        "weights_sumcheck": float(wL.sum())
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[commutator] wrote {args.out} :: Δ_comm={delta:.3e}")

if __name__ == "__main__":
    main()
