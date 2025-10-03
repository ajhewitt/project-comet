#!/usr/bin/env bash
set -euo pipefail
# Quick end-to-end: validate data, build templates, run both orderings, null sims, compute Δ, summarize.
DATA_DIR=${1:-data}

python scripts/fetch_planck_pr4.py --out "$DATA_DIR"
python scripts/build_context_template.py --nside 256 --out artifacts/templates.npy
python scripts/run_order_A_to_B.py --data-dir "$DATA_DIR" --quick-nside 256 --nlb 50 --out artifacts/order_A_to_B.npz
python scripts/run_order_B_to_A.py --data-dir "$DATA_DIR" --quick-nside 256 --nlb 50 --out artifacts/order_B_to_A.npz
python scripts/run_null_sims.py --data-dir "$DATA_DIR" --quick-nside 256 --nlb 50 --nsims 50 --out-cov artifacts/cov_delta.npy
python scripts/compute_commutator.py --order-a artifacts/order_A_to_B.npz --order-b artifacts/order_B_to_A.npz --cov artifacts/cov_delta.npy --out-delta artifacts/delta_ell.npy --out-summary artifacts/summary.json
python scripts/summarize_results.py --delta artifacts/delta_ell.npy --summary artifacts/summary.json --cov artifacts/cov_delta.npy --outdir docs/figures
