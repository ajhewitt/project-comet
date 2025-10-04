#!/usr/bin/env bash
# Quiet, minimal smoke pipeline. Only requires numpy (auto-installs into .venv_quick if missing).

set -euo pipefail

DATA_DIR="${COMET_DATA_DIR:-${1:-data}}"
NSIDE="${COMET_NSIDE:-64}"     # not used here but kept for interface stability
NSIMS="${COMET_NSIMS:-5}"
NBINS="${COMET_NBINS:-10}"

TFILE="COM_CompMap_CMB-smica_2048_R1.20.fits"
KFILE="COM_CompMap_Lensing_2048_R1.10.fits"

mkdir -p "$DATA_DIR" artifacts

# Pick a python
if command -v python3 >/dev/null 2>&1; then PYBIN=python3
elif command -v python >/dev/null 2>&1; then PYBIN=python
else echo "No python found. Install Python 3.9+." >&2; exit 1; fi

# Probe for numpy silently; no traceback noise
set +e
$PYBIN - <<'PY' >/dev/null 2>&1
import numpy
PY
probe_np=$?
set -e

# Bootstrap venv to get numpy if needed
if [[ $probe_np -ne 0 ]]; then
  VENV=".venv_quick"
  $PYBIN -Im venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -Im ensurepip --upgrade >/dev/null 2>&1 || true
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
  python -m pip install -q numpy
  PYRUN=python
else
  PYRUN="$PYBIN"
fi

# Print versions for sanity (once, no stack traces)
$PYRUN - <<'PY'
import sys, numpy
print("python", sys.version.split()[0], "numpy", numpy.__version__)
PY

# Log where we’re looking for data; do not fail on missing files (this is a smoke run)
$PYRUN - <<PY
import json, os
dd = os.path.abspath("${DATA_DIR}")
files = {
  "${KFILE}": os.path.isfile(os.path.join(dd, "${KFILE}")),
  "${TFILE}": os.path.isfile(os.path.join(dd, "${TFILE}"))
}
print(json.dumps({"data_dir": dd, "files": files}))
PY

# Deterministic fake compute that always produces artifacts
$PYRUN - <<PY
import os, numpy as np
rng = np.random.default_rng(int(os.environ.get("COMET_NSIMS","${NSIMS}")))
bins = int(os.environ.get("COMET_NBINS","${NBINS}"))

# 1) residuals
delta = rng.normal(0.0, 0.1, size=bins).astype("float32")

# 2) covariance with diag matching var(delta)
var = float(delta.var())  # sample variance
# small symmetric noise on top of diagonal var
eps = 1e-3 * var
A = rng.normal(0.0, eps, size=(bins, bins)).astype("float32")
S = (A + A.T) / 2.0
cov = (np.eye(bins, dtype="float32") * var) + S

# enforce positive-definite
w, V = np.linalg.eigh(cov)
w = np.clip(w, 1e-6 * var, None).astype("float32")
cov = (V * w) @ V.T  # V @ diag(w) @ V.T

# 3) z-scores
std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
z = (delta / std).astype("float32")

# 4) write artifacts
os.makedirs("artifacts", exist_ok=True)
np.save("artifacts/delta.npy", delta)
np.save("artifacts/cov_delta.npy", cov)
np.save("artifacts/z_scores.npy", z)
print("Artifacts written.")
PY

echo "quick pipeline done: DATA_DIR=${DATA_DIR} NSIDE=${NSIDE} NSIMS=${NSIMS} NBINS=${NBINS}"
