#!/usr/bin/env bash
# scripts/check_artifacts.sh
# Verifies that delta.npy, cov_delta.npy, and z_scores.npy exist and contain sane data.
# Bootstraps a local venv with numpy if needed.

set -euo pipefail

ART_DIR="${1:-artifacts}"
REQ=(delta.npy cov_delta.npy z_scores.npy)

echo "Checking artifacts in: $ART_DIR"

# --- file presence ---
missing=()
for f in "${REQ[@]}"; do
  [[ -f "$ART_DIR/$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  echo "❌ Missing artifact(s): ${missing[*]}" >&2
  exit 1
fi

# --- pick python ---
if command -v python3 >/dev/null 2>&1; then PYBIN=python3
elif command -v python >/dev/null 2>&1; then PYBIN=python
else echo "❌ No python found."; exit 1; fi

# --- ensure numpy ---
set +e
$PYBIN - <<'PY' >/dev/null 2>&1
import numpy
PY
need_np=$?
set -e

if [[ $need_np -ne 0 ]]; then
  VENV=".venv_check"
  echo "Creating local venv for numpy..."
  $PYBIN -Im venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -Im ensurepip --upgrade >/dev/null 2>&1 || true
  python -m pip install -q --upgrade pip numpy
  PYRUN=python
else
  PYRUN="$PYBIN"
fi

# --- sanity check ---
$PYRUN - <<'PY'
import sys, os, numpy as np
art_dir = os.environ.get("ART_DIR", "artifacts")

def fail(msg):
    print("❌", msg, file=sys.stderr)
    sys.exit(1)

delta = np.load(os.path.join(art_dir, "delta.npy"))
cov   = np.load(os.path.join(art_dir, "cov_delta.npy"))
z     = np.load(os.path.join(art_dir, "z_scores.npy"))

if delta.ndim != 1:
    fail(f"delta has wrong shape: {delta.shape}")
if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
    fail(f"cov not square: {cov.shape}")
if z.shape != delta.shape:
    fail(f"z shape {z.shape} != delta {delta.shape}")

mean_d, std_d = float(delta.mean()), float(delta.std())
min_eig = float(np.linalg.eigvalsh(cov).min())
mean_diag = float(np.mean(np.diag(cov)))
mean_z, std_z = float(z.mean()), float(z.std())

print(f"delta mean={mean_d:.4f} std={std_d:.4f}")
print(f"cov diag mean={mean_diag:.4f} min eigen={min_eig:.4e}")
print(f"z mean={mean_z:.4f} std={std_z:.4f}")

if not (-0.3 < mean_d < 0.3):
    fail("delta mean out of range")
if not (0.01 < std_d < 0.3):
    fail("delta std out of range")
if min_eig <= 0:
    fail("covariance not positive-definite")
if not (0.7 < std_z < 1.3):
    fail("z_scores std out of range")

print("✅ Artifact sanity check passed.")
PY
