# Project Comet: Roadmap

This document outlines the phased plan for taking **Project Comet** from its current state (synthetic null pipeline with quick plots) to a fully validated analysis of real Planck CMB (SMICA) and lensing (κ) maps.

---

## Phase 0 — Freeze What Works (Baseline)

**Goal:** Lock the current “quick” pipeline and make CI catch regressions.

- Keep `scripts/` for science code, `bin/` for shortcuts.
- CI runs: ruff format + lint, unit tests, and `python -m comet.cli demo`.
- Add a *smoke* job in CI: run quick pipeline at **nside=64** with tiny bins and **nsims=5**. Never use real 2048 maps in CI.

**Tests**
- `tests/test_env_smoke.py`: import numpy, healpy, pymaster.
- `tests/test_quick_specs.py`: run quick pipeline on synthetic inputs; assert file creation.
- `tests/test_units.py`: assert expected units.

**Acceptance criteria**
- CI green.
- Quick pipeline produces Δ, Cov, Z on local 256.

---

## Phase 1 — Data Ingestion & Units Sanity

**Goal:** Never again guess what units/conventions we’re using.

- **Units**:
  - SMICA T in **K_CMB** (thermodynamic). Convert if necessary.
  - Lensing map κ is dimensionless.
- **Geometry**:
  - Confirm **RING** ordering, NSIDE=2048.
  - Record coordinate system.
- **Beam/pixel window**:
  - Record effective beam FWHM.
  - Store Planck pixel window reference.

**Code**
- `io_maps.read_fits_map_with_meta(path) -> MapData`.
- Save metadata JSON per map.
- `scripts/check_data.py`: validates headers.

**Tests**
- Mock FITS headers, assert parsing.
- Fail if units unexpected.

**Acceptance**
- Metadata JSON correct for both datasets.

---

## Phase 2 — Masking, Apodization, and Binning

**Goal:** Realistic mask and controlled binning.

- Mask: threshold + apodization (e.g., 30′).
- Binning: configurable via prereg YAML.

**Code**
- Extend `commutator_common.build_mask`.
- `scripts/make_mask.py` writes artifacts/mask.

**Tests**
- f_sky within [0.5,0.9].
- Bin count matches config.

---

## Phase 3 — Spectra Deconvolution & Windows

**Goal:** Account for beam and pixel effects.

- Apply pixel window.
- Optionally deconvolve beam.

**Code**
- `namaster_utils.py`: helpers.
- Config toggle.

**Tests**
- Beam on/off test changes high-ℓ tail.

---

## Phase 4 — Null Tests

**Goal:** Add orthogonal nulls.

- Rotate map by 90°.
- Hemisphere jackknife.
- Curl/null field (later).

**Code**
- `scripts/run_null_variants.py`.

**Tests**
- Synthetic skies → nulls return Z~0.

---

## Phase 5 — Simulations with Theory Cℓ

**Goal:** Null covariance reflects cosmology.

- Generate Gaussian alms with TT, κκ, Tκ.
- Pass through same mask/beam/binning.
- nsims ≈2000+ offline.

**Code**
- `scripts/theory.py` loader.
- Extend `run_null_sims.py`.

**Tests**
- Positive-definite covariance.

---

## Phase 6 — Real T×κ Science Product

**Goal:** Deliver interpretable results.

- Compute cross-spectra Cℓ^{Tκ}.
- Compare to theory.
- Δ per bin with σ from covariance.

**Code**
- `compute_cross_spectrum.py`.
- Extend `summarize_results.py`.

**Tests**
- Injected signal recovery test.

---

## Phase 7 — Reproducibility & Sanity Locks

**Goal:** No accidental inches.

- **Unit lock:** assert K_CMB, dimensionless κ.
- **Geometry lock:** assert RING, coords.
- **Hash lock:** record SHA256 of inputs.
- **Version lock:** record package versions and git commit.

**CI gates**
- Lint + tests.
- Quick smoke job.
- Refuse unexpected units/orderings.
- Block committing data/artifacts/figures.

---

## Example Developer Flow

```bash
# Quick run (nside=256)
micromamba run -n comet bash scripts/pipeline_quick.sh data

# Validate headers
micromamba run -n comet python scripts/check_data.py --data-dir data

# Make mask
micromamba run -n comet python scripts/make_mask.py --data-dir data --nside 2048 --apod-arcmin 30

# Bandpowers
micromamba run -n comet python scripts/run_order_A_to_B.py --data-dir data --quick-nside 1024 --nlb 50

# Null sims
micromamba run -n comet python scripts/run_null_sims.py --data-dir data --nsims 1000 --theory theory/tk_kk_tt.npz

# Commute
micromamba run -n comet python scripts/compute_commutator.py --order-a artifacts/order_A_to_B.npz --order-b artifacts/order_B_to_A.npz --cov artifacts/cov_delta.npy

# Summarize
micromamba run -n comet python scripts/summarize_results.py
```

---

## Sanity Checks Per Run

- Units line: CMB units K_CMB, κ dimensionless.
- Geometry line: NSIDE, ORDERING, COORDSYS.
- Mask line: f_sky.
- Bins line.
- Windows line.
- Results line: Δ, Z.
- Hashes: data SHA256, config.

---

This roadmap is the canonical reference, resume at the last completed phase and confirm CI passes smoke tests before moving forward.
