---
title: "Phase 5: Simulations with Theory Cℓ"
tags:
  - phase-5
  - simulations
  - theory
status: done
issues:
  - #16
  - #17
  - #18
  - #19
---

## Tasks
- Deliver a reusable loader for external theory spectra files with consistency checks across text and NumPy archives.
- Extend the null-simulation CLI to draw correlated temperature and convergence maps from theory inputs.
- Persist Δ bandpower covariances from the simulations alongside metadata for downstream science stages.
- Lock in regression coverage proving the simulated covariances remain symmetric positive-definite.

## Outcome
- `comet.theory.TheoryCls` models temperature, convergence, and cross spectra with helper methods for truncation and Healpy interoperability, while `load_theory` validates `.npz` and plain-text formats before constructing instances.【F:src/comet/theory.py†L1-L166】
- `scripts/run_null_sims.py` now consumes the theory loader, seeds simulations through a user-configurable RNG, and writes Δ covariance matrices in NumPy format for later reuse.【F:scripts/run_null_sims.py†L1-L94】
- Simulation helpers in `comet.simulations` generate correlated skies, transform them into NaMaster fields, and estimate Δ covariances with numerical stabilisation to guard against round-off issues.【F:src/comet/simulations.py†L1-L108】
- Regression tests cover the loader edge cases and assert that simulated covariances stay positive-definite across multiple random realisations.【F:tests/test_theory_loader.py†L1-L157】【F:tests/test_simulation_covariance.py†L1-L36】

## Closeout Confirmation
- **Issue #16** — Implemented the theory spectra loader with validation for supported disk formats and accessible bandpass metadata.
- **Issue #17** — Upgraded `run_null_sims.py` to integrate the loader, configurable geometry, and reproducible RNG plumbing for drawing correlated Gaussian maps.
- **Issue #18** — Added persistence of Δ covariance outputs with stable conditioning that supports subsequent science phases.
- **Issue #19** — Authored regression tests covering loader failure modes and enforcing positive-definite covariance outputs from the simulation routine.

## Issue Closure Evidence
- **#16** — `tests/test_theory_loader.py::test_load_text_theory_matches_npz` and related cases guarantee that `.npz` and `.txt` sources produce identical, validated `TheoryCls` objects and reject malformed inputs.【F:tests/test_theory_loader.py†L17-L157】
- **#17** — Invoking `scripts/run_null_sims.py` wires the RNG seed through `numpy.random.Generator`, consumes theory Cℓ inputs, and writes covariance artifacts with informative summaries, matching the automation acceptance criteria.【F:scripts/run_null_sims.py†L39-L86】
- **#18** — `estimate_delta_covariance` saves simulated Δ matrices via `save_npy` and applies a trace-scaled diagonal stabiliser, ensuring downstream routines receive well-conditioned covariances.【F:src/comet/simulations.py†L66-L108】
- **#19** — `tests/test_simulation_covariance.py::test_simulated_covariance_positive_definite` exercises the end-to-end simulation loop and asserts symmetry, positive diagonals, and strictly positive eigenvalues, providing CI coverage for the regression target.【F:tests/test_simulation_covariance.py†L12-L36】

## Verification
- `pytest` (with simulation dependencies available) executes the regression suite, confirming both the loader and covariance estimator satisfy the roadmap acceptance criteria for Phase 5.【F:tests/test_simulation_covariance.py†L12-L36】【F:tests/test_theory_loader.py†L17-L157】
- Manual production run: `scripts/run_null_sims.py --theory artifacts/theory_planck_smica_lensing.npz --nsims 1000 --nside 2048` completed after approximately three days on a 12-thread Zen 5 workstation and produced `artifacts/cov_delta_full.npy` with shape 69×69 when invoked with the legacy CLI binning (`--disable-prereg --nlb 50`), matching the publication configuration logged by the CLI summary. Later science stages can reuse that covariance either by running both ordering scripts with the same CLI geometry or, when only a couple of trailing high-ℓ bins are extra, by trimming them via `scripts/compute_commutator.py --trim-covariance`.
