# Phase 5 — Simulations with Theory Cℓ

## Summary
- Added a reusable loader for theory spectra with validation of `.npz` and text formats.
- Extended the null-simulation pipeline to draw correlated maps from theory Cℓ and build Δ covariance matrices.
- Introduced regression tests that exercise the loader and ensure covariance matrices remain positive-definite when simulations run.

## Follow-ups
- Integrate high-resolution theory inputs once full-production files are available.
- Run the long (nsims ≳ 2000) campaign offline and archive the resulting covariance.
