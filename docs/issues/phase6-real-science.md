---
title: "Phase 6: Real T×κ Science Product"
tags:
  - phase-6
  - cross-spectrum
  - science
status: done
issues:
  - #20
  - #21
  - #22
---

## Tasks
- Produce a reproducible `compute_cross_spectrum.py` CLI that combines both commutator orderings, bins the theory prediction, and reports Δ/σ statistics per bandpower.
- Expand `summarize_results.py` so science runs emit diagnostic plots for the commutator residuals and T×κ cross-spectrum, plus a machine-readable summary payload.
- Add regression coverage that verifies the cross-spectrum pipeline recovers known injected signals and that the summarization step materialises the expected artifacts.

## Outcome
- Authored `scripts/compute_cross_spectrum.py` to average the two commutator orderings, bin the theory spectrum with preregistration metadata or CLI fallbacks, and persist Δ, σ, and per-bin z-scores alongside provenance for downstream analysis.【F:scripts/compute_cross_spectrum.py†L1-L207】
- Extended `scripts/summarize_results.py` with CLI-configurable cross-spectrum ingestion, new plots (data vs. theory and per-bin z-scores), and a JSON summary capturing the key diagnostics required for phase closeout.【F:scripts/summarize_results.py†L1-L112】
- Added unit tests that inject synthetic cross-spectra, exercise the fallback binning path, and assert figure generation and summary content for the reporting CLI.【F:tests/test_compute_cross_spectrum.py†L1-L72】【F:tests/test_summarize_results.py†L1-L58】

## Closeout Confirmation
- **Issue #20** — Delivered the science-grade `compute_cross_spectrum.py` entry point that consolidates both NaMaster orderings, bins theory predictions, and emits Δ/σ diagnostics for each bandpower.
- **Issue #21** — Updated `summarize_results.py` to plot and summarise the cross-spectrum outputs alongside the existing commutator diagnostics.
- **Issue #22** — Implemented regression coverage for the cross-spectrum builder and summariser, validating both numerical results and artifact creation under controlled inputs.

## Verification
- `pytest` exercises the new regression cases, confirming the cross-spectrum CLI reproduces injected signals and the summariser emits the expected plots and JSON summary payloads.【F:tests/test_compute_cross_spectrum.py†L1-L72】【F:tests/test_summarize_results.py†L1-L58】
