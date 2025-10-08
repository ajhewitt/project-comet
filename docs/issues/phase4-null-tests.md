---
title: "Phase 4: Null tests"
tags:
  - phase-4
  - nulls
status: done
issues:
  - #13
  - #14
  - #15
---

## Tasks
- Introduce reusable null-variant helpers that cover 90° rotations, hemisphere jackknives, and curl-like residuals.
- Provide a dedicated CLI (`scripts/run_null_variants.py`) that consumes quick-pipeline artifacts and writes null-test summaries.
- Lock in regression coverage asserting that synthetic skies produce Z-scores consistent with zero across all null variants.

## Outcome
- `comet.nulls` now offers deterministic helpers for rotation, hemisphere, and curl nulls plus an evaluation routine returning structured `NullTestResult` payloads.
- The `run_null_variants.py` script loads existing `delta` and covariance artifacts, emits residual arrays per variant, and records a machine-readable JSON summary of null Z-scores.
- Unit tests validate each helper and confirm that perfectly symmetric synthetic skies yield null residuals and Z ≈ 0, ensuring the pipeline flags deviations.

## Closeout Confirmation
- **Issue #13** — Added `comet.nulls.rotation_null`, `hemisphere_jackknife`, and `curl_null_field`, enabling reusable residual construction for the Phase 4 variants.
- **Issue #14** — Delivered the `scripts/run_null_variants.py` CLI that orchestrates null evaluation and persists per-variant artifacts alongside a consolidated summary report.
- **Issue #15** — Introduced `tests/test_null_variants.py`, covering helper behaviour and asserting near-zero Z-scores for synthetic skies via `evaluate_null_tests`.

## Issue Closure Evidence
- **#13** — Unit tests `test_rotation_null_zero_for_symmetric_map`, `test_hemisphere_jackknife_sign_flip_symmetry`, and `test_curl_null_field_annihilates_constant_modes` in `tests/test_null_variants.py` validate the rotation, hemisphere jackknife, and curl helpers respectively.
- **#14** — Manual invocation of `scripts/run_null_variants.py` with quick-pipeline artifacts produces per-variant residual arrays and an aggregated JSON report, matching the automation acceptance criteria in the roadmap. The CLI guards for missing inputs and persists machine-readable outputs for downstream tooling.
- **#15** — `test_evaluate_null_tests_reports_small_z_for_synthetic_skies` exercises the evaluator end-to-end, asserting residual shapes and ensuring Z-scores remain ≈0 for idealized skies, demonstrating that regressions would be caught by CI.

## Verification
- `tests/test_null_variants.py::test_evaluate_null_tests_reports_small_z_for_synthetic_skies` confirms the aggregated evaluator yields Z ≈ 0 for mock skies (Issue #15).
- `tests/test_null_variants.py::test_rotation_null_zero_for_symmetric_map` and related tests cover each residual helper individually (Issue #13).
