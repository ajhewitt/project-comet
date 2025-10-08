---
title: "Phase 3: Spectra deconvolution and windows"
tags:
  - phase-3
  - spectra
  - windows
status: done
issues:
  - #10
  - #11
  - #12
---

## Tasks
- Implement reusable NaMaster helper utilities that expose pixel and beam window handling.
- Add preregistration-controlled toggles for applying the pixel window and enabling beam deconvolution in the bandpower pipeline.
- Cover window toggles with regression tests to confirm the beam correction modifies the high-ℓ tail.

## Outcome
- `comet.namaster_utils` now provides window-aware bandpower helpers plus a structured `WindowConfig`, enabling both CLI scripts and library code to consistently deconvolve NaMaster spectra.
- Pipeline runners read the preregistration `windows` block and record the applied settings, ensuring pixel/beam corrections are reproducible via configuration alone.
- New unit tests validate configuration parsing and demonstrate that enabling beam deconvolution changes the high-ℓ bandpower tail, locking in the expected behaviour.

## Closeout Confirmation

- **Issue #10** — Added `WindowConfig`, pixel/beam window evaluation, and configurable bandpower deconvolution helpers in `comet.namaster_utils`, fulfilling the helper expansion requirement.
- **Issue #11** — Updated both `run_order_A_to_B.py` and `run_order_B_to_A.py` to respect preregistration window toggles, recording metadata alongside the generated spectra.
- **Issue #12** — Introduced `tests/test_namaster_windows.py` to exercise configuration parsing and confirm that activating beam deconvolution elevates the high-ℓ bandpower tail versus the uncorrected spectrum.
