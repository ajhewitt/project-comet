---
title: "Phase 1: Data validation CLI"
tags:
  - phase-1
  - cli
status: done
---

## Tasks
- Create `scripts/check_data.py` to validate Planck FITS headers.
- Enforce expected units (K_CMB for SMICA, dimensionless for κ).
- Write metadata JSON artefacts and a summary manifest.
- Add regression tests covering success and failure cases.

## Outcome
- Running the script against mocked maps produces per-map metadata JSON files and a summary document.
- Validation failures surface clear errors and stop the run with a non-zero exit code.
