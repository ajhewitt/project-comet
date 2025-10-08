---
title: "Phase 2: Masking, apodization, and binning"
tags:
  - phase-2
  - masking
  - binning
status: done
issues:
  - 7
  - 8
  - 9
---

## Tasks
- Extend `commutator_common.build_mask` to support configurable thresholds and C1 apodization.
- Provide a `scripts/make_mask.py` CLI that writes the combined mask and metadata artefacts.
- Wire preregistration binning into the commutator helpers with regression tests covering the expected bin count.

## Outcome
- Masks derived from Planck maps now apply RMS thresholding followed by configurable apodization, with sky fraction summaries.
- The new CLI produces reproducible mask artefacts and reports metadata including the resulting `f_sky`.
- Bin definitions follow the preregistration YAML, and tests assert both the mask sky fraction window and bin multiplicity.

## Closeout Confirmation

- **Issue #7** — Mask construction now applies configurable thresholding and optional apodization via `threshold_mask` and `apodize_mask`, with regression coverage enforcing the expected sky fraction window. See `comet.masking` and `tests/test_masking_and_binning.py::test_build_mask_f_sky_within_expected_range` for implementation and validation details.
- **Issue #8** — The dedicated `scripts/make_mask.py` CLI writes reproducible mask artefacts alongside metadata, satisfying the automation requirements for producing reusable masks.
- **Issue #9** — `nm_bins_from_config` consumes the preregistration bin settings and passes the associated regression test to guarantee the bin multiplicity matches the YAML configuration.
