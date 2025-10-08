---
title: "Phase 2: Masking, apodization, and binning"
tags:
  - phase-2
  - masking
  - binning
status: done
---

## Tasks
- Extend `commutator_common.build_mask` to support configurable thresholds and C1 apodization.
- Provide a `scripts/make_mask.py` CLI that writes the combined mask and metadata artefacts.
- Wire preregistration binning into the commutator helpers with regression tests covering the expected bin count.

## Outcome
- Masks derived from Planck maps now apply RMS thresholding followed by configurable apodization, with sky fraction summaries.
- The new CLI produces reproducible mask artefacts and reports metadata including the resulting `f_sky`.
- Bin definitions follow the preregistration YAML, and tests assert both the mask sky fraction window and bin multiplicity.
