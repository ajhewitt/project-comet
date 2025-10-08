---
title: "Phase 1: Map metadata ingestion"
tags:
  - phase-1
  - io
status: done
---

## Tasks
- Implement `io_maps.read_fits_map_with_meta` returning structured metadata.
- Persist metadata JSON alongside pixel data utilities.
- Normalise units, ordering, and coordinate system info for downstream use.

## Notes
- Metadata JSON includes the raw FITS header (JSON-serialisable) for provenance.
- `MapData.write_metadata_json` now records `npix` in addition to curated fields.
