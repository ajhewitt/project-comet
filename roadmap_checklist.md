# Project Comet: Roadmap Progress Checklist

### Phase 0 — Freeze What Works (Baseline)
- [x] Quick pipeline (`comet demo`) runs on synthetic data  
- [x] Unit tests present (`pytest`)  
- [x] Lint/format checks (ruff) wired in CI  
- [ ] Smoke CI job at **nside=64**, **nsims=5**  
- [ ] Confirm quick pipeline produces Δ, Cov, Z at nside=256  

### Phase 1 — Data Ingestion & Units Sanity
- [x] Implement `io_maps.read_fits_map_with_meta`
- [x] Write metadata JSON per input map
- [x] Add `scripts/check_data.py` for header validation
- [x] Tests for FITS header parsing and unit sanity checks

### Phase 2 — Masking, Apodization, and Binning
- [ ] Extend `commutator_common.build_mask`  
- [ ] Create `scripts/make_mask.py`  
- [ ] Tests: f_sky within [0.5, 0.9], bin count matches config  

### Phase 3 — Spectra Deconvolution & Windows
- [ ] Implement `namaster_utils.py` helpers  
- [ ] Config toggles for beam/pixel deconvolution  
- [ ] Tests: on/off toggle shifts high-ℓ tail  

### Phase 4 — Null Tests
- [ ] Add `scripts/run_null_variants.py`  
- [ ] Implement: rotation, hemisphere jackknife, curl/null field  
- [ ] Tests: synthetic skies → Z ≈ 0  

### Phase 5 — Simulations with Theory Cℓ
- [ ] Add `scripts/theory.py` loader  
- [ ] Extend `run_null_sims.py` with Gaussian alm generator  
- [ ] Run nsims ≈ 2000 offline  
- [ ] Tests: covariance is positive-definite  

### Phase 6 — Real T×κ Science Product
- [ ] Implement `compute_cross_spectrum.py`  
- [ ] Extend `summarize_results.py` with Δ/σ reporting  
- [ ] Tests: injected signal recovery  

### Phase 7 — Reproducibility & Sanity Locks
- [ ] Lock units (K_CMB, dimensionless κ)  
- [ ] Lock geometry (RING, coords)  
- [ ] Hash lock: record SHA256 of inputs  
- [ ] Version lock: record package versions + git commit  
- [ ] CI blocks data/artifacts/figures commits  
- [ ] Quick smoke test wired as gate  
