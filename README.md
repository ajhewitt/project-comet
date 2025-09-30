# Project Comet — A Lensing–ISW Commutator Test

A crisp, pre-registered consistency test for cosmological pipelines:
compute the **order non‑commutativity** between two logically equivalent
orderings of the ISW×lensing cross and test for **context coupling**
aligned with ecliptic/scan templates.

If ΛCDM + context‑free priors are right, the commutator is consistent with zero.
If not, there is a repeatable, sign‑fixed deviation.

---

## TL;DR

- **Primary outcome:** Z‑score of the commutator \(\Delta_{\rm comm}\) and the context‑projected statistic \(S_\gamma\).
- **Data:** Planck PR4 (NPIPE) SMICA temperature, PR4 lensing \(\hat\phi\), WMAP9 ILC for split checks, official masks, exposure/scan maps.
- **Tools:** `healpy`, `pymaster` (NaMaster), `numpy`, `scipy`, `astropy`.
- **Runtime:** Minutes for analysis; hours for 1k null simulations on a 16‑core box.
- **Result:** Detection (pre‑registered sign) or tight null bounds on a rank‑1 context coupling \(\lambda\).

---

## Repo layout

```
project_comet/
├─ README.md
├─ environment.yml
├─ .gitignore
├─ config/
│  ├─ prereg.yaml                # frozen masks, ℓ‑ranges, bins, sign predictions
│  └─ paths.yaml                 # local paths to data products
├─ data/                         # FITS here (ignored by git)
├─ docs/
│  ├─ pbc_dual_construction.pdf  # concept PDF
│  ├─ pbc_inference.pdf          # framework PDF
│  ├─ p2_commutator_program.pdf  # program PDF
│  └─ tex/                       # doc LaTeX source
├─ scripts/
│  ├─ fetch_planck_pr4.py        # downloads PR4 maps/masks/lensing
│  ├─ build_context_template.py  # builds ecliptic/scan template basis (frozen)
│  ├─ run_order_A_to_B.py        # lensing → ISW ordering
│  ├─ run_order_B_to_A.py        # ISW → lensing ordering
│  ├─ compute_commutator.py      # Δ_comm + S_γ with pre‑registered weights
│  ├─ run_null_sims.py           # ΛCDM + mask/beam/noise sims
│  └─ summarize_results.py       # tables/plots; writes docs/figures
├─ src/
│  ├─ io.py
│  ├─ masking.py
│  ├─ namaster_utils.py
│  ├─ isw_filters.py
│  ├─ commutator.py
│  └─ context_template.py
└─ docs/figures/                 # auto‑created by summarize_results
```

---

## Pre‑registration

All pre‑registered choices live in `config/prereg.yaml`:

- **Masks:** Planck PR4 common T mask; official lensing mask. A single conservative variant allowed.
- **Multipoles:** `T: 2–64`, `φ: 8–2048`. Fixed binning (e.g., `nbin=10`, `lmin=8`, `lmax=100` for the commutator statistic).
- **Frequency combo:** SMICA primary; WMAP9 ILC only for split validation.
- **Estimator:** Hu–Okamoto QE for φ **(using public PR4 φ maps to avoid re‑running QE)**; pseudo‑Cℓ with MASTER for all cross‑spectra.
- **Context template:** ecliptic‑aligned basis from exposure/scan maps; orthogonalized to zodiacal and beam templates. **Publish the code and the fixed leading mode `c`.**
- **Sign predictions:** pre‑declare the expected sign of `Δ_comm` projection onto `c`.
- **Held‑out split:** e.g., Planck half‑mission B is never touched until the end.
- **Multiplicity:** testing K nearby context modes, use FDR at q=0.1. Default is K=1.

Commit `config/prereg.yaml` before any analysis run.

---

## Data

You need these public products (paths go in `config/paths.yaml`):

- **Planck PR4 (NPIPE) temperature maps**: [ESA Planck Legacy Archive](https://pla.esac.esa.int/#cosmology)
- **Planck PR4 lensing maps**: [Planck PR4 lensing products](https://pla.esac.esa.int/#cosmology)
- **WMAP9 maps**: [NASA LAMBDA archive](https://lambda.gsfc.nasa.gov/product/map/dr5/m_products.cfm)

Typical working set on disk: **1–3 GiB**. The full PR4 bundle is ~10 GiB, but not required in full.

---

## Installation

```bash
conda env create -f environment.yml
conda activate comet
# or: uv / pip, if preferred. Just mirror environment.yml packages.
```

Key deps: `python>=3.10`, `healpy`, `pymaster` (NaMaster), `numpy`, `scipy`, `astropy`, `pyyaml`, `tqdm`, `matplotlib`.

---

## Quickstart (happy path)

1) **Fetch data** (or copy in and edit `config/paths.yaml`):
```bash
python scripts/fetch_planck_pr4.py --out data/
```

2) **Freeze the context template** (creates `docs/context_template.npz` and `docs/context_template_report.md`):
```bash
python scripts/build_context_template.py --exposure data/exposure.fits --scan data/scan.fits   --zodiacal data/zodi_template.fits --mask data/common_tmask.fits --out docs/context_template.npz
```

3) **Run the two orderings** (A→B and B→A) with identical binning and masks:
```bash
python scripts/run_order_A_to_B.py --config config/prereg.yaml --paths config/paths.yaml --out data/out_A.npz
python scripts/run_order_B_to_A.py --config config/prereg.yaml --paths config/paths.yaml --out data/out_B.npz
```

4) **Compute commutator and context‑projection**:
```bash
python scripts/compute_commutator.py --A data/out_A.npz --B data/out_B.npz   --context docs/context_template.npz --config config/prereg.yaml --out data/commutator.json
```

5) **Null sims (optional but recommended)**:
```bash
python scripts/run_null_sims.py --config config/prereg.yaml --paths config/paths.yaml   --nsims 1000 --out data/null_sims/
```

6) **Summarize and plot**:
```bash
python scripts/summarize_results.py --comm data/commutator.json --sims data/null_sims/   --out docs/figures/
```

The final PDF/plots go in `docs/figures/`. The primary numbers are the Z‑scores for `Δ_comm` and `S_γ`.

---

## Code hooks (what each script does)

- `fetch_planck_pr4.py` — download or verify presence of FITS products; write `paths.yaml`.
- `build_context_template.py` — assemble feature matrix from exposure/scan, regress out zodiacal & beam templates, normalize, export leading mode `c` and an orthogonal basis.
- `run_order_A_to_B.py` — construct ISW proxy from low‑ℓ T, cross with lensing φ using NaMaster (MASTER deconvolution), write binned C_L.
- `run_order_B_to_A.py` — identical binning/masks; compute the reversed ordering.
- `compute_commutator.py` — compute `Δ_comm = Σ w_L (C_L^{A→B} − C_L^{B→A})` and `S_γ` by projecting onto context mode(s); weights `w_L` come from sims or analytic N0.
- `run_null_sims.py` — ΛCDM Gaussian skies + masks/beam/noise; produce distributions and covariances for the stats.
- `summarize_results.py` — one table + two plots: (i) binned C_L difference with 1σ band, (ii) null distribution with observed `Δ_comm`/`S_γ` marker.

---

## Config files

- `config/prereg.yaml`
  ```yaml
  masks:
    T_mask: data/planck_common_tmask.fits
    phi_mask: data/planck_lensing_mask.fits
  ells:
    T: [2, 64]
    phi: [8, 2048]
    bins: {lmin: 8, lmax: 100, nlb: 10}
  frequency: {T: SMICA}
  splits: {held_out: HM_B, use: [HM_A, DETSET1, DETSET2, F143, F217]}
  multiplicity: {method: FDR, q: 0.1, K: 1}
  context_sign_prediction: +1  # example; set before analysis
  ```

- `config/paths.yaml`
  ```yaml
  T_map: data/SMICA_PR4_T.fits
  T_mask: data/planck_common_tmask.fits
  phi_map: data/planck_pr4_phi.fits
  phi_mask: data/planck_lensing_mask.fits
  exposure_map: data/planck_exposure.fits
  scan_map: data/planck_scan.fits
  zodiacal_template: data/zodi_template.fits
  ```

---

## Performance & budget

- **Local workstation (16 cores, 32–64 GB RAM):** full analysis < 1 hour; 1k null sims overnight.
- **AWS c6i.4xlarge (16 vCPU, 32 GB, ~$0.68/hr):** 1k sims in a few hours with parallel batches; total compute well **< $100** for the whole study.
- **Storage:** < 3 GB working set; S3 < $1/mo.

---

## Reproducibility
- Code is deterministic given seeds; configs are versioned; held‑out split is enforced by the driver.
- All figures regenerate from `data/*.npz` and `data/null_sims/`.
- Pre‑registration is explicit in `config/prereg.yaml` and must be committed before any run.

---

## Related Work

Cross-correlations of CMB temperature with lensing potential have been studied as probes of the ISW effect and structure growth.
- Planck Collaboration 2015 ISW analysis ([A&A 594, A21](https://doi.org/10.1051/0004-6361/201525831))
- Carron et al. 2022, “Joint ISW–lensing constraints” ([arXiv:2206.07773](https://arxiv.org/abs/2206.07773))

These works validate the ISW×φ observable but treat it as a single pipeline. **Project Comet** instead introduces an explicit *commutator test* that compares two orderings of the analysis, with deviations interpreted as context-sensitive systematics.

---

## License
BSD-3-Clause for code; data under their respective public licenses (ESA/Planck, NASA/WMAP).
