![Project Comet](docs/project-comet.png)

**Project Comet** is a reproducible analysis pipeline for studying the **cosmic microwave background (CMB)** and **CMB lensing** using Planck satellite data. The aim is to provide a modern, open, and automatable framework for exploring cosmological signals, validating theoretical models, and benchmarking analysis workflows.

This project integrates:
- **High-resolution Planck component maps** (SMICA CMB map and lensing convergence map).
- **NaMaster** for pseudo-$C_\ell$ estimation and cross-spectrum analysis.
- A modular **Python CLI** (`comet`) for configuration, running, and summarizing results.
- **Continuous integration** with scientific software dependencies pinned for reproducibility.

![CI](https://github.com/ajhewitt/project-comet/actions/workflows/ci.yaml/badge.svg?branch=main)

---

## Scientific Background

The **CMB** provides a snapshot of the universe at $z \sim 1100$, encoding both the primordial density fluctuations and later physical effects. Of particular interest:

- **CMB lensing**: Deflections of CMB photons by intervening large-scale structure. This remaps CMB anisotropies and encodes information about the matter distribution at $z \sim 2$.
- **Cross-correlations**: Combining CMB lensing with galaxy surveys or internal Planck products constrains cosmological parameters and tests $\Lambda$CDM.
- **Pseudo-$C_\ell$ techniques**: Estimation of angular power spectra in the presence of masks, implemented here via [NaMaster](https://namaster.readthedocs.io).

For background, see the references in the [docs](docs/) directory:
- Planck Collaboration (2018): *Planck 2018 results. VIII. Gravitational lensing*
- Alonso et al. (2019): *NaMaster: Master of the Mask*
- Other project-specific notes in `docs/*.pdf`.

---

## Features

- **Config-driven runs**: input data and pipeline steps specified via YAML (`config/prereg.yaml`, `config/paths.example.yaml`).
- **Automated data checks**: verifies presence and integrity of large Planck maps before processing.
- **Stable CLI interface**:
  ```bash
  ./bin/comet-run
  ```
  produces a JSON summary of run metadata and results.
- **Local + CI reproducibility**: identical environments with `micromamba`, verified via `./bin/ci`.
- **Extensible analysis**: current pipeline stubs compute metadata; next stage integrates NaMaster for $C_\ell$ estimation.

---

## Getting Started

### 1. Create environment
```bash
micromamba create -f environment.yml
micromamba run -n comet pip install -e ".[dev]"
```

### 2. Run tests & lint
```bash
./bin/ci
```

### 3. Stage the Planck maps

Download the Planck SMICA temperature and lensing convergence maps and place
them in the repository's `data/` directory:

```
project-comet/
└── data/
    ├── COM_CompMap_CMB-smica_2048_R1.20.fits
    └── COM_CompMap_Lensing_2048_R1.10.fits
```

If you keep the maps somewhere else, point the pipeline at that directory by
setting `COMET_DATA_DIR` before running the CLI:

```bash
export COMET_DATA_DIR=/path/to/planck/maps
```

You can confirm that the data are discoverable with the helper command:

```bash
micromamba run -n comet python -m comet.cli data --list
```

### 4. Run the pipeline on the staged data

Run the default analysis (the helper script now forwards any extra arguments to
the CLI, so you can tweak options such as `--ordering` if desired):

```bash
./bin/comet-run
```

The run writes its output to `artifacts/summary.json`. Inspect it with your
preferred JSON viewer (for example, `jq`):

```bash
jq . artifacts/summary.json
```

This will produce a JSON output like:
```json
{
  "ordering": "both",
  "results": {
    "nbins": 0,
    "z": 0.0,
    "notes": "stub"
  }
}
```

---

## Full science analysis workflow

The quick stub above is useful for smoke tests. To reproduce the
science-grade null test and cross-spectrum that the collaboration uses
for publication, follow the staged steps below. All commands assume you
are inside the repository root, have activated the environment with
`micromamba run -n comet`, and have staged the Planck maps as described
earlier.

1. **Confirm data discovery and record configuration hashes.**
   ```bash
   micromamba run -n comet python -m comet.cli data --list
   git status --short
   git rev-parse HEAD
   ```
   Capture the git commit ID and any environment hashes in your run log.

2. **Prepare a theory spectrum file.** The repository does **not** ship a
   fiducial lensing theory. If you only have the two Planck maps staged above,
   you can build a self-consistent theory table by computing the full-sky auto
   and cross spectra from those maps. Run the provided helper script, which
   reads the FITS files, evaluates the temperature auto-spectrum
   $C_\ell^{TT}$, the lensing convergence auto-spectrum $C_\ell^{\kappa\kappa}$,
   and their cross-spectrum $C_\ell^{T\kappa}$ with `healpy.anafast`, then
   writes the four-column ASCII file expected by the CLI. Feel free to pass
   `--lmax`, `--cmb-map`, or `--kappa-map` if your analysis setup differs.

   ```bash
   micromamba run -n comet python scripts/derive_theory_from_maps.py \
     --output-npz data/theory/tk_planck2018.npz
   ```

   By default the script saves `data/theory/tk_planck2018.txt`. The CLI
   utilities accept this plain-text file directly: the first column must be
   the multipole $\ell$, followed by $C_\ell^{TT}$, $C_\ell^{\kappa\kappa}$, and
   $C_\ell^{T\kappa}$. Supplying `--output-npz` also writes the NumPy archive
   used by the tests and scripts in this repository. Afterwards, inspect the
   theory coverage to confirm it matches your analysis range:
   ```bash
   micromamba run -n comet python scripts/theory.py data/theory/tk_planck2018.npz \
     --summary artifacts/theory_summary.json
   ```

3. **Generate both commutator orderings at full resolution.** Use the
   shared mask and preregistered binning (if available) when running the
   two orderings. Adjust `--quick-nside`, `--nlb`, `--lmin`, and related
   arguments to your publication settings (the example below runs at
   NSIDE 2048 with 30-wide bins):
   ```bash
   micromamba run -n comet python scripts/run_order_A_to_B.py \
     --data-dir "${COMET_DATA_DIR:-data}" \
     --quick-nside 2048 --nlb 30 --lmin 30 --lmax 2048 \
     --threshold-sigma 4.0 --apod-arcmin 60.0 \
     --out artifacts/order_A_to_B_full.npz

   micromamba run -n comet python scripts/run_order_B_to_A.py \
     --data-dir "${COMET_DATA_DIR:-data}" \
     --quick-nside 2048 --nlb 30 --lmin 30 --lmax 2048 \
     --threshold-sigma 4.0 --apod-arcmin 60.0 \
     --out artifacts/order_B_to_A_full.npz
   ```
   Each script emits a JSON sidecar summarizing the binning and mask
   choices. Archive both `.npz` payloads and their `.json` companions.

4. **Build the null covariance from simulations.** Supply the same
   geometry choices (NSIDE, binning, mask) and the theory spectrum from
   step 2. Increase `--nsims` until the minimum eigenvalue is stable;
   for publication we typically use ≥1000 realizations.
   ```bash
   micromamba run -n comet python scripts/run_null_sims.py \
     --data-dir "${COMET_DATA_DIR:-data}" \
     --quick-nside 2048 --nlb 30 --lmax 2048 \
     --theory data/theory/tk_planck2018.npz \
     --nsims 1000 --seed 2025 \
     --out-cov artifacts/cov_delta_full.npy
   ```
   Inspect the terminal summary for the covariance size and record the
   random seed alongside the command in your lab notebook.

   *Reusing legacy binning:* if you already have a long-running
   covariance generated with the pre-preregistration CLI defaults (for
   example, a 69×69 matrix from `--nlb 50`), rerun both ordering scripts
   with `--disable-prereg` and matching `--nlb`/`--lmax` settings so the
   Δ bandpowers align with that covariance.

5. **Form the commutator residual and null statistic.**
   ```bash
   micromamba run -n comet python scripts/compute_commutator.py \
     --order-a artifacts/order_A_to_B_full.npz \
     --order-b artifacts/order_B_to_A_full.npz \
     --cov artifacts/cov_delta_full.npy \
     --out-delta artifacts/delta_ell_full.npy \
     --out-summary artifacts/summary_full.json
   ```
   The resulting JSON contains the Δ vector length and the stabilized
   χ ("z") statistic for the null test.

6. **Assemble the science cross-spectrum.** Average the two orderings,
   compare to theory, and compute per-bin significances:
   ```bash
   micromamba run -n comet python scripts/compute_cross_spectrum.py \
     --order-a artifacts/order_A_to_B_full.npz \
     --order-b artifacts/order_B_to_A_full.npz \
     --theory data/theory/tk_planck2018.npz \
     --cov artifacts/cov_delta_full.npy \
     --out artifacts/cross_tk_full.npz \
     --summary artifacts/cross_summary_full.json
   ```
   Check `artifacts/cross_summary_full.json` to confirm the mean and
   maximum |z| are consistent with a null detection.

7. **Generate publication figures and a textual digest.**
   ```bash
   micromamba run -n comet python scripts/summarize_results.py \
     --delta artifacts/delta_ell_full.npy \
     --summary docs/summaries/full_run.json \
     --cov artifacts/cov_delta_full.npy \
     --cross artifacts/cross_tk_full.npz \
     --outdir docs/figures/full_run
   ```
   This produces plots of Δ bandpowers, the null histogram, the
   T×κ spectrum with uncertainties, and per-bin z-scores. Include these
   figures and the JSON summaries in your archival package.

8. **Archive provenance.** Save the executed command list, git commit
   hash, configuration files (`config/*.yaml`), the artifacts under
   `artifacts/`, and generated figures under `docs/figures/full_run/` in
   a versioned, timestamped directory for future audits and publication
   supplements.

Following these steps yields a repeatable end-to-end analysis: staging
data, constructing both commutator orderings, calibrating the covariance
from simulations, computing the null statistic, and delivering the
science cross-spectrum together with diagnostic plots ready for
publication.

---

## Project Structure

```
project-comet/
├── bin/               # CLI wrappers (comet-run, ci)
├── config/            # Example preregistration + paths configs
├── data/              # Large Planck FITS maps (ignored by Git)
├── docs/              # Scientific documentation, PDFs, figures
├── src/comet/         # Python package (cli, run, io_maps, etc.)
├── tests/             # Unit and smoke tests
├── artifacts/         # Generated outputs (ignored by Git)
├── environment.yml    # Micromamba environment definition
├── Makefile           # Common commands (make ci, make run, make lint)
└── README.md          # You are here
```

---

## References

- [Planck 2018 lensing paper (A&A, 641, A8)](https://arxiv.org/abs/1807.06210)
- [NaMaster: Master of the Mask](https://arxiv.org/abs/1809.09603)
- Additional project design notes and figures: see [`docs/`](docs/).
