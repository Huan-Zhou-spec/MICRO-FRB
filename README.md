# Micro-Lensing_FRB

**Evidence for Intermediate-Mass Black Holes From Microlensing Signatures in CHIME/FRB Catalog 2** 

This project selects multi-component Fast Radio Burst (FRB) candidates from the **Second CHIME/FRB Catalog (Catalog 2, 4539 sources)** that are potentially microlensed by compact point-mass objects (e.g. Primordial Black Holes, PBHs) along the line of sight. A non-detection of significant lensing pairs across the full sample is then used to derive a statistical upper bound on the cosmic fraction of PBHs, **f_PBH(M_PBH)**, as a function of the lens mass.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Data Description](#data-description)
- [Installation & Requirements](#installation--requirements)
- [Repository Structure](#repository-structure)
- [User Guide](#user-guide)
  - [1. CHIME/FRB Catalog Sub-setting](#1-chimefrb-catalog-sub-setting)
  - [2. Batch Download of CANFAR Dynamic Spectra](#2-batch-download-of-canfar-dynamic-spectra)
  - [3. Microlensing Candidate Selection Pipeline](#3-microlensing-candidate-selection-pipeline)
  - [4. Hardness-Ratio Consistency Test](#4-hardness-ratio-consistency-test)
  - [5. Sub-band Autocorrelation (ACF) Analysis](#5-sub-band-autocorrelation-acf-analysis)
  - [6. PBH Abundance Constraints & Summary Plot](#6-pbh-abundance-constraints--summary-plot)
  - [7. Deep-Dive Examples for Individual Candidates](#7-deep-dive-examples-for-individual-candidates)
- [Core Algorithms](#core-algorithms)
  - [Lensing-Candidate Selection Flow](#lensing-candidate-selection-flow)
  - [Hardness-Ratio Consistency](#hardness-ratio-consistency)
  - [Constraints on f_PBH](#constraints-on-f_pbh)
- [Output Products](#output-products)
- [References](#references)
- [Version History](#version-history)
- [License](#license)

---

## Project Overview

Fast Radio Bursts are millisecond-duration radio pulses of extragalactic origin. If a point-mass lens lies on the line of sight, the FRB signal may be split into multiple resolved images, appearing as **multiple peaks** in the time series with a characteristic time delay set by the lens mass, redshifts and lens geometry. Specifically, this project:

1. Extracts FRBs with multiple fitted sub-components (`sub_num ≥ 1`) from CHIME/FRB Catalog 2.
2. Combines **autocorrelation (ACF) peak detection**, **peak-pair matching**, **SNR ordering cuts**, and **KS spectral-drift tests** to rank true microlensing candidates.
3. Integrates the point-mass lensing optical depth over the whole sample to obtain a statistical upper limit on the PBH dark-matter fraction **f_PBH(M_L)**.
4. Derives lens mass estimates for the most promising individual candidates (e.g. **FRB 20211115A**, **FRB 20190131D**) and highlights them on the combined f_PBH exclusion plot.

---

## Key Features

| Module | Description |
|--------|-------------|
| **Catalog Sub-setting** | Splits the full 4539-source Catalog 2 into reusable subsets: repeaters / non-repeaters / first-repeaters / duplicate TNS-name entries / unique-source lists. |
| **Batch Data Download** | Downloads both HDF5 dynamic spectra and PDF plots for the 340 multi-peaked FRBs directly from the CANFAR repository (urllib / wget / auto). |
| **Lensing-Candidate Filter** | End-to-end pipeline: RFI cleaning → ACF spike detection → peak matching → SNR cuts → KS spectral-drift rejector. Runs with multiple smoothing configurations (SG windows 10/20/30/100, Gaussian σ=3). |
| **Hardness-Ratio Test** | Splits the observed bandpass into **k+2 sub-bands**; compares the high/low hardness ratios HR between the two lensing-image windows at confidence nσ. |
| **Sub-band ACF** | Computes independent autocorrelations for each sub-band and cross-checks that the delay spike aligns in all of them against the total-band reference. |
| **f_PBH Constraints** | Numerically integrates the point-mass lensing optical depth (with colossus ΛCDM distance functions) and outputs the f_PBH upper bound vs. lens mass. |
| **Plotting Suite** | Dynamic spectra (PDF + PNG with peak markers), KS drift heatmaps, QQ-matrices, ACF curves with spike markers, DM-width-SNR scatter plots, and the combined f_PBH multi-experiment limit plot. |

---

## Data Description

### 1. CHIME/FRB Catalog 2

Raw catalog file: `FRB_data/CHIME_cat2_frb/chimefrbcat2.npy` (4539 entries, numpy structured array).

Running `FRB_data.py::data_spearate()` produces the following subsets:

| File | Count | Description |
|------|-------|-------------|
| `chimefrbcat2_duplicates.npy` | 864 | Entries with repeated `tns_name` (one row per fitted sub-peak; number of repeats equals `sub_num`). |
| `chimefrbcat2_first_duplicates.npy` | 340 | First occurrence of each multi-component FRB (the 340 sources processed in the lensing pipeline). |
| `chimefrbcat2_unique.npy` | 4539 | Full catalogue de-duplicated by TNS name: first-occurrence of duplicates + unique sources. |
| `chimefrbcat2_unique_first_repeaters.npy` | 83 | Sources whose `repeater_name` matches the first published detection in Catalog 2. |
| `chimefrbcat2_unique_repeater.npy` | 981 | All sources flagged as repeaters (any `repeater_name`). |
| `chimefrbcat2_unique_non_repeater.npy` | 3558 | All non-repeating sources. |

Relevant fields used in the analysis: `tns_name`, `repeater_name`, `sub_num`, `dm_exc_ymw16`, `width_fitb`, `snr_fitb`, `low_freq`, `high_freq`.

### 2. CANFAR Dynamic Spectra

Stokes I dynamic spectra for the 340 multi-peaked sources are fetched from the CHIME/FRB Catalog 2 release hosted at CANFAR:

<https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data>

File naming:

- HDF5 data: `{tns_name}_stokesi_dynamic_spectrum.h5`
- PDF plot: `{tns_name}_stokesi_dynamic_spectrum_data.pdf`

Default download locations: `FRB_data/canfar_downloads/` (HDF5) or `Figures/canfar_downloads/` (PDF).

---

## Installation & Requirements

### Dependencies

| # | Package | Version | Install Method | Purpose |
|---|---------|---------|---------------|---------|
| 1 | `numpy` | >= 1.24 | conda / pip | Numerical computation, array operations |
| 2 | `scipy` | >= 1.10 | conda / pip | Signal processing (find_peaks, SG filter), statistics (KS test), integration |
| 3 | `matplotlib` | >= 3.7 | conda / pip | All plotting (dynamic spectra, ACF, heatmaps, f_PBH curves) |
| 4 | `pandas` | >= 2.0 | conda / pip | Tabular data handling, CSV export |
| 5 | `h5py` | >= 3.8 | conda / pip | Reading HDF5 dynamic spectrum data |
| 6 | `colossus` | >= 2.0.6 | **pip only** | Cosmological distances & Hubble parameter (Planck 2018) |

### Installation — Choose One Method

#### Method A: Conda (Recommended for local development)

`environment.yml` manages the full environment including Python version and C-level dependencies (HDF5, FFTW, etc.):

```bash
conda env create -f environment.yml
conda activate frb_microlens
```

#### Method B: Pip / Virtual Environment

`requirements.txt` provides a minimal dependency list for pip-based setups (CI/CD, Docker, Google Colab):

```bash
python -m venv frb_lens
source frb_lens/bin/activate   # Linux/macOS
# frb_lens\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Method C: Manual Conda Setup

```bash
conda create -n frb_lens python=3.11
conda activate frb_lens
conda install numpy scipy matplotlib pandas h5py
pip install colossus
```

### Note on `colossus`

`colossus` provides comoving-distance functions for the lensing optical-depth integral. If unavailable, an equivalent implementation using `astropy.cosmology.FlatLambdaCDM` can be swapped in via [modules/fpbh_data.py](file:///home/ubuntu/Desktop/MICRO-FRB/modules/fpbh_data.py).

---

## Repository Structure

```
Micro-Lensing_FRB/
├── modules/                          # Core reusable library
│   ├── __init__.py                   # Unified public API
│   ├── read_data.py                  # Reads Catalog 2 and HDF5 dynamic spectra; produces subsets
│   ├── load_data.py                  # CANFAR batch downloader (urllib / wget / auto)
│   ├── plot_data.py                  # All plotting: scatter, dynamic spectrum, ACF, KS heatmap, QQ
│   ├── analysis_data.py              # Peak finding, spectral extraction, KS test, ACF spikes
│   ├── smooth_data.py                # Smoothing primitives (boxcar / Gaussian / adaptive)
│   ├── hardness_data.py              # FWHM windowing + k+2 sub-band hardness-ratio test
│   └── fpbh_data.py                  # Integrates lensing optical depth → f_PBH(M_L)
│
├── FRB_data.py                       # Step 1+2: catalog subsetting + CANFAR download CLI
├── SearchLensedFRB.py                # Step 3: batch lensing selection for 340 FRBs
├── Hardness_test.py                  # Step 4: single-source hardness-ratio demo (FRB 20190131D)
├── fpbh.py                           # Step 6: f_PBH vs. M_L summary plot
│
├── FRB_data/
│   ├── CHIME_cat2_frb/               # Catalog 2 data + subsets (.npy files)
│   └── canfar_downloads/             # 340 HDF5 dynamic spectra (downloaded from CANFAR)
│
├── Figures/
│   ├── CHIME_cata2/                  # Catalog-level plots
│   ├── FRB_lensing_results_SG_20/    # Lensing pipeline: SG smoothing (window=20)
│   ├── FRB_lensing_results_SG_100/   # Lensing pipeline: SG smoothing (window=100)
│   └── FRB_lensing_results_G_3/      # Lensing pipeline: Gaussian smoothing (σ=3)
│
├── fpbh_bound/                       # Constraint curves + combined plot
│   ├── LSS.txt, Dynamical.txt, Accretion.txt, GWs.txt,
│   │   Microlensing.txt, Evaporation.txt, Ly.txt
│   ├── FRB_fpbh_vs_ML.txt            # f_PBH(M_L) data computed by this work
│   └── fpbh.pdf                      # Multi-experiment combined exclusion plot
│
├── environment.yml                   # Conda environment spec
├── requirements.txt                  # Python dependencies
├── .gitignore
├── CITATION.cff                      # Academic citation metadata
├── LICENSE                           # MIT License
└── README.md
```

---

## User Guide

### 1. CHIME/FRB Catalog Sub-setting

Invoke `data_spearate()` from [FRB_data.py](file:///home/ubuntu/Desktop/MICRO-FRB/FRB_data.py#L24-L60):

```bash
python -c "from FRB_data import data_spearate; data_spearate()"
```

- Input: `FRB_data/CHIME_cat2_frb/chimefrbcat2.npy`
- Output: the six `.npy` subsets listed under [Data Description](#1-chimefrb-catalog-2).

### 2. Batch Download of CANFAR Dynamic Spectra

The `data_load()` function is exposed as a `argparse` CLI in [FRB_data.py](file:///home/ubuntu/Desktop/MICRO-FRB/FRB_data.py#L64-L171):

```bash
# Default: download PDF plots for the 340 multi-peaked FRBs
python FRB_data.py --npy-file FRB_data/CHIME_cat2_frb/chimefrbcat2_first_duplicates.npy \
                   --output Figures/canfar_downloads

# Optional: benchmark the three download methods (urllib / wget / auto)
python FRB_data.py --method auto --benchmark
```

To switch to HDF5 raw data download instead, uncomment the corresponding `url` line inside `data_load()`.

### 3. Microlensing Candidate Selection Pipeline

```bash
python SearchLensedFRB.py
```

Runs the full pipeline over 340 sources as implemented in `process_frb_catalog_lens()` / `analyze_lensing_candidate()` in [SearchLensedFRB.py](file:///home/ubuntu/Desktop/MICRO-FRB/SearchLensedFRB.py#L27-L452). The per-source steps are:

1. **ACF spike detection** – smooth the autocorrelation and threshold peaks to obtain candidate delays `Δt_candidate`.
2. **Peak-pair matching** – cross-check each ACF spike with the actual spacing of the N highest-SNR peaks (± 2 ms tolerance).
3. **SNR ordering cuts** – for every matched pair (i, j), require `SNR_i ≥ SNR_j` and `SNR_j ≥ 10`; additionally require the highest-SNR peak overall to be covered by at least one matched pair.
4. **KS spectral-drift test** – extract peak spectra and compare with noise bootstrap samples via KS D-statistic; reject pairs with D > D_ci_upper (significantly drifting spectra).

Outputs are written into multiple result directories under `Figures/`, each corresponding to a different smoothing configuration. The currently committed results are:

| Directory | Smoothing Method | Parameter |
|-----------|-----------------|-----------|
| `Figures/FRB_lensing_results_SG_20/` | Savitzky–Golay | window = 20 |
| `Figures/FRB_lensing_results_SG_100/` | Savitzky–Golay | window = 100 |
| `Figures/FRB_lensing_results_G_3/` | Gaussian | σ = 3 |

Additional configurations (SG window=10, 30) can be generated by changing the `smooth_sigma` parameter in `SearchLensedFRB.py`.

Each directory contains per-FRB outputs (dynamic spectrum PNG, ACF PNG, KS heatmap, QQ matrix, report.txt) and aggregate summaries (`lens_catalog_summary.csv`, `lens_analysis_summary.txt`).

### 4. Hardness-Ratio Consistency Test

```bash
python Hardness_test.py
```

Runs the hardness-ratio workflow for FRB 20190131D (see [Hardness_test.py](file:///home/ubuntu/Desktop/MICRO-FRB/Hardness_test.py#L14-L109)):

- Splits the observed CHIME bandpass into `k+2` contiguous sub-bands (default `k=3` → 5 bands).
- Determines symmetric integer windows for each peak from the total-band FWHM.
- Computes net intensity per band with noise subtraction and Gaussian error propagation.
- Forms adjacent-band hardness ratios `HR_{i+1,i} = I_{i+1,net} / I_{i,net}` with uncertainties.
- Tests consistency between the peak1 (left image) and peak2 (right image) HR vectors at nσ confidence; print `PASS` if every pair is consistent.
- If both windows have positive net intensity, derives the **lens mass** from the time delay `Δt` and magnification ratio `R = I_L_net / I_R_net`:

  ```
  M_L (1+z_L) = (c³ Δt / 4G) · (R / (R-1)²)      [units: solar masses]
  ```

Tunable parameters (at top of the script): `dt_ms`, `k`, `n_sigma` (1.0 → 68 %, 2.0 → 95 %, 3.0 → 99.7 %).

### 5. Sub-band Autocorrelation (ACF) Analysis

[analyze_frb_candidate.py](file:///home/ubuntu/Desktop/MICRO-FRB/analyze_frb_candidate.py) includes a `run_acf_analysis()` function that:

1. Slices the 2D dynamic spectrum to the CHIME-observed frequency range.
2. Splits into `k_ACF = k_HR + 2` sub-bands, sums each sub-band over frequency → sub-band time series.
3. Calls `compute_autocorr_with_spikes()` on the total-band series and each sub-band series.
4. Plots all ACF curves overlaid; for the total-band curve highlights the detected spikes with grey ±2 ms shaded windows; for the sub-band curves, only spikes that fall within those shaded windows are individually marked, confirming a coherent delay across the band.

```bash
python analyze_frb_candidate.py FRB20190131D    # HR + ACF for FRB 20190131D
python analyze_frb_candidate.py FRB20211115A    # HR + ACF for FRB 20211115A
```

### 6. PBH Abundance Constraints & Summary Plot

#### 6.1 Compute f_PBH(M_L) data points

Un-comment the top example in [fpbh.py](file:///home/ubuntu/Desktop/MICRO-FRB/fpbh.py#L13-L41) and run:

```python
repeater_file     = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_first_repeaters.npy'
non_repeater_file = 'FRB_data/CHIME_cat2_frb/chimefrbcat2_unique_non_repeater.npy'

data = load_and_clean_data(repeater_file, non_repeater_file)
ML_values = np.logspace(0, 5, 50)
for ML in ML_values:
    fpbh(ML, data['DM_wo_mw'], data['wi'], data['snr'])
```

The resulting two-column `M_L [M_sun]`, `f_PBH_upper` array is saved to `fpbh_bound/FRB_fpbh_vs_ML.txt`.

#### 6.2 Draw the combined f_PBH limit plot

```bash
python fpbh.py
```

Produces `fpbh_bound/fpbh.pdf`, a single log-log canvas that compares:

- LSS          – Large Scale Structure (solid grey)
- Dynamical    – dynamical constraints (dashed grey)
- Accretion    – CMB-energy injection bounds (dash-dot grey)
- GWs (LIGO)   – Gravitational-wave merger rate (dashed grey + star marker)
- Microlensing – Optical microlensing (OGLE, dotted grey)
- **FRB (this work)** – red solid line with filled exclusion region
- **Inset zoom panel** – green horizontal band for FRB 20211115A; blue horizontal band for FRB 20190131D; connected via `ConnectionPatch` to the main plot region.

### 7. Deep-Dive Examples for Individual Candidates

A unified analysis script is provided for the strongest candidates:

| File | Usage | Target FRB | Δt (ms) | Sub-band config | Analysis performed |
|------|-------|------------|---------|-----------------|--------------------|
| [analyze_frb_candidate.py](file:///home/ubuntu/Desktop/MICRO-FRB/analyze_frb_candidate.py) | `python FRB 20190131D.py` | FRB 20190131D | 8.82 | k_HR = 7 (9 bands) | HR consistency + sub-band ACF + estimated lens mass M_L(1+z_L) ≈ 466.5 M_sun |
| (same script) | `python FRB20211115A.py` | FRB 20211115A | 6.86 | k_HR = 2 (4 bands) | HR consistency + sub-band ACF + scaled waveform superposition (peak2 → peak4) + M_L(1+z_L) ≈ 609 M_sun |

---

## Core Algorithms

### Lensing-Candidate Selection Flow

```
Input: FRB dynamic spectrum (F × T)
 │
 ├─► RFI clipping (3-sigma clusters) + f_down frequency downsample
 ├─► ts = sum over frequency channels → 1D total intensity time series
 │
 ├─► ACF(ts) → smooth + threshold → candidate delays Δt_spike
 │
 ├─► SNR peak detection → top-N peaks (N = sub_num+1 or sub_num+2)
 │
 ├─► Cross-match: search for peak pairs with |Δt_peak − Δt_spike| ≤ 2 ms
 │
 ├─► Hard SNR filter (exclude if any fails)
 │    ① max-SNR peak must be part of any matched pair
 │    ② for each pair (i, j): SNR_i ≥ SNR_j
 │    ③ each posterior image must have SNR_j ≥ 10
 │
 └─► KS spectral-drift test (1000 bootstrap noise samples):
       D_stat ≤ D_ci_upper  →  no significant drift ⇒ candidate = YES
```

### Hardness-Ratio Consistency

- Define per-band net intensity with noise bias subtraction:

  `I_i,net = Σ band_i(t) − N_t · ⟨noise_i⟩`

  with Gaussian error `σ_I_i = RMS_i · √N_t + extra terms from bg subtraction`.

- Adjacent hardness ratio and propagated error:

  ```
  HR_{i+1,i} = I_{i+1,net} / I_{i,net}
  σ_HR / HR  = √[(σ_{I,i+1} / I_{i+1,net})² + (σ_{I,i} / I_{i,net})²]
  ```

- Each HR measured at peak1 is compared to the same HR at peak2. If for every sub-band pair

  `|HR^{(peak1)} − HR^{(peak2)}| ≤ n_sigma · √(σ_1² + σ_2²)`

  we declare the pair to be consistent (PASS).

### Constraints on f_PBH

**Point-mass time-delay** between the close pair of images parametrised by `y_min ≤ y ≤ y_max`:

```
Δt(y) = (4 G M_L / c³) · (1+z_l)
        · [ y/2 · √(y²+4) + ln( (√(y²+4)+y) / (√(y²+4)−y) ) ]
```

Enforce `Δt(y_max) = W_i` (pulse width) and `R_i = 1 + SNR_i/10` to bound the magnification ratio and hence `y_max = √[(1+R)/√R − 2]`.

**Differential optical depth** (using Planck-2018 distances from `colossus`):

```
dτ/dz_l = 1.5 · Ω_m · H_0² / (H(z_l) · c)
          · (D_l · D_ls / D_s)
          · (y_max² − y_min²)
          · (1 + z_l)²
```

Integrate each FRB over lens redshifts from 0 → z_s(DM_exc_ymw16) via `scipy.integrate.quad`, then sum over the full catalog:

```
τ_total(M_L) = Σ_i τ_i(M_L, z_s,i, W_i, R_i)
```

Finally the Poisson-corrected upper bound (with N_len2=1, i.e. 68 % CL for zero observed events):

```
f_PBH(M_L) ≤ ln(1 − N_len2/N_src) / ln(1 − 1/N_src) · (1/τ_total)
```

Details in [modules/fpbh_data.py](file:///home/ubuntu/Desktop/MICRO-FRB/modules/fpbh_data.py#L25-L85).

---

## Output Products

After running the major scripts the repository produces:

```
FRB_data/
  ├─ CHIME_cat2_frb/             # 7 catalog .npy files (raw + 6 subsets)
  └─ canfar_downloads/           # 340 HDF5 dynamic spectra (downloaded from CANFAR)
Figures/
  ├─ CHIME_cata2/                # Catalog-level plots (e.g. frb_redshift_width_snr.pdf)
  └─ FRB_lensing_results_*/      # Lensing pipeline output per smoothing config:
                                 #   SG_20, SG_100, G_3 (committed);
                                 #   SG_10, SG_30 also available by re-running SearchLensedFRB.py
fpbh_bound/
  ├─ FRB_fpbh_vs_ML.txt          # f_PBH(M_L) data computed from the FRB sample
  ├─ fpbh.pdf                    # Multi-experiment combined exclusion plot
  └─ *.txt                       # External experiment constraint curves
```

Each lensing-result directory contains per-FRB outputs (dynamic spectrum PNG, ACF PNG, KS heatmap, QQ matrix, report.txt) and aggregate summaries (`lens_catalog_summary.csv`, `lens_analysis_summary.txt`).

---

## References
- "Huan Zhou, Zhengxiang Li, Cheng-Gang Shao, et al. (2025). Evidence for Intermediate-Mass Black Holes From Microlensing Signatures in CHIME/FRB Catalog 2. arXiv:2605.19653"
- **CHIME/FRB Catalog 2 Data Release (CANFAR)** – <https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data>
- **colossus** cosmology library – <https://bdiemer.bitbucket.io/colossus/>

---

## Version History

- **Version**: v1.0.0
- **Status**: actively maintained
- **Changelog**:
  - **v1.0.0**: Full initial release — CHIME/FRB Catalog 2 subsetting, 340-source lensing candidate pipeline, hardness-ratio test, sub-band ACF diagnostics, and combined f_PBH vs. other experiments plot.

---

## License

This project is released under the **MIT License**. See the separate [LICENSE](LICENSE) file for the full text.
