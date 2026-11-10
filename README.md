# Robust Co-Design of Measurement Operator, Electrode Geometry and Frequency Subset for Planar Spiral Impedance Sensors

Code, derived data and results that accompany the manuscript of the same title, submitted to *IEEE Transactions on Instrumentation and Measurement*.

The repository implements:

- a measurement model of a bifilar (double) spiral coplanar-electrode EIS sensor: conformal-mapping cell constant verified by 2-D finite elements, Cole–Cole medium, electrode polarization, trace/lead/fixture parasitics, temperature drift and channel-level fabrication tolerances;
- single-channel and differential (sample − reference) measurement operators with digital or analog readout noise;
- the nuisance-marginalized (efficient) Fisher information and the robust worst-case E/D-criteria;
- the convex frequency-subset relaxation, rounding with exchange refinement and a rounding certificate, integer geometry search, and the exchange method for the worst case over the medium parameters;
- baselines (B1–B7) and ablations, worst-case information evaluation, Monte Carlo estimation with a higher-fidelity generator and stress tests;
- the experimental validation on measured printed-electrode spectra.

## Repository layout

```
code/      Python sources (run from inside this folder)
data/      binned measured spectra (see "Data" below)
results/   all intermediate and final results (JSON / pickle)
figs/      figures of the manuscript and the supplementary material
tables/    LaTeX table rows and number macros generated from the results
```

| File | Purpose |
|---|---|
| `model.py` | forward model, operators, noise, nuisance prior, Fisher information |
| `fem_cell.py` | finite-element conductance of periodic coplanar strips |
| `design.py` | relaxation, rounding, exchange method, geometry search, baselines, worst-case evaluation |
| `run_designs.py` | computes every design (baselines, proposed, ablations) → `results/designs.json` |
| `evaluate.py` | worst-case information of every design → `results/eval_fim.json` |
| `estimate.py` | Monte Carlo estimation: `main`, `ideal`, `stress` |
| `worstmc.py` | repeated estimation at the hardest medium |
| `analysis_rounding.py` | rounding certificate and exhaustive check |
| `realdata.py`, `realexp.py`, `conc_eval.py` | validation on measured spectra |
| `fig_real.py` | summary statistics of the measured-spectra study → `results/real_summary.json` |
| `fig_concept2.py`, `fig_v2.py`, `fig_results.py` | manuscript and supplementary figures |
| `gen_tables.py` | LaTeX tables and number macros → `tables/` |
| `figstyle.py` | common plotting style |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Tested with Python 3.12, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 and scikit-fem 12.0.

## Reproducing the results

All scripts are run from `code/`. The stored files in `results/` allow the figures and tables to be regenerated without repeating the computations.

```bash
cd code
python fem_cell.py                 # FEM vs conformal mapping (console check)
python run_designs.py              # designs (about 1 h on two cores)
python evaluate.py                 # worst-case information
python estimate.py main            # Monte Carlo, higher-fidelity generator
python estimate.py ideal           # Monte Carlo, design model as generator
python estimate.py stress          # nuisance amplification and reference error
python worstmc.py                  # estimation at the hardest medium
python analysis_rounding.py        # rounding certificate
python realexp.py                  # measured spectra: designs and estimates
python fig_real.py                 # measured spectra: summary statistics
python fig_concept2.py && python fig_v2.py && python fig_results.py   # figures
python gen_tables.py               # tables and number macros
```

`realexp.py` reads `results/gapfits.pkl`, the full-spectrum fits of the electrode-gap study, which can be regenerated with the fitting routine in `realdata.py`. Random seeds are fixed in the scripts; the design step is deterministic.

## Data

`data/cam_eis_binned.json` contains logarithmically binned versions (96 bins per spectrum: mean real and imaginary part, within-bin standard deviation and number of points) of measured spectra from the open dataset

> T. Wade, S. Malik, L. Ives, N. Ćatić and S. Kar-Narayan, "Research data supporting 'Purely electrical detection of electrolyte concentration through microfluidic impedance spectroscopy'," Apollo – University of Cambridge Repository, 2024, doi:10.17863/CAM.109929 (CC BY 4.0),

associated with T. Wade *et al.*, *Cell Reports Physical Science* 5(8), 102133, 2024. The electrode-gap, electrode-length, finger-length and sample-to-sample studies are included. Please cite the original dataset when reusing these data.

## License

Code: MIT (see `LICENSE`). The derived data in `data/` follow the CC BY 4.0 license of the original dataset.
