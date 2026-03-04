# Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth

**Authors:** Nathalie Hilmi, Loua Aurel De Vince Vehi, Marina Treskova, Cécile Sabourault, Lisa Levin

---

This repository provides supporting data, code, and visualization tools for the article:

**Hilmi, N., Vehi, L. A. D. V., Treskova, M., Sabourault, C., & Levin, L. (2025). _Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth_.**

The study develops an integrated metric of **Blue Carbon Wealth** (BCW) that combines **Coastal Blue Carbon Ecosystems** (BCEs) — mangroves, seagrasses, and saltmarshes — with the open-ocean **Biological Carbon Pump** (BCP), the large-scale transfer of organic carbon driven by phytoplankton. Uncertainty is propagated throughout via Monte Carlo simulation (B = 20 000 draws), and the **Global Social Cost of Carbon** (GSCC) is derived from a ridge-regression structure-adjusted estimator built on the Ricke et al. (2018) country-level CSCC database.

---

## Repository structure

```
├── data_source/                        # Core input datasets
│   ├── bcp/                            # Biological Carbon Pump data (BCP_dta.csv)
│   ├── economy/                        # GDP, population, CO₂ emissions, external debt
│   ├── gscc/                           # CSCC database (Ricke et al. 2018) + pre-computed draws
│   ├── shp/                            # EEZ shapefiles, world map, sequestration rates JSON
│   └── summary/                        # Intermediate aggregated outputs written by main.py and notebooks/eda.ipynb
│
├── src/                                # Python source package
│   ├── config.py                       # All paths, constants, and ISO/continent lookup tables
│   ├── bce_areas.py                    # BCE area extraction, grouping, and sequestration-rate sampling
│   ├── adding_eco_data.py              # Merging of economic/social variables (GDP, population, debt, CO₂)
│   ├── compute_gscc.py                 # Structure-adjusted GSCC distribution (ridge decomposition)
│   ├── compute_bcw.py                  # BCW and BCP valuation with full uncertainty propagation
│   ├── utils.py                        # General utilities (missingness audit, safe arithmetic, per-capita)
│   ├── visualization.py                # All figure-generation helpers (maps, charts, tables)
│   └── gscc_expl_utils.py              # Self-contained GSCC exploration toolkit for notebooks
│
├── notebooks/
│   ├── paper.ipynb                     # Figures and tables of the original paper
│   ├── eda.ipynb                       # Exploratory data analysis
│   └── gscc_explorer.ipynb             # Interactive GSCC computation explorer and robustness checks
│
├── output/                             # Outputs written by main.py
│   ├── country_level_bcw.csv           # Final BCW dataset (one row per country/territory)
│   ├── draws.npz                       # Full Monte Carlo draw matrices (GSCC, BCE, BCW)
│   ├── bcw_finance_indicators.csv      # BCW relative to GDP, debt, and CO₂ liability
│   ├── switchers_quadrant.csv          # Countries changing BCW quartile under sensitivity scenarios
│   ├── switchers_sink_status.csv       # Countries changing net sink/source status
│   ├── top_movers_delta_coverage.csv   # Largest shifts in BCE coverage
│   └── top_movers_delta_relief.csv     # Largest shifts in debt-relief potential
│
├── main.py                             # Pipeline entry point
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Methodology overview

| Step | Module | Description |
|------|--------|-------------|
| 1. BCE extraction | `src/bce_areas.py` | Merge EEZ boundaries with mangrove, seagrass, and saltmarsh area layers; sample sequestration rates from `sequestration_rates.json` using B draws |
| 2. Economic variables | `src/adding_eco_data.py` | Attach GDP (constant 2015 USD), population, CO₂ emissions (2023), and total external debt to each EEZ |
| 3. GSCC distribution | `src/compute_gscc.py` | Filter the CSCC database; aggregate country-level medians to scenario-level sums; apply ridge decomposition in log-space to remove label-structured variance; resample B adjusted draws |
| 4. BCW computation | `src/compute_bcw.py` | Multiply BCE sequestration draws × GSCC draws to obtain coastal BCW; add BCP point-estimate converted to monetary value; compute mean and SE from the joint draw matrix |
| 5. Post-processing | `main.py` + `src/visualization.py` | Per-capita normalisation; ISO / continent gap-filling; save outputs |

---

## Reproduce results

### 1. Clone this repository

```bash
git clone https://github.com/aurvl/bcw-paper-2025.git
cd bcw-paper-2025
```

### 2. Set up a Python environment

Python **3.11.9** is required.

```bash
python -m venv .venv
```

- **Windows:** `.\\.venv\\Scripts\\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python -m src.compute_gscc  # Optional: pre-compute the structure-adjusted GSCC distribution
python -m main              # Run the full pipeline from BCE extraction to BCW computation and output generation
```

This preprocesses all inputs, computes BCE sequestration uptakes, derives the structure-adjusted GSCC distribution, propagates uncertainty through the BCW calculation, and writes all outputs to `output/` and intermediate summaries to `data_source/summary/`.

### 5. Explore results in notebooks

Open the `notebooks/` folder and run:

| Notebook | Purpose |
|----------|---------|
| `paper.ipynb` | Reproduces all figures and tables from the article |
| `eda.ipynb` | Dataset-level exploration and quality checks |
| `gscc_explorer.ipynb` | Step-through of the GSCC ridge-adjustment method with robustness diagnostics |


---

## Key outputs

| File | Description |
|------|-------------|
| `output/country_level_bcw.csv` | BCW (mean ± SE), BCE and BCP components, economic indicators, ISO codes, and continent assignments for all countries and territories |
| `output/draws.npz` | Compressed NumPy archive of the full Monte Carlo draw matrices (`gscc_draws`, `bcw_draws`, `bce_*_draws`) |
| `data_source/summary/bce_data.csv` | Processed BCE areas with sequestration uptake statistics |
| `data_source/summary/unadjusted_bcw_data.csv` | BCW dataset before ISO/continent gap-filling |

---

## License

This repository uses a dual open-license model.

- All **code** (Python files and notebooks) is released under the **MIT License**.
- All **data and written content** (datasets, figures, and documentation) are released under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**.

---

## Citation

When reusing or citing this work, please reference:

**BibTeX:**

```bibtex
@misc{bcw_data_2025,
  author    = {Hilmi, Nathalie and Vehi, Loua Aurel De Vince and Treskova, Marina and Sabourault, Cécile and Levin, Lisa},
  title     = {Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth — Data and Code Repository},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/aurvl/bcw-paper-2025},
  note      = {Version 1.0. Licensed under MIT (code) and CC-BY 4.0 (data and text)},
}
```

**APA:**
> Hilmi, N., Vehi, L. A. D. V., Treskova, M., Sabourault, C., & Levin, L. (2025). *Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth — Data and Code Repository.* GitHub. https://github.com/aurvl/bcw-paper-2025