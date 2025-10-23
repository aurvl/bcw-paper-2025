# Revisiting Blue Natural Capital  
### Assessing Global and National Blue Carbon Wealth  

**Authors:** Nathalie Hilmi, Loua Aurel De Vince Vehi, Marina Treskova, Cécile Sabourault, Lisa Levin  

---

This repository provides supporting data, code, and visualization tools for the article:

**Hilmi, N., Vehi, L. A. D. V., Treskova, M., Sabourault, C., & Levin, L. (2025). _Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth_.**

The study develops an integrated metric of **Blue Carbon Wealth (BCW)** that combines **Coastal Blue Carbon Ecosystems (BCEs)** — mangroves, seagrasses, and saltmarshes — with the **open-ocean Biological Carbon Pump (BCP)**, the large-scale transfer of organic carbon driven by phytoplankton.

Using an updated **Global Social Cost of Carbon** (GSCC = US$ 573.8 ± 144.7 /tCO₂) from Ricke et al. (2018), the analysis quantifies both the **carbon removals** and their **monetized value** at global and national scales. Coastal ecosystems sequester about **112 ± 24.6 Mt C yr⁻¹**, valued at **US$ 236 ± 51.8 billion yr⁻¹**, while adding the BCP raises global BCW to **US$ 2.38 ± 0.35 trillion yr⁻¹** for **1.13 ± 0.17 Gt C yr⁻¹** of total removals.

Results show that **Small Island Developing States** hold the highest per-capita BCW (≈ US$ 5,200 yr⁻¹), whereas large economies dominate in absolute value. By integrating spatial data on coastal and oceanic processes under an equity-sensitive framework, the study positions BCW as a component of **inclusive wealth** and a foundation for **blue finance instruments** such as blue bonds, debt-for-nature swaps, and payments for ecosystem services.


---

## Repository content and usage guide

This repository hosts data and reproducible python scripts and notebooks accompanying the study.  
The structure below summarizes its components.

```
├── data_source/                   # Core input datasets
│ ├── bcp/                         # Biological Carbon Pump data
│ ├── economy/                     # GDP, debt, population, etc.
│ ├── gscc/                        # Global Social Cost of Carbon (Ricke et al. 2018)
│ ├── shp/                         # Spatial shapefiles (EEZ, BCEs, etc.)
│ └── summary/                     # Aggregated statistics and metadata
│
├── utils/                         # Helper modules and computation tools
│ ├── bce_areas.py                 # BCE area extraction and cleaning
│ ├── compute_gscc.py              # GSCC and BCW valuation functions
│ ├── 
│ └── 
│
├── generate_data.py               # Main script to build harmonized datasets
├── compute_bcw.py                 # Main script to compute Blue Carbon Wealth
├── notebook.ipynb                 # Jupyter notebook for analysis and visualization
├── bcw.csv                        # Final Blue Carbon Wealth outputs
├── requirements.txt               # Dependencies list for environment setup 
└── README.md                      # Repository description and documentation
```

### Reproduce results

1. **Clone this repository**  
    ```bash
    git clone https://github.com/<username>/<repo_name>.git
    cd <repo_name>
    ```
2. **Set up a Python environment**
    ```bash
    python -m venv env
    ```
    - ***Windows:***
        ```bash
        .\env\Scripts\activate
        ```
    - ***macOS/Linux:***
        ```bash
        source env/bin/activate
        ```
3. **Install required packages**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Python scripts**
    ```bash
    python generate_data.py # To preprocess and compile datasets
    python compute_bcw.py   # To calculate Blue Carbon Wealth of nations
    ```
5. **Visualize results using Jupyter Notebooks**
    Run `jupyter notebook` and open the notebooks in the `notebooks/` directory to generate figures and tables.

---

This repository uses a dual open-license model to ensure transparency and reproducibility.  

All *code* (Python files, utilities, and notebooks) is released under the **MIT License**, allowing free use, modification, and redistribution with proper attribution.  

All *data and written content* (datasets, figures, and documentation) are released under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**, permitting reuse and adaptation with attribution.  

When reusing or citing this work, please reference: 

* **BibTex:**

    ```latex
    @misc{bcw_repo,
    author       = {Hilmi, Nathalie and Vehi, Loua Aurel De Vince and Treskova, Marina and Sabourault, Cécile and Levin, Lisa},
    title        = {Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth — Data and Code Repository},
    year         = {2025},
    publisher    = {GitHub},
    url          = {https://github.com/aurvl/bcw-paper-2025},
    note         = {Version 1.0. Licensed under MIT (code) and CC-BY 4.0 (data and text)},
    }
    ```

* **APA:**
    > Hilmi, N., Vehi, L. A. D. V., Treskova, M., Sabourault, C., & Levin, L. (2025). *Revisiting Blue Natural Capital: Assessing Global and National Blue Carbon Wealth — Data and Code Repository.* GitHub. Available at: [https://github.com/aurvl/bcw-paper-2025](https://github.com/aurvl/bcw-paper-2025)