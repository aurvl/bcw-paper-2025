from src.config import (
    EEZ_PATH, SALTMARSHES_PATH, SEAGRASSES_PATH, MANGROVES_PATH, 
    SELECT_COLS, SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL, CMOL,
    MANGROVES_AREA_COL, JSON_PATH, GROUP_PATH, POP_PATH, GDP_PATH, 
    CB_PATH, DEBT_PATH, BCP_PATH, CSCC_CSV, B_LIGHT, SEED_LIGHT,
    OUTPUT_DIR, ISO_MAP2, CONTINENT_MAP2, ISO_OVERRIDES2,
    SUMMARY_DIR,
)
from src.bce_areas import compute_uptakes, generate_bce_data
from src.compute_gscc import get_gscc_dist
from src.compute_bcw import bcw_computer
from src.adding_eco_data import add_eco_data
from src.utils import (
    audit_missingness, correct_kiribati, 
    per_capita
)
from src.eda_utils import complete_iso_and_continent

import numpy as np
from tabulate import tabulate
from colorama import Fore, Style


DEBUG_AUDIT = False

# ==========================
# PREPARING THE DATA
# ==========================
# BCEs areas by EEZs
bce_areas_df = generate_bce_data(EEZ_PATH,
                                 SALTMARSHES_PATH, SALTMARSHES_AREA_COL,
                                 SEAGRASSES_PATH, SEAGRASSES_AREA_COL, 
                                 MANGROVES_PATH, MANGROVES_AREA_COL, SELECT_COLS)

area_cols = [SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL, MANGROVES_AREA_COL]
if DEBUG_AUDIT:
    audit_missingness(
        bce_areas_df,
        area_cols,
        "after_generate_bce_data",
    )

# Adding other data
bce_areas_df = add_eco_data(bce_areas_df, GROUP_PATH, POP_PATH, GDP_PATH, CB_PATH, DEBT_PATH)

if DEBUG_AUDIT:
    audit_missingness(
        bce_areas_df,
        area_cols,
        "after_add_eco_data",
    )

# Compute BCEs sequestration rates
bce_columns = [SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL, MANGROVES_AREA_COL]
bce_df, bce_draws = compute_uptakes(
    bce_areas_df, JSON_PATH, 
    B=B_LIGHT, 
    bce_columns=bce_columns, 
    seed=SEED_LIGHT,
    keep_draws=True, 
    quantiles=None
)

if DEBUG_AUDIT:
    audit_missingness(
        bce_df,
        [
            *area_cols,
            "uptake_salt_mean",
            "uptake_salt_se",
            "uptake_seag_mean",
            "uptake_seag_se",
            "uptake_mang_mean",
            "uptake_mang_se",
            "uptake_total_mean",
            "uptake_total_se",
        ],
        "after_compute_uptakes",
        pairs=[
            (SALTMARSHES_AREA_COL, "uptake_salt_mean"),
            (SEAGRASSES_AREA_COL, "uptake_seag_mean"),
            (MANGROVES_AREA_COL, "uptake_mang_mean"),
            (SALTMARSHES_AREA_COL, "uptake_total_mean"),
        ],
    )
bce_df, bce_draws = correct_kiribati(bce_df, bce_draws)

if DEBUG_AUDIT:
    audit_missingness(
        bce_df,
        [
            *area_cols,
            "uptake_salt_mean",
            "uptake_seag_mean",
            "uptake_mang_mean",
            "uptake_total_mean",
            "uptake_total_partial",
        ],
        "after_correct_kiribati",
    )
path = SUMMARY_DIR / 'bce_data.csv'
bce_df.to_csv(path, index=False)

# ==========================
# COMPUTE BCW
# ==========================
# GSCC value
gscc_dict = get_gscc_dist(
    CSCC_CSV,
    B=B_LIGHT,
    seed=SEED_LIGHT,
    dmgfuncpar=["bootstrap", "estimates"],
    sampling_scheme="uniform",
    # SSP/RCP left as ALL by default
    value_col="50%",
)
gscc_draws = gscc_dict["draws_adj"]
gscc_value = np.mean(gscc_draws)
gscc_std = np.std(gscc_draws, ddof=1) if len(gscc_draws) > 1 else 0.0
gscc_se = gscc_std / np.sqrt(len(gscc_draws)) if len(gscc_draws) > 0 else 0.0

print(Fore.BLUE + "*"*25)
print("SUMMARY")
print("*"*25 + Style.RESET_ALL)
print(f"    - Global Social Cost of Carbon (GSCC): {Fore.BLUE}{gscc_value:.1f} US$/tCO2 (SE: {gscc_se:.2f}, std: {gscc_std:.2f}){Style.RESET_ALL}")

# BCW computation (uncertainty propagation)
data, bcw_draws = bcw_computer(
    bce_df,
    bce_tC_draws=bce_draws["total"],
    gscc_draws=gscc_draws,
    cmol=CMOL,
    bcp_path=BCP_PATH,
    debug_audit=DEBUG_AUDIT,
)
pcap_cols = ['Area_EEZ_KM2', 'GDP', 'CO2_emissions_2023', 'Debt (2015 US$)', 'Total BCW']
data = per_capita(data, pcap_cols)

data = data.drop_duplicates().reset_index(drop=True)

# ==========================
# FINAL SUMMARY STATISTICS
# ==========================
print(f"    - Number of countries/territories: {Fore.BLUE}{data.shape[0]}{Style.RESET_ALL}")

# Global totals
total = data['Total BCW'].sum(min_count=1)
total_se_sq = (data['Total BCW_se'] ** 2).sum(min_count=1)
total_se = np.sqrt(total_se_sq) if np.isfinite(total_se_sq) else np.nan
print(f"    - Global BCW : {Fore.BLUE}{(total / 1e12):.2f} trillion US$ (SE: {total_se / 1e12:.3f}){Style.RESET_ALL}")

# Per capita
bcw_per_capita = total / data['Population'].sum()
bcw_per_capita_se = total_se / data['Population'].sum()
print(f"    - Global BCW per capita: {Fore.BLUE}{bcw_per_capita:.2f} US$ (SE: {bcw_per_capita_se:.2f}){Style.RESET_ALL}")

# Average per country
n = data['Total BCW'].notna().sum()
se_of_mean = np.sqrt((data['Total BCW_se']**2).sum()) / n if n > 0 else np.nan
print(f"    - Average BCW per country: {Fore.BLUE}{(data['Total BCW'].mean()/1e9):.2f} billion US$ (SE: {se_of_mean/1e6:.2f} million US$){Style.RESET_ALL}")

# Top 5 countries by BCW
top5 = data.sort_values('Total BCW', ascending=False).head(5)[['country_name', 'Total BCW']]
print("    - Top 5 countries by BCW:\n")
print(Fore.BLUE, tabulate(top5, headers='keys', tablefmt='github', showindex=False, floatfmt=".2e"), Style.RESET_ALL)


# ==========================
# SAVE THE DATA
# ==========================
# Save CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
path = SUMMARY_DIR / 'unadjusted_bcw_data.csv'
data.to_csv(path, index=False)

# Save draws
np.savez_compressed(
    OUTPUT_DIR / "draws.npz",
    country_name=data["country_name"].astype(str).to_numpy(),
    ISO=data["ISO"].astype(str).to_numpy(),
    gscc_draws=np.asarray(gscc_draws, dtype=np.float32),
    bcw_draws=np.asarray(bcw_draws, dtype=np.float32),
    **{f"bce_{k}_draws": np.asarray(v, dtype=np.float32) for k, v in bce_draws.items()},
)

data = complete_iso_and_continent(
    data,
    iso_map=ISO_MAP2,
    continent_map=CONTINENT_MAP2,
    overrides=ISO_OVERRIDES2,
)
save_path = OUTPUT_DIR / 'country_level_bcw.csv'
data.sort_values(by='country_name').to_csv(save_path, index=False)

print(f'\n{Fore.GREEN}Data saved to "{save_path}"{Style.RESET_ALL}')

# end