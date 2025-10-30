from utils.bce_areas import generate_bce_data, compute_rates
from utils.compute_bcw import gscc_computer, bcw_computer
from utils.adding_eco_data import add_eco_data
from utils.functions import correct_kiribati, per_capita
# import pandas as pd


# ==========================
# CONSTANTS AND PATHS
# ==========================
eez_path = r'data_source\shp\data_EEZ_areas_by_zone.xlsx'
saltmarshes_path = r'data_source\shp\data_saltmarshes_areas_by_country.xlsx'
seagrasses_path = r'data_source\shp\data_seagrasses_areas_by_country.xlsx'
mangroves_path = r'data_source\shp\data_mangroves_areas_by_country.xlsx'
gscc_path = r'data_source\gscc\cscc_db_v2.csv'

select = ['UNION', 'TERRITORY1', 'ISO_TER1', 'SOVEREIGN1']

saltmarshes_area_col = 'saltmarshes_area_km2'
seagrasses_area_col = 'seagrasses_area_km2'
mangroves_area_col = 'mangroves_area_km2'

json_path =r'data_source\shp\sequestration_rates.json'

group_path = r'data_source\economy\country_classification.csv'
pop_path = r'data_source\economy\population.xlsx'
gdp_path = r'data_source\economy\gdp.xlsx'
cb_path = r'data_source\economy\annual-co2-emissions-per-country.csv'
debt_path = r'data_source\economy\TotalExternalDebt.csv'

bcp_path = r'data_source\bcp\BCP_dta.csv'

# ==========================
# PREPARING THE DATA
# ==========================
# BCEs areas by EEZs
bce_areas_df = generate_bce_data(eez_path,
                                 saltmarshes_path, saltmarshes_area_col,
                                 seagrasses_path, seagrasses_area_col, 
                                 mangroves_path, mangroves_area_col, select)

# Adding other data
bce_areas_df = add_eco_data(bce_areas_df, group_path, pop_path, gdp_path, cb_path, debt_path)

# Compute BCEs sequestration rates
bce_columns = [saltmarshes_area_col, seagrasses_area_col, mangroves_area_col]
bce_df = compute_rates(bce_areas_df, json_path, bce_columns)
bce_df = correct_kiribati(bce_df)
bce_df.to_csv('data_source/summary/bce_data.csv', index=False)

# ==========================
# COMPUTE BCW
# ==========================
# GSCC value
gscc_value = gscc_computer(gscc_path)

# Carbon to CO2 convertion
cmol = 44 / 12

# BCW computation
data = bcw_computer(bce_df, cmol, gscc_value, bcp_path)
pcap_cols = ['Area_EEZ_KM2', 'GDP', 'CO2_emissions_2023', 'Debt (2015 US$)', 'Total BCW']
data = per_capita(data, pcap_cols, 'Population')

print("========================")
print("SUMMARY")
print("========================")
print(f"Global Social Cost of Carbon (GSCC): {gscc_value:.2f} US$/tCO2")
print(f"number of countries/territories: {data.shape[0]}")
print(f"Global BCW : {(data['Total BCW'].sum() / 1e12):.3f} trillion US$")

data.to_csv('country_level_bcw.csv', index=False)
print('\nData saved to country_level_bcw.csv')