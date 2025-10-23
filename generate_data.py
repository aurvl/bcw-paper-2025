from utils.bce_areas import generate_bce_data, compute_rates
from utils.compute_gscc import gscc_computer
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

# ==========================
# GENERATE DATA
# ==========================
# BCEs areas by EEZs
bce_areas_df = generate_bce_data(eez_path,
                                 saltmarshes_path, saltmarshes_area_col,
                                 seagrasses_path, seagrasses_area_col, 
                                 mangroves_path, mangroves_area_col, select)
print(bce_areas_df.head())

# Compute BCEs sequestration rates
bce_columns = [saltmarshes_area_col, seagrasses_area_col, mangroves_area_col]
bce_areas_df = compute_rates(bce_areas_df, json_path, bce_columns)

# GSCC value
gscc_value = gscc_computer(gscc_path)
print(f"Global Social Cost of Carbon (GSCC): {gscc_value}")