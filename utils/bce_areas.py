# functions used to create the dataframe containing BCEs (Mangroves, Saltmarshes, Seagrasses) areas by EEZs
import pandas as pd
import json
from typing import List, Dict, Any

def import_data(path: str, select: List[str] = None) -> pd.DataFrame:
    """
    This function imports data from a .csv/.xlsx file located at the given path
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path, usecols=select)
    elif path.endswith('.xlsx'):
        df = pd.read_excel(path, usecols=select)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    # concat 4 first cols
    parts = [df.iloc[:, i].fillna('').astype(str) for i in range(4)] # handle missing values safely
    df['concat_identifiers'] = parts[0] + parts[1] + parts[2] + parts[3]
    return df

def group_data(eez: pd.DataFrame, df1: pd.DataFrame, df1_area_col: str, 
               df2: pd.DataFrame, df2_are_col: str, df3: pd.DataFrame, df3_areal_col: str) -> pd.DataFrame:
    """
    This function groups data from multiple DataFrames based on a common identifier.
    - eez : Exclusive Economic Zones (EEZs) dataFrame
    - df1, df2, df3 : DataFrames containing BCE areas
    - df1_area_col, df2_are_col, df3_areal_col : Column names for area values in respective DataFrames
    """
    # Group by the concatenated identifier and sum the areas
    df = eez.merge(df1[['concat_identifiers', df1_area_col]].groupby('concat_identifiers').sum().reset_index(),
                   on='concat_identifiers', how='left')
    df = df.merge(df2[['concat_identifiers', df2_are_col]].groupby('concat_identifiers').sum().reset_index(),
                  on='concat_identifiers', how='left')
    df = df.merge(df3[['concat_identifiers', df3_areal_col]].groupby('concat_identifiers').sum().reset_index(),
                  on='concat_identifiers', how='left')
    return df

def adjust_data(df: pd.DataFrame, area_cols: List[str]) -> pd.DataFrame:
    """
    For countries with incomplete or outdated data, such as the Bahamas and Mauritania, 
    we supplemented the dataset with recent literature. Seagrass meadow areas in the Bahamas 
    were updated based on Fu et al. (2023) and Gallagher et al. (2022), with a mean area of 
    79,757 km² ranging from 66,990 to 92,524 km². Saltmarsh and seagrass data for Mauritania 
    were refined using estimates from Pottier et al. (2021).
    """
    # Update Bahamas seagrass area
    bahamas_mask = (df['TERRITORY1'] == 'Bahamas')
    if 'seagrasses_area_km2' in area_cols:
        df.loc[bahamas_mask, 'seagrasses_area_km2'] = 79757.0

    # Update Mauritania saltmarsh and seagrass areas
    mauritania_mask = (df['ISO_TER1'] == 'MRT')
    if 'saltmarshes_area_km2' in area_cols:
        df.loc[mauritania_mask, 'saltmarshes_area_km2'] = 23.0
    if 'seagrasses_area_km2' in area_cols:
        df.loc[mauritania_mask, 'seagrasses_area_km2'] = 772.0

    return df

def generate_bce_data(eez_path: str, 
                      mangroves_path: str, mangroves_area_col: str,
                      saltmarshes_path: str, saltmarshes_area_col: str,
                      seagrasses_path: str, seagrasses_area_col: str, select: List[str]) -> pd.DataFrame:
    """
    This function generates a DataFrame containing BCE areas by EEZs.
    - eez_path : Path to the EEZ data file
    - mangroves_path : Path to the Mangroves data file
    - mangroves_area_col : Column name for Mangroves area
    - saltmarshes_path : Path to the Saltmarshes data file
    - saltmarshes_area_col : Column name for Saltmarshes area
    - seagrasses_path : Path to the Seagrasses data file
    - seagrasses_area_col : Column name for Seagrasses area
    """
    # importing the data
    eez = import_data(eez_path, select + ['a'])
    mangroves = import_data(mangroves_path, select + [mangroves_area_col])
    saltmarshes = import_data(saltmarshes_path, select + [saltmarshes_area_col])
    seagrasses = import_data(seagrasses_path, select + [seagrasses_area_col])
    
    # merging the data
    bce_df = group_data(eez, mangroves, mangroves_area_col,
                        saltmarshes, saltmarshes_area_col,
                        seagrasses, seagrasses_area_col)
    
    # adjusting the data
    bce_df = adjust_data(bce_df, [mangroves_area_col, saltmarshes_area_col, seagrasses_area_col])
    
    bce_df = bce_df.drop(columns=['concat_identifiers']).rename(columns={'a':'Area_EEZ_KM2'}).sort_values(by='UNION').reset_index(drop=True)

    return bce_df

def load_sequestration_json(json_path: str) -> Dict[str, Any]:
    """
    Open a JSON file containing normalized sequestration rates
    and return them as a Python dictionary.

    Paramètres
    ----------
    json_path : str
        Path of the JSON file.

    Return
    ------
    Dict[str, Any]
        Dictionary containing sequestration rates by BCEs.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_dict = {}
    for eco, values in data.items():
        eco_key = eco.lower()
        flat_dict[f"{eco_key}_sr"] = values.get("rate")
        flat_dict[f"{eco_key}_sr_low"] = values.get("ci_lower")
        flat_dict[f"{eco_key}_sr_high"] = values.get("ci_upper")
        flat_dict[f"{eco_key}_sr_se"] = values.get("se")

    return flat_dict
    
def compute_rates(df: pd.DataFrame, json_path: str, bce_columns: List[str]) -> pd.DataFrame:
    """
    Open a JSON file containing normalized sequestration rates
    and return them as a Python dictionary.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame containing data on BCEs areas in km2 by EEZs.
    json_path : str
        Path of the JSON file.
    bce_columns : List
        List of BCE area column names in the DataFrame, in this order : saltmarshes, seagrasses, mangroves

    Return
    ------
    Dataframe
        DataFrame containing sequestration rates by BCEs and EEZs.
    """
    # Loading sequestration rates
    rates = load_sequestration_json(json_path)
    mangroves_sr = rates['mangroves_sr']
    mangroves_sr_low = rates['mangroves_sr_low']
    mangroves_sr_high = rates['mangroves_sr_high']
    saltmarshes_sr = rates['saltmarshes_sr']
    saltmarshes_sr_low = rates['saltmarshes_sr_low']
    saltmarshes_sr_high = rates['saltmarshes_sr_high']
    seagrasses_sr = rates['seagrasses_sr']
    seagrasses_sr_high = rates['seagrasses_sr_high']
    seagrasses_sr_low = rates['seagrasses_sr_low']
    
    saltmarshes_col = bce_columns[0]
    seagrasses_col = bce_columns[1]
    mangroves_col = bce_columns[2]
    
    # compute :
    #     'Uptake Salt (t/km2)', 'Uptake Seag (t/km2)', 'Uptake Mang (t/km2)',
    #     'Uptake Salt LOW', 'Uptake Seag LOW', 'Uptake Mang LOW', 
    #     'Uptake Salt HIGH', 'Uptake Seag HIGH', 'Uptake Mang HIGH',
    #     'tot_uptake (tC)', 'tot_upt_LOW', 'tot_upt_HIGH'
    df['Uptake Salt (t/km2)'] = df[saltmarshes_col] * saltmarshes_sr
    df['Uptake Seag (t/km2)'] = df[seagrasses_col] * seagrasses_sr
    df['Uptake Mang (t/km2)'] = df[mangroves_col] * mangroves_sr
    df['Uptake Salt LOW'] = df[saltmarshes_col] * saltmarshes_sr_low
    df['Uptake Seag LOW'] = df[seagrasses_col] * seagrasses_sr_low
    df['Uptake Mang LOW'] = df[mangroves_col] * mangroves_sr_low
    df['Uptake Salt HIGH'] = df[saltmarshes_col] *  saltmarshes_sr_high
    df['Uptake Seag HIGH'] = df[seagrasses_col] * seagrasses_sr_high
    df['Uptake Mang HIGH'] = df[mangroves_col] * mangroves_sr_high
    
    # compute total uptake
    df['tot_uptake (tC)'] = df['Uptake Salt (t/km2)'] + df['Uptake Seag (t/km2)'] + df['Uptake Mang (t/km2)']
    df['tot_upt_LOW'] = df['Uptake Salt LOW'] + df['Uptake Seag LOW'] + df['Uptake Mang LOW']
    df['tot_upt_HIGH'] = df['Uptake Salt HIGH'] + df['Uptake Seag HIGH'] + df['Uptake Mang HIGH']
    
    return df