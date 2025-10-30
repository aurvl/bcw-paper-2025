# Computing Blue Carbon Weath functions
import pandas as pd
import numpy as np

def gscc_computer(path: str) -> float:
    """
    This function computes the global Social Cost of Carbon (GSCC) by reading country-level SCC data
    from a CSV file located at the given path.
    """
    df = pd.read_csv(path)
    df_filtered = df[(df['dmgfuncpar'] == 'bootstrap') & 
                     (df['climate'] == 'uncertain') & 
                     (df['prtp'].isna()) & 
                     (df['eta'].isna()) & 
                     (df['dr'] == 3.) & 
                     (df['run'].isin(['bhm_richpoor_lr']))]
    
    group = df_filtered.groupby('ISO3')[['16.7%', '50%', '83.3%']].median()
    group['country'] = group.index
    group = group[['country', '16.7%', '50%', '83.3%']]
    group = group.reset_index(drop=True)
    group.rename(columns={'50%':'median'}, inplace=True)
    
    cscc = group[group['country'] != 'WLD']
    gscc = cscc['median'].sum()
    group.to_csv('data_source/gscc/country_level_gscc.csv', index=False)
    return gscc
    
def cbcw_calculator(df: pd.DataFrame, cmol: float, gscc: float) -> pd.DataFrame:
    """
    This function use data on BCEs and compute the Coastal BCW
    """
    df['total_sequestration'] = df['tot_uptake (tC)'] * cmol

    df['cBCW'] = df['total_sequestration'] * gscc
    df = df.drop(columns='total_sequestration')
    return df

def bcp_inclusion(df: pd.DataFrame, bcp_path: str, cmol: float, gscc: float) -> pd.DataFrame:
    """This function add Blue carbon pump to the data"""
    bcp = pd.read_csv(bcp_path)
    bcp['BCP sequestration in EEZ (tC/year)'] = bcp['BCP sequestration in EEZ (GtC/year)'] * 1e9
    bcp = bcp[['Country', 'BCP sequestration in EEZ (tC/year)']]
    name_map = {
        "Antigua & B.": "Antigua and Barbuda",
        "Chagos Archip.": "Chagos Archipelago",
        "Dem. Rep. Congo": "Democratic Republic of the Congo",
        "Eq. Guinea": "Equatorial Guinea",
        "FS of Micronesia": "Micronesia",
        "Papua N. Guinea": "Papua New Guinea",
        "Rep. of Congo": "Republic of the Congo",
        "Sao Tome & P.": "Sao Tome and Principe",
        "St. Vincent & Gr.": "Saint Vincent and the Grenadines",
        "UK": "United Kingdom",
        "Mauritius": "Republic of Mauritius",
        "Somalia": "Federal Republic of Somalia"
    }
    bcp['Country'] = bcp['Country'].replace(name_map)
    data = df.merge(bcp, left_on='country_name', right_on='Country', how='left')
    data.rename(columns={'BCP sequestration in EEZ (tC/year)': 'BCP Seq (tC)'}, inplace=True)
    data.drop(columns=['Country'], inplace=True)
    data['oBCW'] = data['BCP Seq (tC)'] * cmol * gscc
    return data

def group_claims(df, pattern, new_name, key_column='Country'):
    """
    Groups rows containing a given pattern (e.g., 'Overlapping claim', 'Joint regime')
    into a single row with:
    - Sum of all numeric columns
    - NaN for object columns (except for key_column, which takes new_name)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    pattern : str
        Pattern to search for in the key_column (e.g., 'Overlapping claim').
    new_name : str
        Name to assign to the grouped row (e.g., 'Overlapping Claims').
    key_column : str, default='Country'
        Column in which to search for the pattern.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the grouped row added.
    """

    # 1. Identify the rows to group
    mask = df[key_column].str.contains(pattern, na=False)

    # 2. Split the DataFrame
    df_rest = df[~mask]
    df_to_group = df[mask]

    if df_to_group.empty:
        return df.copy()  # Nothing to group

    # 3. Sum numeric columns
    def sum_if_any(s):
        return s.sum(skipna=True) if s.notna().any() else np.nan

    numeric_sum = df_to_group.select_dtypes(include='number').apply(sum_if_any)

    # 4. Object columns: set to NaN except for key_column
    object_cols = df_to_group.select_dtypes(include='object').columns
    object_values = {col: np.nan for col in object_cols}
    object_values[key_column] = new_name

    # 5. Combine both parts
    combined_row = {**object_values, **numeric_sum.to_dict()}

    # 6. Return the final DataFrame
    df_final = pd.concat([df_rest, pd.DataFrame([combined_row])], ignore_index=True)

    return df_final

def bcw_computer(df: pd.DataFrame, cmol: float, gscc: float, bcp_path: str) -> pd.DataFrame:
    """
    This function compute the Blue Carbon Weath including Blue Carbon Pump
    """
    df = cbcw_calculator(df, cmol, gscc)
    df = bcp_inclusion(df, bcp_path, cmol, gscc)
    df['Total BCseq'] = df[['tot_uptake (tC)', 'BCP Seq (tC)']].sum(axis=1, min_count=1)
    df['Total BCW'] = df['Total BCW'] = df[['cBCW', 'oBCW']].sum(axis=1, min_count=1)
    df = group_claims(
        df, pattern='Overlapping claim', new_name='Overlapping Claims', key_column='country_name'
    )
    df = group_claims(
        df, pattern='Joint regime area', new_name='Joint Regimes', key_column='country_name'
    )
    return df