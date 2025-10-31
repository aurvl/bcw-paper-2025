import pandas as pd

def correct_kiribati(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function corrects the data for Kiribati by summing the values
    of its different island groups into a single entry for Kiribati.
    """
    
    df = df.copy()
    df = df.rename(columns={'UNION': 'country_name'})

    islands = ["Line Group", "Gilbert Islands", "Phoenix Group"]
    kiribati_rows = df[df['country_name'].isin(islands)]

    def sum_if_any(series):
        if series.notna().any():
            return series.sum(skipna=True)
        return pd.NA

    agg_map = {}
    for col in df.columns:
        if col == 'country_name':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = sum_if_any
        else:
            agg_map[col] = lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA

    kiribati_agg = kiribati_rows.groupby(lambda _: "Kiribati").agg(agg_map).reset_index()
    kiribati_agg = kiribati_agg.rename(columns={"index": "country_name"})

    df = df[~df['country_name'].isin(islands)].copy()
    kiribati_agg = kiribati_agg.dropna(axis=1, how='all')
    df = pd.concat([df, kiribati_agg], ignore_index=True)

    return df.sort_values(by='country_name')

def per_capita(df: pd.DataFrame, columns: list, pop_columns: str) -> pd.DataFrame:
    """
    This function computes per capita values for specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    columns : list
        List of column names for which to compute per capita values.
    pop_columns : str
        Name of the population column.
    Returns
    -------
    pandas.DataFrame
        DataFrame with new per capita columns added.
    """
    df = df.copy()
    for col in columns:
        per_capita_col = f"{col}_per_capita"
        df[per_capita_col] = df[col] / df[pop_columns]
    return df

def ratio_computer(df: pd.DataFrame, numerator: str, denominator: str, new_column: str) -> pd.DataFrame:
    """
    This function computes the ratio of two specified columns and adds it as a new column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    numerator : str
        Name of the numerator column.
    denominator : str
        Name of the denominator column.
    new_column : str
        Name of the new column to store the ratio.
    Returns
    -------
    pandas.DataFrame
        DataFrame with the new ratio column added.
    """
    df = df.copy()
    df[new_column] = df[numerator] / df[denominator]
    return df