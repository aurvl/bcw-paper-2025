# functions used to create the dataframe containing BCEs (Mangroves, Saltmarshes, Seagrasses) areas by EEZs
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.utils import safe_mul, safe_sum, median_and_se_nanaware
from src.config import (
    EEZ_PATH, SALTMARSHES_PATH, SEAGRASSES_PATH, MANGROVES_PATH,
    SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL, MANGROVES_AREA_COL,
    SELECT_COLS,
)


def import_data(path: str, select: List[str] = None) -> pd.DataFrame:
    """
    This function imports data from a .csv/.xlsx file located at the given path
    """
    if str(path).endswith('.csv'):
        df = pd.read_csv(path, usecols=select)
    elif str(path).endswith('.xlsx'):
        df = pd.read_excel(path, usecols=select)
    else:
        raise ValueError(f"Unknown file type: {path}")

    # concat 4 first cols
    parts = [df.iloc[:, i].fillna('').astype(str) for i in range(4)] # handle missing values safely
    df['concat_identifiers'] = parts[0] + parts[1] + parts[2] + parts[3]
    return df

def group_data(
    eez: pd.DataFrame, df1: pd.DataFrame, df1_area_col: str, 
    df2: pd.DataFrame, df2_are_col: str, df3: pd.DataFrame, 
    df3_areal_col: str
) -> pd.DataFrame:
    """
    This function groups data from multiple DataFrames based on a common identifier.
    - eez : Exclusive Economic Zones (EEZs) dataFrame
    - df1, df2, df3 : DataFrames containing BCE areas
    - df1_area_col, df2_are_col, df3_areal_col : Column names for area values in respective DataFrames
    """
    # Group by the concatenated identifier and sum the areas.
    # IMPORTANT: use min_count=1 so groups with all-NaN remain NaN (not 0).
    g1 = df1[['concat_identifiers', df1_area_col]].copy()
    g1[df1_area_col] = pd.to_numeric(g1[df1_area_col], errors='coerce')
    g1 = g1.groupby('concat_identifiers')[df1_area_col].sum(min_count=1).reset_index()

    g2 = df2[['concat_identifiers', df2_are_col]].copy()
    g2[df2_are_col] = pd.to_numeric(g2[df2_are_col], errors='coerce')
    g2 = g2.groupby('concat_identifiers')[df2_are_col].sum(min_count=1).reset_index()

    g3 = df3[['concat_identifiers', df3_areal_col]].copy()
    g3[df3_areal_col] = pd.to_numeric(g3[df3_areal_col], errors='coerce')
    g3 = g3.groupby('concat_identifiers')[df3_areal_col].sum(min_count=1).reset_index()

    df = eez.merge(g1, on='concat_identifiers', how='left')
    df = df.merge(g2, on='concat_identifiers', how='left')
    df = df.merge(g3, on='concat_identifiers', how='left')
    return df


def _uptake_draws_from_areas_and_rates(
    areas_N3: np.ndarray,
    rates_3B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (uptakes_eco, uptakes_total) with NaN-aware missingness semantics."""

    areas = np.asarray(areas_N3)
    rates = np.asarray(rates_3B)
    if areas.ndim != 2 or areas.shape[1] != 3:
        raise ValueError("areas_N3 must be (N,3)")
    if rates.ndim != 2 or rates.shape[0] != 3:
        raise ValueError("rates_3B must be (3,B)")

    uptakes_eco = safe_mul(areas[:, :, None], rates[None, :, :])  # (N,3,B)
    uptakes_total = safe_sum(uptakes_eco, axis=1, min_count=1)  # (N,B)
    return uptakes_eco, uptakes_total

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

def generate_bce_data(
    eez_path: str = EEZ_PATH, mangroves_path: str = MANGROVES_PATH, mangroves_area_col: str = MANGROVES_AREA_COL,
    saltmarshes_path: str = SALTMARSHES_PATH, saltmarshes_area_col: str = SALTMARSHES_AREA_COL,
    seagrasses_path: str = SEAGRASSES_PATH, seagrasses_area_col: str = SEAGRASSES_AREA_COL, 
    select: List[str] = SELECT_COLS
) -> pd.DataFrame:
    """
    This function generates a DataFrame of BCE areas by country EEZ.
        - eez_path             : Path to the EEZ data file
        - mangroves_path       : Path to the Mangroves data file
        - mangroves_area_col   : Column name for Mangroves area
        - saltmarshes_path     : Path to the Saltmarshes data file
        - saltmarshes_area_col : Column name for Saltmarshes area
        - seagrasses_path      : Path to the Seagrasses data file
        - seagrasses_area_col  : Column name for Seagrasses area
        - select               : List of country territorial identifiers column names
    """
    # importing the data
    eez = import_data(eez_path, select + ['a'])
    mangroves = import_data(mangroves_path, select + [mangroves_area_col])
    saltmarshes = import_data(saltmarshes_path, select + [saltmarshes_area_col])
    seagrasses = import_data(seagrasses_path, select + [seagrasses_area_col])
    
    # merging the data
    bce_df = group_data(
        eez, mangroves, mangroves_area_col,
        saltmarshes, saltmarshes_area_col,
        seagrasses, seagrasses_area_col
    )
    
    # adjusting the data
    bce_df = adjust_data(bce_df, [mangroves_area_col, saltmarshes_area_col, seagrasses_area_col])
    
    bce_df = bce_df.drop(columns=['concat_identifiers']).rename(columns={'a':'Area_EEZ_KM2'}).sort_values(by='UNION').reset_index(drop=True)

    return bce_df

def load_sequestration_json(json_path: str) -> Dict[str, Any]:
    """
    Open a JSON file containing normalized sequestration rates
    and return them as a Python dictionary.

    Parameters
    ----------
    json_path : str
        Path of the JSON file.

    Return
    ------
    Dict[str, Any]
        Dictionary containing sequestration rates by BCEs.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding='utf-8'))
    
    flat_dict = {}
    for eco, values in data.items():
        eco_key = eco.lower()
        flat_dict[f"{eco_key}_mean_sr"] = values.get("mean_rate")
        flat_dict[f"{eco_key}_median_sr"] = values.get("median_rate")
        flat_dict[f"{eco_key}_sr_se"] = values.get("se")
    return flat_dict


def lognormal_from_mean_median(mean: float, median: float):
    """
    Compute the parameters of a lognormal distribution from its mean and median.
    """
    if mean <= 0 or median <= 0:
        raise ValueError("Values must be positive")

    mu = np.log(median)
    sigma = np.sqrt(2 * (np.log(mean) - np.log(median)))
    return mu, sigma

def bootstrap_bce_rates(json_path: str, B: int, seed: int) -> Dict[str, np.ndarray]:
    """
    Generate bootstrap distributions of BCE sequestration rates based on the parameters of lognormal distributions.
    """
    rng = np.random.default_rng(int(seed))
    rates = load_sequestration_json(json_path)
    
    output: Dict[str, np.ndarray] = {}
    for eco in ['mangroves', 'saltmarshes', 'seagrasses']:
        mean = rates.get(f"{eco}_mean_sr")
        median = rates.get(f"{eco}_median_sr")
        if mean is None or median is None:
            raise ValueError(f"Missing mean/median for {eco} in {json_path}")
        mu, sigma = lognormal_from_mean_median(mean, median)
        output[eco] = rng.lognormal(mean=mu, sigma=sigma, size=int(B))
    return output

def compute_uptakes(
    df: pd.DataFrame,
    json_path: str,
    B: int,
    bce_columns: List[str],
    seed: int,
    *,
    quantiles: tuple[float, ...] | None = (0.05, 0.50, 0.95),
    keep_draws: bool = False,
    dtype=np.float32,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute uptake distributions per EEZ/country by combining:
      - BCE areas (km2) in df
      - BCE sequestration rate distributions (B draws) from json_path
      
    Parameters:
        - df: DataFrame containing BCE areas by EEZs, with columns specified in bce_columns
        - json_path: Path to JSON file containing parameters for BCE sequestration rates
        - B: Number of bootstrap draws to generate for each BCE sequestration rate
        - bce_columns: List of column names in df corresponding to BCE areas, in order [saltmarshes_col, seagrasses_col, mangroves_col]
        - seed: RNG seed for reproducibility
        - quantiles: Tuple of quantiles to compute for the uptake distributions (default: 5th, 50th, and 95th percentiles)
        - keep_draws: Whether to return the full draws matrices in addition to the summary DataFrame (default: False)
        - dtype: Data type for the computed uptake values (default: np.float32)

    Returns:
      - df_out with summary stats (mean/sd and requested quantiles) per BCE + total
      - optionally, a dict of full draws matrices (N x B) if keep_draws=True
    """
    if len(bce_columns) != 3:
        raise ValueError("bce_columns must be [saltmarshes_col, seagrasses_col, mangroves_col]")

    saltmarshes_col, seagrasses_col, mangroves_col = bce_columns

    # 1) sample rate distributions (each is shape (B,))
    rates_dist = bootstrap_bce_rates(json_path, B=B, seed=seed)

    # order MUST match bce_columns: saltmarshes, seagrasses, mangroves
    rates_mat = np.vstack(
        [
            np.asarray(rates_dist["saltmarshes"], dtype=dtype),
            np.asarray(rates_dist["seagrasses"], dtype=dtype),
            np.asarray(rates_dist["mangroves"], dtype=dtype),
        ]
    )  # (3, B)

    # 2) areas matrix (N, 3)
    # NOTE: we deliberately keep NaNs so missing area -> missing uptake (NaN), not 0.
    areas = df[[saltmarshes_col, seagrasses_col, mangroves_col]].to_numpy(dtype=dtype)

    # 3) per-draw uptakes (NaN-aware): (N, 3, B) and total (N, B)
    uptakes_eco, uptakes_total = _uptake_draws_from_areas_and_rates(areas, rates_mat)

    def _summarize(mat_NB: np.ndarray, prefix: str) -> pd.DataFrame:
        # mat_NB: (N, B)
        # Point-estimate policy: use the mean (not the median).
        # SE is computed from the draw variability as std/sqrt(n_finite).
        _, se = median_and_se_nanaware(mat_NB)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            mean = np.nanmean(mat_NB, axis=1)
        out = {f"{prefix}_mean": mean, f"{prefix}_se": se}
        if quantiles is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                qs = np.nanquantile(mat_NB, q=np.array(quantiles, dtype=float), axis=1)
            for k, q in enumerate(quantiles):
                label = f"p{int(round(q * 100)):02d}"
                out[f"{prefix}_{label}"] = qs[k, :]

        return pd.DataFrame(out, index=df.index)

    df_out = df.copy()

    # Partial-total flag: some ecosystems missing but not all.
    area_is_nan = np.isnan(areas)
    df_out["uptake_total_partial"] = area_is_nan.any(axis=1) & (~area_is_nan).any(axis=1)

    # summaries per ecosystem (N,B) slices
    salt_NB = uptakes_eco[:, 0, :]
    seag_NB = uptakes_eco[:, 1, :]
    mang_NB = uptakes_eco[:, 2, :]

    df_out = pd.concat(
        [
            df_out,
            _summarize(salt_NB, "uptake_salt"),
            _summarize(seag_NB, "uptake_seag"),
            _summarize(mang_NB, "uptake_mang"),
            _summarize(uptakes_total, "uptake_total"),
        ],
        axis=1,
    )

    if not keep_draws:
        return df_out

    draws = {
        "saltmarshes": salt_NB,
        "seagrasses": seag_NB,
        "mangroves": mang_NB,
        "total": uptakes_total,
        "rates": rates_mat,  # (3,B) if you want to reuse draws later
    }
    return df_out, draws


if __name__ == "__main__":
    # Example usage
    from src.config import (
        EEZ_PATH, SALTMARSHES_PATH, SEAGRASSES_PATH, MANGROVES_PATH, 
        SELECT_COLS, SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL,
        MANGROVES_AREA_COL, JSON_PATH, SEED_LIGHT, B_LIGHT
    )
    
    bce_df = generate_bce_data(
        eez_path=EEZ_PATH,
        mangroves_path=MANGROVES_PATH, mangroves_area_col=MANGROVES_AREA_COL,
        saltmarshes_path=SALTMARSHES_PATH, saltmarshes_area_col=SALTMARSHES_AREA_COL,
        seagrasses_path=SEAGRASSES_PATH, seagrasses_area_col=SEAGRASSES_AREA_COL,
        select=SELECT_COLS,
    )
    
    # Compute BCEs sequestration rates
    bce_columns = [SALTMARSHES_AREA_COL, SEAGRASSES_AREA_COL, MANGROVES_AREA_COL]
    bce_df, draws = compute_uptakes(
        bce_df, JSON_PATH, B=B_LIGHT, 
        bce_columns=bce_columns, seed=SEED_LIGHT,
        keep_draws=True, quantiles=None
    )
    print(bce_df.head())
    print(bce_df.shape)
    print(bce_df.columns)
    
    print(draws.keys())
    print('Rates draws shape:', draws['rates'].shape)
    print('Total uptake draws shape:', draws['total'].shape)
    print('Example rates draws (first 5):', draws['rates'][:, :5])
    print('Seagrasses draws shape:', draws['seagrasses'].shape)