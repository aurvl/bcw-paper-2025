from __future__ import annotations

# Computing Blue Carbon Weath functions
import numpy as np
import pandas as pd

from src.utils import audit_missingness, nansum_min_count


# def _median_and_se(mat_NB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """Return (median, SE) along axis=1 for a (N,B) matrix (NaN-aware)."""
#     return median_and_se_nanaware(mat_NB)


def _mean_and_se(mat_NB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, SE) along axis=1 for a (N,B) matrix (NaN-aware)."""

    x = np.asarray(mat_NB)
    if x.ndim != 2:
        raise ValueError("Expected a (N,B) draws matrix")

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(x, axis=1)

    finite = np.isfinite(x)
    n = np.sum(finite, axis=1)
    diff = x - mean[:, None]
    diff = np.where(finite, diff, 0.0)
    sumsq = np.sum(diff * diff, axis=1)

    var = np.where(n > 1, sumsq / (n - 1), 0.0)
    std = np.sqrt(var)
    se = np.full_like(std, np.nan, dtype=float)
    mask = n > 0
    se[mask] = std[mask] / np.sqrt(n[mask])
    return mean, se

def cbcw_calculator(df: pd.DataFrame, cmol: float, gscc: float) -> pd.DataFrame:
    """
    This function use data on BCEs and compute the Coastal BCW
    """
    df['total_sequestration'] = df['tot_uptake (tC)'] * cmol

    df['cBCW'] = df['total_sequestration'] * gscc
    df = df.drop(columns='total_sequestration')
    return df

def add_bcp_point(df: pd.DataFrame, bcp_path: str) -> pd.DataFrame:
    """Add Blue Carbon Pump point estimate (tC/year) by country."""
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
    return data


def bcp_inclusion(df: pd.DataFrame, bcp_path: str, cmol: float, gscc: float) -> pd.DataFrame:
    """Backward-compatible wrapper (point-value oBCW)."""
    data = add_bcp_point(df, bcp_path)
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


def bcw_computer(
    df: pd.DataFrame,
    *,
    bce_tC_draws: np.ndarray,
    gscc_draws: np.ndarray,
    cmol: float,
    bcp_path: str,
    debug_audit: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute BCW with uncertainty propagation.

    For each draw b:
      BCW_b = GSCC_b * (U_BCE_b + U_BCP_point) * 44/12

    The output DataFrame reports median and SE (not std) for cBCW, oBCW and Total BCW.
    """

    if "country_name" not in df.columns:
        raise ValueError("Expected a 'country_name' column in df")

    bce = np.asarray(bce_tC_draws)
    if bce.ndim != 2:
        raise ValueError("bce_tC_draws must be a (N,B) array")
    if bce.shape[0] != len(df):
        raise ValueError(f"bce_tC_draws first dim must match df rows: {bce.shape[0]} != {len(df)}")

    g = np.asarray(gscc_draws)
    if g.ndim != 1:
        raise ValueError("gscc_draws must be a 1D array")
    if g.size != bce.shape[1]:
        raise ValueError(f"gscc_draws length must match B in bce_tC_draws: {g.size} != {bce.shape[1]}")

    out = add_bcp_point(df, bcp_path)
    bcp_point = pd.to_numeric(out.get("BCP Seq (tC)"), errors="coerce").to_numpy(dtype=bce.dtype)

    # Keep computations in the BCE dtype to avoid float64 upcasting for large B.
    g = g.astype(bce.dtype, copy=False)
    cmol = np.asarray(cmol, dtype=bce.dtype)

    if debug_audit:
        audit_missingness(out, ["BCP Seq (tC)"], "bcw_computer:after_bcp_merge")

    # Draw matrices
    cbcw_draws = (bce * cmol) * g[None, :]
    obcw_draws = (bcp_point[:, None] * cmol) * g[None, :]
    # IMPORTANT missingness semantics:
    # - If BCE is missing but BCP is present => total equals BCP component
    # - If BCP is missing but BCE is present => total equals BCE component
    # - If both are missing => total is NaN
    # i.e. NaN + x = x
    bcp_mat = np.broadcast_to(bcp_point[:, None], bce.shape)
    total_bcseq_draws = nansum_min_count(
        np.stack([bce, bcp_mat], axis=0),
        axis=0,
        min_count=1,
    )
    total_bcw_draws = (total_bcseq_draws * cmol) * g[None, :]

    # Summaries (median + SE)
    out["cBCW"], out["cBCW_se"] = _mean_and_se(cbcw_draws)
    out["oBCW"], out["oBCW_se"] = _mean_and_se(obcw_draws)
    out["Total BCseq"], out["Total BCseq_se"] = _mean_and_se(total_bcseq_draws)
    out["Total BCW"], out["Total BCW_se"] = _mean_and_se(total_bcw_draws)

    if debug_audit:
        audit_missingness(
            out,
            ["cBCW", "cBCW_se", "oBCW", "oBCW_se", "Total BCseq", "Total BCseq_se", "Total BCW", "Total BCW_se"],
            "bcw_computer:after_compute",
        )

    out = out.rename(columns={"ISO_TER1": "ISO"})
    # out.to_csv('data_source/summary/bcw_data_before_grouping.csv', index=False)

    def _group_claims_row_only(
        df_in: pd.DataFrame,
        *,
        mask: np.ndarray,
        new_name: str,
        key_column: str,
        idx: np.ndarray,
    ) -> dict[str, object]:
        """Build a single grouped row dict without materializing new (N,B) matrices."""

        df_to_group = df_in.iloc[idx]

        draw_cols = set()
        for base in ["cBCW", "oBCW", "Total BCW", "Total BCseq"]:
            draw_cols.add(base)
            draw_cols.add(f"{base}_se")

        def sum_if_any(s: pd.Series):
            return s.sum(skipna=True) if s.notna().any() else np.nan

        numeric_sum = (
            df_to_group.select_dtypes(include='number')
            .drop(columns=[c for c in draw_cols if c in df_to_group.columns], errors="ignore")
            .apply(sum_if_any)
        )

        object_cols = df_to_group.select_dtypes(include='object').columns
        object_values = {col: np.nan for col in object_cols}
        object_values[key_column] = new_name

        row: dict[str, object] = {**object_values, **numeric_sum.to_dict()}

        grouped_cbcw = nansum_min_count(cbcw_draws[idx], axis=0, min_count=1)
        grouped_obcw = nansum_min_count(obcw_draws[idx], axis=0, min_count=1)
        grouped_seq = nansum_min_count(total_bcseq_draws[idx], axis=0, min_count=1)
        grouped_bcw = nansum_min_count(total_bcw_draws[idx], axis=0, min_count=1)

        c_med, c_se = _mean_and_se(grouped_cbcw[None, :])
        o_med, o_se = _mean_and_se(grouped_obcw[None, :])
        s_med, s_se = _mean_and_se(grouped_seq[None, :])
        t_med, t_se = _mean_and_se(grouped_bcw[None, :])

        row["cBCW"], row["cBCW_se"] = float(c_med[0]), float(c_se[0])
        row["oBCW"], row["oBCW_se"] = float(o_med[0]), float(o_se[0])
        row["Total BCseq"], row["Total BCseq_se"] = float(s_med[0]), float(s_se[0])
        row["Total BCW"], row["Total BCW_se"] = float(t_med[0]), float(t_se[0])

        return row

    key_column = "country_name"
    mask_overlap = out[key_column].str.contains('Overlapping claim', na=False).to_numpy()
    mask_joint = out[key_column].str.contains('Joint regime area', na=False).to_numpy()

    grouped_rows: list[pd.DataFrame] = []
    grouped_bcw_draws: list[np.ndarray] = []
    if mask_overlap.any():
        idx = np.flatnonzero(mask_overlap)
        grouped_rows.append(pd.DataFrame([_group_claims_row_only(out, mask=mask_overlap, new_name='Overlapping Claims', key_column=key_column, idx=idx)], columns=out.columns))
        grouped_bcw_draws.append(nansum_min_count(total_bcw_draws[idx], axis=0, min_count=1))
    if mask_joint.any():
        idx = np.flatnonzero(mask_joint)
        grouped_rows.append(pd.DataFrame([_group_claims_row_only(out, mask=mask_joint, new_name='Joint Regimes', key_column=key_column, idx=idx)], columns=out.columns))
        grouped_bcw_draws.append(nansum_min_count(total_bcw_draws[idx], axis=0, min_count=1))

    if grouped_rows:
        keep = ~(mask_overlap | mask_joint)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            )
            out = pd.concat([out.iloc[np.flatnonzero(keep)], *grouped_rows], ignore_index=True)

            # Align the returned draws matrix to the grouped output rows:
            # - kept rows first, then grouped rows in the same order as `grouped_rows`.
            kept_draws = total_bcw_draws[np.flatnonzero(keep)]
            total_bcw_draws = np.vstack([kept_draws, *[d[None, :] for d in grouped_bcw_draws]])

    return out, total_bcw_draws