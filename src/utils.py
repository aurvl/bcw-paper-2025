from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    else:
        return obj
    
def audit_missingness(
    df: pd.DataFrame,
    cols: list[str],
    name: str,
    *,
    pairs: list[tuple[str, str]] | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Print missingness/zero diagnostics for selected columns.

    Returns a small summary DataFrame with counts per column.

    Parameters
    ----------
    df:
        DataFrame to audit.
    cols:
        Columns to include in the per-column summary.
    name:
        Label to print (pipeline stage).
    pairs:
        Optional list of (area_col, value_col) to flag suspicious rows like
        area is NaN but value is 0.
    n:
        Number of example rows to print for each suspicious pattern.
    """

    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[audit_missingness:{name}] WARNING missing columns: {missing}")

    summary_rows: list[dict[str, object]] = []
    for c in present:
        s = pd.to_numeric(df[c], errors="coerce")
        is_na = s.isna()
        is_zero = (s == 0) & ~is_na
        is_pos = (s > 0) & ~is_na
        is_neg = (s < 0) & ~is_na
        summary_rows.append(
            {
                "col": c,
                "n": int(len(s)),
                "na": int(is_na.sum()),
                "zero": int(is_zero.sum()),
                "pos": int(is_pos.sum()),
                "neg": int(is_neg.sum()),
            }
        )

    out = pd.DataFrame(summary_rows)
    print("=" * 60)
    print(f"AUDIT: {name}")
    if not out.empty:
        print(out.to_string(index=False))
    else:
        print("(no columns to audit)")

    if pairs:
        for area_col, val_col in pairs:
            if area_col not in df.columns or val_col not in df.columns:
                continue

            area = pd.to_numeric(df[area_col], errors="coerce")
            val = pd.to_numeric(df[val_col], errors="coerce")

            mask_a_na_v_zero = area.isna() & (val == 0)
            if mask_a_na_v_zero.any():
                print("-" * 60)
                print(f"Suspicious: {area_col} is NaN but {val_col} == 0")
                show_cols = [c for c in ["country_name", "UNION", "ISO_TER1", area_col, val_col] if c in df.columns]
                print(df.loc[mask_a_na_v_zero, show_cols].head(int(n)).to_string(index=False))

            mask_a_na_v_notna = area.isna() & val.notna()
            if mask_a_na_v_notna.any():
                print("-" * 60)
                print(f"Check: {area_col} is NaN but {val_col} is not NaN")
                show_cols = [c for c in ["country_name", "UNION", "ISO_TER1", area_col, val_col] if c in df.columns]
                print(df.loc[mask_a_na_v_notna, show_cols].head(int(n)).to_string(index=False))

    print("=" * 60)
    return out


def nansum_min_count(x: np.ndarray, *, axis: int, min_count: int = 1) -> np.ndarray:
    """NumPy equivalent of pandas sum(min_count=...).

    - Sums finite values, ignoring NaN.
    - If the number of finite values along `axis` is < min_count, returns NaN.
    """
    arr = np.asarray(x)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axis)
    summed = np.nansum(arr, axis=axis)
    return np.where(count >= int(min_count), summed, np.nan)


def safe_mul(area: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Multiply with BCE missingness semantics.

    Rules (vectorized):
    - If area is NaN -> NaN
    - Else if area == 0 -> 0 (even if rate is NaN)
    - Else if rate is NaN -> NaN
    - Else -> area * rate
    """

    a = np.asarray(area)
    r = np.asarray(rate)
    out = a * r
    out = np.where(np.isfinite(a) & (a == 0.0), 0.0, out)
    return out


def safe_sum(values: np.ndarray, *, axis: int, min_count: int = 1) -> np.ndarray:
    """Sum finite values with 'all-NaN -> NaN' behavior."""

    return nansum_min_count(values, axis=axis, min_count=min_count)


def median_and_se_nanaware(mat_NB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, SE) along axis=1 for a (N,B) matrix, ignoring NaNs.

    SE is computed per row using only finite draws.
    - If a row has 0 finite draws: median and SE are NaN.
    - If a row has 1 finite draw: SE is 0.
    """

    x = np.asarray(mat_NB)
    if x.ndim != 2:
        raise ValueError("Expected a (N,B) draws matrix")

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        median = np.nanmedian(x, axis=1)
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
    return median, se

def correct_kiribati(
    df: pd.DataFrame,
    draws: dict[str, object] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    This function corrects the data for Kiribati by summing the values
    of its different island groups into a single entry for Kiribati.
    """
    
    df = df.copy()
    df = df.rename(columns={"UNION": "country_name"})

    islands = ["Line Group", "Gilbert Islands", "Phoenix Group"]
    kiribati_mask = df["country_name"].isin(islands)
    kiribati_rows = df[kiribati_mask]

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

    df_rest = df[~kiribati_mask].copy()
    kiribati_agg = kiribati_agg.dropna(axis=1, how="all")
    df2 = pd.concat([df_rest, kiribati_agg], ignore_index=True)

    sort_idx = df2["country_name"].astype(str).sort_values(kind="mergesort").index.to_numpy()
    df2 = df2.iloc[sort_idx].reset_index(drop=True)

    if draws is None:
        return df2

    # Align/aggregate (N,B) draws in the exact same way.
    draws_out: dict[str, object] = dict(draws)
    for k in ["saltmarshes", "seagrasses", "mangroves", "total"]:
        mat = draws.get(k)
        if mat is None:
            continue
        mat_arr = np.asarray(mat)
        if mat_arr.ndim != 2 or mat_arr.shape[0] != len(df):
            continue

        mat_rest = mat_arr[~kiribati_mask.to_numpy()]
        mat_k = mat_arr[kiribati_mask.to_numpy()]
        if mat_k.size == 0:
            mat2 = mat_rest
        else:
            grouped = nansum_min_count(mat_k, axis=0, min_count=1)
            mat2 = np.vstack([mat_rest, grouped[None, :]])

        draws_out[k] = mat2[sort_idx]

    return df2, draws_out

def per_capita(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    This function computes per capita values for specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    columns : list
        List of column names for which to compute per capita values.
    Returns
    -------
    pandas.DataFrame
        DataFrame with new per capita columns added.
    """
    pop_column = "Population"
    if pop_column not in df.columns:
        raise ValueError(f"Population column '{pop_column}' not found in DataFrame.")
    df = df.copy()
    for col in columns:
        per_capita_col = f"{col}_per_capita"
        df[per_capita_col] = df[col] / df[pop_column]
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


def _as_finite_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def approx_ks_distance(x: np.ndarray | list[float], y: np.ndarray | list[float], grid: int = 1000) -> float:
    """Approximate KS distance without SciPy.

    Computes ECDFs on a grid of quantiles of the pooled sample.
    """
    xs = np.sort(_as_finite_1d(x))
    ys = np.sort(_as_finite_1d(y))
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Empty draws for KS")

    pooled = np.concatenate([xs, ys])
    qs = np.linspace(0.0, 1.0, int(grid), endpoint=True)
    grid_x = np.quantile(pooled, qs)

    Fx = np.searchsorted(xs, grid_x, side="right") / float(xs.size)
    Fy = np.searchsorted(ys, grid_x, side="right") / float(ys.size)
    return float(np.max(np.abs(Fx - Fy)))


def overlap_iqr(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """IQR overlap ratio: intersection length / union length."""
    xs = _as_finite_1d(x)
    ys = _as_finite_1d(y)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Empty draws for overlap")

    x25, x75 = np.quantile(xs, [0.25, 0.75])
    y25, y75 = np.quantile(ys, [0.25, 0.75])

    inter = max(0.0, min(x75, y75) - max(x25, y25))
    union = max(x75, y75) - min(x25, y25)
    return float(inter / union) if union > 0 else float("nan")


def summarize_draws(x: np.ndarray | list[float]) -> dict:
    xs = _as_finite_1d(x)
    if xs.size == 0:
        raise ValueError("No finite draws")

    return {
        "B": int(xs.size),
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "std": float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0,
        "p05": float(np.quantile(xs, 0.05)),
        "p25": float(np.quantile(xs, 0.25)),
        "p75": float(np.quantile(xs, 0.75)),
        "p95": float(np.quantile(xs, 0.95)),
    }


def compare_two_distributions(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict:
    sx = summarize_draws(x)
    sy = summarize_draws(y)

    delta_median_pct = (sy["median"] - sx["median"]) / sx["median"] if sx["median"] != 0 else float("nan")
    delta_mean_pct = (sy["mean"] - sx["mean"]) / sx["mean"] if sx["mean"] != 0 else float("nan")

    ks = approx_ks_distance(x, y, grid=1000)
    overlap = overlap_iqr(x, y)

    return {
        "delta_median_pct": float(delta_median_pct),
        "delta_mean_pct": float(delta_mean_pct),
        "ks": float(ks),
        "overlap": float(overlap),
    }


def choose_scheme_uniform_vs_bertram(metrics: dict, thresholds: dict) -> str:
    med_pct_threshold = float(thresholds.get("med_pct_threshold", 0.05))
    ks_threshold = float(thresholds.get("ks_threshold", 0.05))
    overlap_threshold = float(thresholds.get("overlap_threshold", 0.8))

    if (
        abs(float(metrics["delta_median_pct"])) < med_pct_threshold
        and float(metrics["ks"]) < ks_threshold
        and float(metrics["overlap"]) > overlap_threshold
    ):
        return "uniform"
    return "bertram_proxy"


def choose_best_baseline(baseline_reports: dict, thresholds: dict) -> dict:
    """Choose baseline among bootstrap / estimates / mixed.

    Rules (simple & transparent):
    - bootstrap-only preferred a priori.
    - If estimates corroborates bootstrap (ks < 0.05 and |Δmedian| < 0.05), keep bootstrap and report corroboration.
    - If mixed differs a lot from bootstrap (|Δmedian| > 0.05 OR ks > 0.05), mark mixed non-homogeneous: avoid as main baseline.
    - If bootstrap and estimates diverge strongly and mixed overlaps well with both, recommend bootstrap as baseline and keep others as sensitivity.
    """
    close_ks = float(thresholds.get("baseline_close_ks", 0.05))
    close_med = float(thresholds.get("baseline_close_med_pct", 0.05))
    mixed_nonhom_ks = float(thresholds.get("mixed_nonhom_ks", 0.05))
    mixed_nonhom_med = float(thresholds.get("mixed_nonhom_med_pct", 0.05))
    between_overlap = float(thresholds.get("between_overlap", 0.8))

    needed = {"bootstrap", "estimates", "mixed"}
    missing = needed.difference(baseline_reports.keys())
    if missing:
        raise ValueError(f"Missing baseline reports: {sorted(missing)}")

    chosen = {k: baseline_reports[k]["chosen_draws"] for k in needed}

    pairwise = {}
    for a, b in combinations(["bootstrap", "estimates", "mixed"], 2):
        pairwise[(a, b)] = compare_two_distributions(chosen[a], chosen[b])

    m_be = pairwise[("bootstrap", "estimates")]
    m_bm = pairwise[("bootstrap", "mixed")]
    m_em = pairwise[("estimates", "mixed")]

    estimates_close = (m_be["ks"] < close_ks) and (abs(m_be["delta_median_pct"]) < close_med)
    mixed_nonhomogeneous = (m_bm["ks"] > mixed_nonhom_ks) or (abs(m_bm["delta_median_pct"]) > mixed_nonhom_med)

    boot_est_diverge = (m_be["ks"] > close_ks) or (abs(m_be["delta_median_pct"]) > close_med)
    mixed_between = (m_bm["overlap"] > between_overlap) and (m_em["overlap"] > between_overlap)

    selected_baseline = "bootstrap"

    if estimates_close:
        rationale = "Selected bootstrap baseline: estimates corroborates bootstrap (very similar distributions)."
    elif mixed_nonhomogeneous:
        rationale = "Selected bootstrap baseline (homogeneous). Mixed baseline is non-homogeneous vs bootstrap; keep mixed as sensitivity only."
    elif boot_est_diverge and mixed_between:
        rationale = "Selected bootstrap baseline by homogeneity prior. Bootstrap and estimates diverge; mixed sits between them (high overlap with both). Use estimates/mixed as sensitivity."
    elif boot_est_diverge:
        rationale = "Selected bootstrap baseline by homogeneity prior. Bootstrap and estimates differ materially; report estimates/mixed as sensitivity."
    else:
        rationale = "Selected bootstrap baseline by homogeneity prior; no strong evidence to prefer estimates or mixed as main baseline."

    metrics_rows = []
    for (a, b), m in pairwise.items():
        metrics_rows.append({"a": a, "b": b, **m})

    return {
        "selected_baseline": selected_baseline,
        "rationale": rationale,
        "metrics_table": metrics_rows,
        "flags": {
            "estimates_close_to_bootstrap": bool(estimates_close),
            "mixed_nonhomogeneous_vs_bootstrap": bool(mixed_nonhomogeneous),
            "bootstrap_estimates_diverge": bool(boot_est_diverge),
            "mixed_between": bool(mixed_between),
        },
    }
