"""Global Social Cost of Carbon — structure-adjusted estimand.

Public API
----------
get_gscc_dist(...)
    Returns the **structure-adjusted GSCC distribution** (Option B) together
    with its raw counterpart, summary statistics, and decomposition diagnostics.

    The estimand is built in four steps:

    1.  Scenario table.  Filter the CSCC database by the requested
        dmgfuncpar / climate / run / SSP / RCP / discounting parameters.
        Aggregate by summing country-level median CSCCs per scenario cell.

    2.  Raw draws.  Resample B scenario rows with replacement under the
        chosen sampling scheme (``"uniform"`` or ``"weighted"``).

    3.  Ridge decomposition.  Fit a ridge-regularised additive model in
        log space on the *scenario table*::

            log(GSCC_s) ~ run + discount_id + climate + scenario_id + dmgfuncpar

        Select the penalty lambda_star that maximises the R² drop (raw minus
        adjusted), which is equivalent to maximising the share of variance
        explained by label-structured factors.

    4.  Centering and adjustment.  Compute the systematic score
        ``s = X @ beta`` on the draws, then **centre** it::

            s_centered = s - mean(s)

        The adjusted log is ``log_adj = log_raw - s_centered``, and the
        adjusted draw is ``gscc_adj = exp(log_adj)``.

        By construction, ``mean(log_raw) == mean(log_adj)`` (the mean in log
        space is preserved); the centering merely redistributes variance.

All monetary values are in 2015 USD / tCO₂ as in the CSCC source database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSCC_CSV = "data_source/gscc/cscc_db_v2.csv"

# Ordered factor labels used in the ridge decomposition.
DECOMP_FACTORS: list[str] = [
    "run",
    "discount_id",
    "climate",
    "scenario_id",
    "dmgfuncpar",
]

# Default lambda search grid for select_lambda_star.
LAMBDA_GRID_DEFAULT: list[float] = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0]

# Scenario key definition for Part 1.
# NOTE: we intentionally do NOT include column "N" here:
# "N" describes how many bootstrap replications were used upstream (a sampling
# configuration / metadata), not a scenario dimension, and can vary without
# changing the underlying socio-climate-discount scenario.
scenario_cols = ["run", "dmgfuncpar", "climate", "SSP", "RCP", "prtp", "eta", "dr"]


def load_cscc(path: str) -> pd.DataFrame:
    """Load and validate the CSCC database."""
    df = pd.read_csv(path)

    needed = {"run", "dmgfuncpar", "climate", "SSP", "RCP", "ISO3", "prtp", "eta", "dr", "50%"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSCC CSV: {sorted(missing)}")

    # Required coercions for stable filtering / grouping
    for col in ["ISO3", "run", "climate", "SSP", "RCP", "dmgfuncpar"]:
        df[col] = df[col].astype(str)

    return df


def compute_gscc_per_scenario(
    df_filtered: pd.DataFrame,
    *,
    value_col: str = "50%",
) -> pd.DataFrame:
    """Build scenario-level GSCC: for each scenario, sum country medians."""
    if value_col not in df_filtered.columns:
        raise ValueError(f"Missing value column: {value_col}")

    by_country = (
        df_filtered.groupby(scenario_cols + ["ISO3"], dropna=False)[value_col]
        .median()
        .reset_index()
    )
    scenario_sum = (
        by_country.groupby(scenario_cols, dropna=False)[value_col]
        .sum()
        .rename("gscc")
        .reset_index()
    )
    n_iso3 = (
        by_country.groupby(scenario_cols, dropna=False)["ISO3"]
        .nunique()
        .rename("n_iso3")
        .reset_index()
    )
    return scenario_sum.merge(n_iso3, on=scenario_cols, how="left", validate="one_to_one")


def bertram_proxy_weights(scenarios_df: pd.DataFrame, run_col: str = "run") -> np.ndarray:
    """Bertram proxy: equal total mass per `run` (weight per row = 1 / count(run))."""
    if scenarios_df.empty:
        raise ValueError("Empty scenarios_df")
    if run_col not in scenarios_df.columns:
        raise ValueError(f"Missing run column: {run_col}")

    run_sizes = scenarios_df.groupby(run_col, dropna=False)[run_col].transform("count").to_numpy(dtype=float)
    w = 1.0 / run_sizes
    w = w / float(np.sum(w))
    return w


def _as_list_str(x: str | list[str]) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def _filter_float_values(df: pd.DataFrame, col: str, values: list[float]) -> pd.DataFrame:
    if not values:
        return df
    s = pd.to_numeric(df[col], errors="coerce")
    mask = np.zeros(len(df), dtype=bool)
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        mask |= np.isclose(s.to_numpy(dtype=float), float(v), rtol=0.0, atol=1e-12)
    return df[mask]


def _fmt_val(v: Any) -> str:
    """Format a scalar for a discount_id key: 1.5 → '1p5'."""
    try:
        f = float(v)
        return f"{f:.1f}".replace(".", "p")
    except (TypeError, ValueError):
        return "na"


def infer_discounting_type(prtp: Any, eta: Any, dr: Any) -> str:
    """Classify a discounting regime as 'ramsey', 'fixed', or 'unknown'."""
    if pd.notna(prtp) and pd.notna(eta):
        return "ramsey"
    if pd.notna(dr):
        return "fixed"
    return "unknown"


def make_discount_id(discounting_type: str, prtp: Any, eta: Any, dr: Any) -> str:
    """Build a unique discount-parameter key string."""
    if discounting_type == "ramsey":
        prtp_s = _fmt_val(prtp) if pd.notna(prtp) else "nan"
        eta_s = str(eta).replace(".", "p").replace("-", "m") if pd.notna(eta) else "nan"
        return f"ramsey_prtp{prtp_s}_eta{eta_s}"
    if discounting_type == "fixed":
        dr_s = _fmt_val(dr) if pd.notna(dr) else "nan"
        return f"fixed_dr{dr_s}"
    return "unknown"


# ---------------------------------------------------------------------------
# Scenario-table enrichment
# ---------------------------------------------------------------------------

def _enrich_scenario_table(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """Add scenario_id, discount_id, log_gscc to a scenario-level table.

    Input must include scenario_cols + ['gscc']. Rows with non-positive
    GSCC are dropped.
    """
    df = scenarios_df.copy()

    ssp = (
        df["SSP"].astype(str).str.lower()
        if "SSP" in df.columns
        else pd.Series(["na"] * len(df), index=df.index)
    )
    rcp = (
        df["RCP"].astype(str).str.lower()
        if "RCP" in df.columns
        else pd.Series(["na"] * len(df), index=df.index)
    )
    df["scenario_id"] = ssp + "_" + rcp

    def _row_discount_id(row: pd.Series) -> str:
        dtype = infer_discounting_type(row.get("prtp"), row.get("eta"), row.get("dr"))
        return make_discount_id(dtype, row.get("prtp"), row.get("eta"), row.get("dr"))

    df["discount_id"] = df.apply(_row_discount_id, axis=1)

    df = df[df["gscc"] > 0].copy()
    df["log_gscc"] = np.log(df["gscc"].astype(float))

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ridge decomposition (inlined to avoid circular import with new_test_gscc)
# ---------------------------------------------------------------------------

def _build_design_matrix(
    df: pd.DataFrame,
    factors: list[str],
    design_info: dict | None = None,
) -> tuple[Any, dict]:
    """Sparse one-hot design matrix with drop-first encoding.

    Returns (X csr_matrix, info_dict).
    """
    from scipy import sparse  # type: ignore

    n = len(df)
    row_idx: list[int] = []
    col_idx: list[int] = []
    data_vals: list[float] = []

    baselines: dict[str, str] = {} if design_info is None else dict(design_info["baselines"])
    levels: dict[str, list[str]] = (
        {}
        if design_info is None
        else {k: list(v) for k, v in design_info["levels"].items()}
    )
    col_slices: dict[str, slice] = {}
    column_names: list[str] = []
    col_offset = 0

    for factor in factors:
        if factor not in df.columns:
            raise ValueError(f"Missing factor column: {factor}")
        s = df[factor]
        if s.isna().any():
            raise ValueError(f"Factor '{factor}' contains NaNs")

        if design_info is None:
            levs = sorted(pd.unique(s.astype(str)))
            levels[factor] = levs
            baselines[factor] = levs[0] if levs else ""
        else:
            levs = levels[factor]
            baselines.setdefault(factor, levs[0] if levs else "")

        if len(levs) <= 1:
            col_slices[factor] = slice(col_offset, col_offset)
            continue

        code = pd.Categorical(s.astype(str), categories=levs, ordered=True).codes
        nz = np.nonzero(code)[0]
        if nz.size:
            row_idx.extend(nz.tolist())
            col_idx.extend((col_offset + (code[nz] - 1)).tolist())
            data_vals.extend([1.0] * int(nz.size))

        start, end = col_offset, col_offset + (len(levs) - 1)
        col_slices[factor] = slice(start, end)
        column_names.extend([f"{factor}={lev}" for lev in levs[1:]])
        col_offset = end

    p = col_offset
    X = sparse.csr_matrix((data_vals, (row_idx, col_idx)), shape=(n, p), dtype=float)
    info = {
        "factors": list(factors),
        "baselines": baselines,
        "levels": levels,
        "col_slices": col_slices,
        "column_names": column_names,
    }
    return X, info


def _ridge_solve_intercept(y: np.ndarray, X: Any, lam: float) -> tuple[float, np.ndarray, dict]:
    """Ridge with unpenalised intercept via LSQR on augmented system."""
    from scipy import sparse  # type: ignore
    from scipy.sparse.linalg import lsqr  # type: ignore

    y = np.asarray(y, dtype=float)
    n, p = int(X.shape[0]), int(X.shape[1])
    if p == 0:
        return float(np.mean(y)), np.zeros(0, dtype=float), {}

    ones = sparse.csr_matrix(np.ones((n, 1), dtype=float))
    Z = sparse.hstack([ones, X], format="csr")
    pen = sparse.hstack(
        [sparse.csr_matrix((p, 1), dtype=float), sparse.identity(p, format="csr", dtype=float)],
        format="csr",
    )
    A = sparse.vstack([Z, np.sqrt(float(lam)) * pen], format="csr")
    b = np.concatenate([y, np.zeros(p, dtype=float)])
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=2000)
    w = np.asarray(sol[0], dtype=float)
    return float(w[0]), w[1:], {"istop": int(sol[1]), "itn": int(sol[2])}


def _fit_ridge(
    df: pd.DataFrame,
    factors: list[str],
    lam: float,
    target_col: str = "log_gscc",
    design_info: dict | None = None,
) -> dict:
    """Fit ridge decomposition; return result dict."""
    work = df.loc[:, [target_col] + list(factors)].dropna()
    y = work[target_col].astype(float).to_numpy()
    X, info = _build_design_matrix(work, factors, design_info=design_info)

    theta, beta, lsqr_info = _ridge_solve_intercept(y, X, lam)
    systematic_raw = (X @ beta).astype(float) if beta.size else np.zeros(len(y))
    systematic_centered = systematic_raw - float(np.mean(systematic_raw))
    fitted = theta + systematic_centered
    residual = y - fitted

    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - fitted) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else 0.0

    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    var_sys = float(np.var(systematic_centered, ddof=1)) if y.size > 1 else 0.0
    var_res = float(np.var(residual, ddof=1)) if y.size > 1 else 0.0

    effects_tables: dict[str, dict] = {}
    for factor in info["factors"]:
        levs = info["levels"].get(factor, [])
        baseline = info["baselines"].get(factor, "")
        sl = info["col_slices"].get(factor, slice(0, 0))
        coef_f = beta[sl] if sl.stop > sl.start else np.zeros(0, dtype=float)
        effects: dict[str, float] = {str(baseline): 0.0}
        for lev, c in zip(levs[1:], coef_f, strict=False):
            effects[str(lev)] = float(c)
        effects_tables[factor] = effects

    return {
        "target_col": target_col,
        "factors": list(factors),
        "lam": float(lam),
        "design_info": info,
        "theta_hat": float(theta),
        "beta_hat": beta,
        "systematic_raw": systematic_raw,
        "systematic_centered": systematic_centered,
        "residual": residual,
        "fitted": fitted,
        "effects_tables": effects_tables,
        "r2": float(r2),
        "var_y": var_y,
        "var_systematic": var_sys,
        "var_residual": var_res,
        "structured_var_share": float(var_sys / var_y) if var_y > 0 else 0.0,
        "residual_var_share": float(var_res / var_y) if var_y > 0 else 0.0,
        "lsqr_info": lsqr_info,
    }


def _apply_decomp(fit: dict, draws_df: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted ridge to draws_df; adds systematic_centered, log_gscc_adj, gscc_adj."""
    target_col = str(fit["target_col"])
    factors = list(fit["factors"])
    theta = float(fit["theta_hat"])
    beta = np.asarray(fit["beta_hat"], dtype=float)
    design_info = fit["design_info"]

    df = draws_df.copy()
    X, _ = _build_design_matrix(df, factors, design_info=design_info)
    systematic_raw = (X @ beta).astype(float) if beta.size else np.zeros(len(df))
    systematic_centered = systematic_raw - float(np.mean(systematic_raw))

    log_raw = df[target_col].astype(float).to_numpy()
    log_adj = log_raw - systematic_centered
    gscc_adj = np.exp(log_adj)

    df["systematic_centered"] = systematic_centered
    df["log_gscc_adj"] = log_adj
    df["gscc_adj"] = gscc_adj
    df["fitted_log"] = theta + systematic_centered
    df["residual"] = log_raw - (theta + systematic_centered)
    return df


# ---------------------------------------------------------------------------
# Lambda-selection helpers (inlined from new_test_gscc/robustness.py)
# ---------------------------------------------------------------------------

def _lambda_stability(
    scenarios_df: pd.DataFrame,
    factors: list[str],
    lambda_grid: list[float],
    apply_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refit decomposition per lambda; track r2_raw, r2_adj, r2_drop."""
    if apply_df is None:
        apply_df = scenarios_df

    rows = []
    for lam in lambda_grid:
        fit_raw = _fit_ridge(scenarios_df, factors, lam, target_col="log_gscc")

        df_sc_adj = _apply_decomp(fit_raw, scenarios_df)
        fit_adj = _fit_ridge(df_sc_adj, factors, lam, target_col="log_gscc_adj")

        df_ap = _apply_decomp(fit_raw, apply_df)
        adj_f = df_ap["gscc_adj"].to_numpy(dtype=float)
        adj_f = adj_f[np.isfinite(adj_f)]

        raw_col = "gscc" if "gscc" in df_ap.columns else None
        raw_f = (
            df_ap[raw_col].to_numpy(dtype=float)
            if raw_col is not None
            else np.exp(df_ap["log_gscc"].to_numpy(dtype=float))
        )
        raw_f = raw_f[np.isfinite(raw_f)]

        rows.append({
            "lam": float(lam),
            "r2_raw": float(fit_raw["r2"]),
            "r2_adj": float(fit_adj["r2"]),
            "r2_drop": float(fit_raw["r2"] - fit_adj["r2"]),
            "structured_var_share": float(fit_raw["structured_var_share"]),
            "residual_var_share": float(fit_raw["residual_var_share"]),
            "raw_mean": float(np.mean(raw_f)) if raw_f.size else np.nan,
            "raw_median": float(np.median(raw_f)) if raw_f.size else np.nan,
            "adj_mean": float(np.mean(adj_f)) if adj_f.size else np.nan,
            "adj_median": float(np.median(adj_f)) if adj_f.size else np.nan,
            "adj_p95": float(np.quantile(adj_f, 0.95)) if adj_f.size else np.nan,
        })

    return pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)


def _select_lambda_star(stability_df: pd.DataFrame) -> float:
    """Max r2_drop, tie-break by smaller lambda."""
    tmp = stability_df[["lam", "r2_drop"]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        raise ValueError("No finite lambda candidates")
    best = float(tmp["r2_drop"].max())
    return float(
        tmp[tmp["r2_drop"] == best].sort_values("lam", ascending=True).iloc[0]["lam"]
    )


# ---------------------------------------------------------------------------
# Sampling helpers (inlined from new_test_gscc/gscc_variants.py)
# ---------------------------------------------------------------------------

def _sample_indices(
    n: int, B: int, seed: int, weights: np.ndarray | None = None
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if weights is None:
        return rng.choice(np.arange(n), size=int(B), replace=True)
    w = np.asarray(weights, dtype=float)
    w = w / float(np.sum(w))
    return rng.choice(np.arange(n), size=int(B), replace=True, p=w)


def _draw_distribution_enriched(
    scenarios_df: pd.DataFrame,
    B: int,
    seed: int,
    sampling_scheme: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Draw B samples from an enriched scenario table.

    Returns (draws_raw, idx, draws_df).
    """
    scheme = str(sampling_scheme).lower()
    weights: np.ndarray | None
    if scheme == "uniform":
        weights = None
    elif scheme in {"weighted", "bertram_proxy", "bertram"}:
        weights = bertram_proxy_weights(scenarios_df)
    else:
        raise ValueError(f"Unknown sampling_scheme: {sampling_scheme!r}")

    idx = _sample_indices(len(scenarios_df), int(B), int(seed), weights)
    draws_rows = scenarios_df.iloc[idx].copy().reset_index(drop=True)
    draws_rows["draw"] = np.arange(len(draws_rows))
    draws_arr = draws_rows["gscc"].to_numpy(dtype=float)
    return draws_arr, idx, draws_rows


# ---------------------------------------------------------------------------
# Main public API — structure-adjusted GSCC distribution
# ---------------------------------------------------------------------------

def get_gscc_dist(
    cscc_csv: str = DEFAULT_CSCC_CSV,
    *,
    B: int = 20_000,
    seed: int = 42,
    dmgfuncpar: str | list[str] = ["bootstrap", "estimates"],
    climate: list[str] | None = None,       # default ["expected","uncertain"]
    run: list[str] | None = None,
    SSP: list[str] | None = None,
    RCP: list[str] | None = None,
    discounting: str | None = None,         # None | "ramsey" | "fixed"
    prtp: list[float] | None = None,
    eta: list[str] | None = None,
    dr: list[float] | None = None,
    sampling_scheme: str = "uniform",       # "uniform" | "weighted"
    value_col: str = "50%",
    factors: list[str] | None = None,
    lambda_grid: list[float] | None = None,
    lambda_star: float | None = None,       # skip grid-search if provided
    save_dir: str | None = None,            # persist results here if set
) -> dict:
    """Return the **structure-adjusted GSCC distribution**.

    Four-step pipeline
    ------------------
    1.  *Scenario table.*  Filter CSCC database; aggregate country-level median
        CSCCs per scenario cell; enrich with scenario_id, discount_id, log_gscc.
    2.  *Raw draws.*  Resample B rows with replacement under ``sampling_scheme``
        (``"uniform"`` or ``"weighted"``).
    3.  *Ridge decomposition.*  Fit an additive ridge model on the scenario
        table; select ``lambda_star`` via max R²-drop criterion.
    4.  *Centering & adjustment.*  Subtract the centred systematic component
        from each draw's log-GSCC, then exponentiate.

    Returns
    -------
    dict
        * draws_raw       : ndarray (B,) — raw GSCC draws
        * draws_adj       : ndarray (B,) — structure-adjusted GSCC draws
        * summary_raw     : summary stats for draws_raw
        * summary_adj     : summary stats for draws_adj
        * scenarios_df    : enriched scenario-level DataFrame
        * draws_df        : draw-level DataFrame (includes gscc_adj, log_gscc_adj …)
        * fit             : ridge fit result at lambda_star
        * lambda_star     : float — selected penalty
        * stability_df    : lambda-grid stability table (empty if lambda_star given)
        * n_scenarios     : int
        * B               : int
        * sampling_scheme : str
        * factors         : list[str]
    """
    # ── resolve defaults ─────────────────────────────────────────────────────
    _factors: list[str] = list(factors) if factors is not None else list(DECOMP_FACTORS)
    _lambda_grid: list[float] = (
        list(lambda_grid) if lambda_grid is not None else list(LAMBDA_GRID_DEFAULT)
    )

    # ── Step 1: filter CSCC and build enriched scenario table ────────────────
    df_raw = load_cscc(cscc_csv)

    out = df_raw[df_raw["dmgfuncpar"].isin(set(_as_list_str(dmgfuncpar)))]

    if climate is None:
        climate = ["expected", "uncertain"]
    out = out[out["climate"].isin({str(c) for c in climate})]
    out = out[out["ISO3"] != "WLD"]

    if run is not None:
        out = out[out["run"].isin({str(r) for r in run})]
    if SSP is not None:
        out = out[out["SSP"].isin({str(s) for s in SSP})]
    if RCP is not None:
        out = out[out["RCP"].isin({str(r) for r in RCP})]

    if discounting is not None:
        dl = str(discounting).lower()
        if dl == "ramsey":
            out = out[out["prtp"].notna() & out["eta"].notna()]
        elif dl == "fixed":
            out = out[out["dr"].notna()]
        else:
            raise ValueError("discounting must be None, 'ramsey', or 'fixed'")

    if prtp is not None:
        out = _filter_float_values(out, "prtp", [float(v) for v in prtp])
    if dr is not None:
        out = _filter_float_values(out, "dr", [float(v) for v in dr])
    if eta is not None:
        out = out[out["eta"].astype(str).isin({str(v) for v in eta})]

    out = out.copy()
    scenarios_raw = compute_gscc_per_scenario(out, value_col=value_col)
    if scenarios_raw.empty:
        raise ValueError("No scenarios remain after filtering")

    scenarios_df = _enrich_scenario_table(scenarios_raw)
    if scenarios_df.empty:
        raise ValueError("No positive-GSCC scenarios after enrichment")

    # ── Step 2: raw draws ────────────────────────────────────────────────────
    draws_raw, idx, draws_df = _draw_distribution_enriched(
        scenarios_df, B=int(B), seed=int(seed), sampling_scheme=sampling_scheme
    )

    # ── Step 3: ridge decomposition & lambda selection ───────────────────────
    missing_f = [f for f in _factors if f not in scenarios_df.columns]
    if missing_f:
        raise ValueError(f"Factor columns missing from scenario table: {missing_f}")

    if lambda_star is None:
        stability_df = _lambda_stability(
            scenarios_df, _factors, _lambda_grid, apply_df=draws_df
        )
        lambda_star_val = _select_lambda_star(stability_df)
    else:
        stability_df = pd.DataFrame()
        lambda_star_val = float(lambda_star)

    fit = _fit_ridge(scenarios_df, _factors, lam=lambda_star_val, target_col="log_gscc")

    # ── Step 4: apply adjustment to draws ────────────────────────────────────
    draws_df = _apply_decomp(fit, draws_df)
    draws_adj = draws_df["gscc_adj"].to_numpy(dtype=float)

    # ── Summaries ─────────────────────────────────────────────────────────────
    def _summ(x: np.ndarray) -> dict:
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            raise ValueError("No finite draws")
        mean = float(np.mean(a))
        median = float(np.median(a))
        return {
            "B": int(a.size),
            "mean": mean,
            "median": median,
            "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "p05": float(np.quantile(a, 0.05)),
            "p25": float(np.quantile(a, 0.25)),
            "p50": float(np.quantile(a, 0.50)),
            "p75": float(np.quantile(a, 0.75)),
            "p95": float(np.quantile(a, 0.95)),
            "mean_median_ratio": float(mean / median) if median != 0 else np.nan,
        }

    result: dict = {
        "draws_raw": draws_raw,
        "draws_adj": draws_adj,
        "summary_raw": _summ(draws_raw),
        "summary_adj": _summ(draws_adj),
        "scenarios_df": scenarios_df,
        "draws_df": draws_df,
        "fit": fit,
        "lambda_star": float(lambda_star_val),
        "stability_df": stability_df,
        "n_scenarios": int(len(scenarios_df)),
        "B": int(B),
        "sampling_scheme": str(sampling_scheme),
        "factors": _factors,
    }

    # ── Persist (optional) ───────────────────────────────────────────────────
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "draws.npz",
            draws_raw=draws_raw,
            draws_adj=draws_adj,
        )
        with (out_dir / "summary.json").open("w") as fh:
            json.dump(
                {
                    "lambda_star": float(lambda_star_val),
                    "n_scenarios": int(len(scenarios_df)),
                    "B": int(B),
                    "sampling_scheme": str(sampling_scheme),
                    "summary_raw": result["summary_raw"],
                    "summary_adj": result["summary_adj"],
                },
                fh,
                indent=2,
            )
        scenarios_df.to_csv(out_dir / "scenarios.csv", index=False)
        draws_df.to_csv(out_dir / "draws_df.csv", index=False)
        if not stability_df.empty:
            stability_df.to_csv(out_dir / "stability.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args(argv: None | list[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute the structure-adjusted GSCC distribution (Option B). "
            "Runs the 4-step ridge-decomposition pipeline and saves results "
            "to --out-dir."
        )
    )
    p.add_argument("--csv", default=DEFAULT_CSCC_CSV, metavar="PATH",
                   help="CSCC database CSV (default: %(default)s)")
    p.add_argument("--B", type=int, default=20_000,
                   help="Number of bootstrap draws (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default: %(default)s)")
    p.add_argument("--out-dir", default="data_source/gscc",
                   help="Output directory (default: %(default)s)")
    p.add_argument(
        "--sampling-scheme",
        default="uniform",
        choices=["uniform", "weighted"],
        help="'uniform' or 'weighted' (Bertram proxy).",
    )
    p.add_argument("--lambda-star", type=float, default=None,
                   help="Pin λ (skip grid-search).")
    p.add_argument("--dmgfuncpar", nargs="+", default=["bootstrap", "estimates"],
                   help="Filter by damage function parameter (default: %(default)s)")
    p.add_argument("--run", nargs="*", default=None)
    p.add_argument("--SSP", nargs="*", default=None)
    p.add_argument("--RCP", nargs="*", default=None)
    p.add_argument("--discounting", default=None, choices=[None, "ramsey", "fixed"])
    p.add_argument("--prtp", type=float, nargs="*", default=None)
    p.add_argument("--eta", nargs="*", default=None)
    p.add_argument("--dr", type=float, nargs="*", default=None)
    p.add_argument("--no-save", action="store_true", help="Skip persisting output files.")
    return p.parse_args(argv)


def _is_all(x: list | None) -> bool:
    return x is None or len(x) == 0

def _upper_list(x: list[str] | None) -> list[str] | None:
    if _is_all(x):
        return None
    return [str(v).upper() for v in x]

def _fmt_list_plain(x: list[str] | None) -> str:
    """Join values as 'A, B, C' after uppercasing. Assumes x is not ALL."""
    assert x is not None and len(x) > 0
    return ", ".join(_upper_list(x) or [])

def _fmt_ssp_list(ssp: list[str] | None) -> str | None:
    """Return 'SSP1, SSP2' etc (uppercased), or None if ALL."""
    if _is_all(ssp):
        return None
    # ensure explicit SSP prefix is kept as SSP1, SSP2...
    vals = []
    for v in ssp:
        s = str(v).upper()
        vals.append(s if s.startswith("SSP") else f"SSP{s}")
    return ", ".join(vals)

def _fmt_rcp_list(rcp: list[str] | None) -> str | None:
    """Return 'RCP45, RCP60' etc (uppercased), or None if ALL."""
    if _is_all(rcp):
        return None
    vals = []
    for v in rcp:
        s = str(v).upper()
        vals.append(s if s.startswith("RCP") else f"RCP{s}")
    return ", ".join(vals)

def _fmt_discounting(discounting: str | None) -> str | None:
    """Return 'RAMSEY' or 'FIXED' or None."""
    if discounting is None:
        return None
    return str(discounting).upper()

def _context_sentence(
    *,
    SSP: list[str] | None,
    RCP: list[str] | None,
    discounting: str | None,
    run: list[str] | None,
    climate: list[str] | None,
    dmgfuncpar: list[str] | None,
    sampling_scheme: str,
) -> str:
    # Normalize
    ssp_txt = _fmt_ssp_list(SSP)
    rcp_txt = _fmt_rcp_list(RCP)
    run_u = _upper_list(run)
    climate_u = _upper_list(climate)
    disc_u = _fmt_discounting(discounting)
    dmg_u = _upper_list(dmgfuncpar) or ["BOOTSTRAP"]
    scheme_u = str(sampling_scheme).upper()

    # ALL detection: no restrictions on the visible spec dimensions
    no_restrictions = (
        ssp_txt is None
        and rcp_txt is None
        and _is_all(run)
        and (discounting is None)
        and _is_all(climate)
    )

    if no_restrictions:
        base = "In ALL scenarios"
    else:
        chunks: list[str] = ["In scenarios restricted to"]
        if ssp_txt is not None:
            chunks.append(ssp_txt)
        if rcp_txt is not None:
            chunks.append(rcp_txt)
        # climate is optional, include only if restricted
        if climate_u is not None:
            chunks.append(f"climate={{{_fmt_list_plain(climate_u)}}}")
        base = " ".join(chunks)

        # "using ..." clause ONLY for discounting and run, as you requested
        using_parts: list[str] = []
        if disc_u is not None:
            using_parts.append(f"discounting ∈ {{{disc_u}}}")
        if run_u is not None:
            using_parts.append(f"dmg. function ∈ {{{_fmt_list_plain(run_u)}}}")
        if using_parts:
            base = base + " using " + " and ".join(using_parts)

    tech = f"method={{{_fmt_list_plain(dmg_u)}}}, sampling={{{scheme_u}}}"
    return f"{base} | {tech}"


def _print_report(result: dict, *, context: str) -> None:
    try:
        from colorama import Fore, Style, init as colorama_init
        colorama_init(autoreset=True)
        col = Fore.GREEN
        reset = Style.RESET_ALL
        bold = Style.BRIGHT
    except Exception:
        col = reset = bold = ""

    s_adj = result["summary_adj"]
    unit = "/ tCO2"

    B = int(result["B"])
    std = float(s_adj["std"])
    se = std / (B ** 0.5) if B > 0 else float("nan")

    gscc = float(s_adj["mean"])  # or mean if you prefer

    line = "=" * 92
    print(line)
    print(f"{bold}{context}{reset}")
    print(line)
    print(
        f"{col}{bold}The GSCC is: US$ {gscc:.1f} ± {se:.2f} {unit}{reset} "
        f"(std={std:.2f}, B={B:,}, n_scenarios={result['n_scenarios']}, λ*={result['lambda_star']:.4g})"
    )


def main(argv: None | list[str] = None) -> None:
    args = _parse_args(argv)

    save_dir: str | None = None if args.no_save else args.out_dir

    result = get_gscc_dist(
        cscc_csv=args.csv,
        B=args.B,
        seed=args.seed,
        dmgfuncpar=args.dmgfuncpar,
        climate=["expected", "uncertain"],
        run=args.run or None,
        SSP=args.SSP or None,
        RCP=args.RCP or None,
        discounting=args.discounting or None,
        prtp=args.prtp or None,
        eta=args.eta or None,
        dr=args.dr or None,
        sampling_scheme=args.sampling_scheme,
        lambda_star=args.lambda_star,
        save_dir=save_dir,
    )

    s_raw = result["summary_raw"]
    s_adj = result["summary_adj"]
    
    print("Summary of GSCC distribution:")
    print(
        f"Structure-adjusted GSCC  (λ*={result['lambda_star']:.4g}, "
        f"n_scenarios={result['n_scenarios']}, B={result['B']:,})"
    )
    print(
        f"  Raw  : mean={s_raw['mean']:.1f}  median={s_raw['median']:.1f}  "
        f"p05={s_raw['p05']:.1f}  p95={s_raw['p95']:.1f}"
    )
    print(
        f"  Adj  : mean={s_adj['mean']:.1f}  median={s_adj['median']:.1f}  "
        f"p05={s_adj['p05']:.1f}  p95={s_adj['p95']:.1f}"
    )
    print("\n")
    context = _context_sentence(
        SSP=args.SSP or None,
        RCP=args.RCP or None,
        discounting=args.discounting or None,
        run=args.run or None,
        climate=["expected", "uncertain"],
        dmgfuncpar=args.dmgfuncpar,
        sampling_scheme=args.sampling_scheme,
    )
    _print_report(result, context=context)
    
    if save_dir:
        print(f"\nOutput saved to: {save_dir}")


if __name__ == "__main__":
    main()
    # example run : 
    # python -m compute_gscc --dmgfuncpar bootstrap estimates --sampling-scheme uniform --B 20000 --seed 42 --out-dir data_source/gscc