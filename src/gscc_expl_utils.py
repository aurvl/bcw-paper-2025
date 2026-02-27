"""GSCC Explorer utilities — self-contained module.

This module consolidates all logic previously in ``new_test_gscc/`` so that
``notebooks/gscc_explorer.ipynb`` has no dependency on that package.  Copy
everything from new_test_gscc with relative imports replaced by absolute ones.

Sections
--------
1. Config / constants  (from new_test_gscc/config.py)
2. GSCC variant builders  (from new_test_gscc/gscc_variants.py)
3. Ridge decomposition  (from new_test_gscc/decomposition.py)
4. Robustness experiments  (from new_test_gscc/robustness.py)
5. Matplotlib plotting helpers  (from new_test_gscc/plotting.py)
6. Variant selection  (same rule as lasttest.ipynb / src/utils.py)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Config / constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SEED: int = 42
DEFAULT_B: int = 20_000

LAMBDA_GRID_DEFAULT: list[float] = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

FACTORS_DEFAULT: list[str] = [
    "run",
    "discount_id",
    "climate",
    "scenario_id",
    "dmgfuncpar",
]


@dataclass(frozen=True)
class GSCCVariantSpec:
    """Specification of a GSCC construction variant."""

    name: str
    dmgfuncpar: FrozenSet[str]
    sampling_scheme: str  # 'uniform' or 'bertram_proxy'


DEFAULT_VARIANTS: tuple[GSCCVariantSpec, ...] = (
    GSCCVariantSpec(
        name="bootstrap_only_uniform",
        dmgfuncpar=frozenset({"bootstrap"}),
        sampling_scheme="uniform",
    ),
    GSCCVariantSpec(
        name="estimates_only_uniform",
        dmgfuncpar=frozenset({"estimates"}),
        sampling_scheme="uniform",
    ),
    # GSCCVariantSpec(
    #     name="bootstrap_only_weighted",
    #     dmgfuncpar=frozenset({"bootstrap"}),
    #     sampling_scheme="weighted",
    # ),
    # GSCCVariantSpec(
    #     name="estimates_only_weighted",
    #     dmgfuncpar=frozenset({"estimates"}),
    #     sampling_scheme="weighted",
    # ),
    GSCCVariantSpec(
        name="mixed_uniform",
        dmgfuncpar=frozenset({"bootstrap", "estimates"}),
        sampling_scheme="uniform",
    ),
    GSCCVariantSpec(
        name="mixed_weighted",
        dmgfuncpar=frozenset({"bootstrap", "estimates"}),
        sampling_scheme="weighted",
    ),
    
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — GSCC variant builders
# ═══════════════════════════════════════════════════════════════════════════

from src.compute_gscc import (  # noqa: E402
    DEFAULT_CSCC_CSV,
    bertram_proxy_weights,
    compute_gscc_per_scenario,
    load_cscc,
)


def _as_list_str(x: str | Iterable[str]) -> list[str]:
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _fmt_float_id(v: float) -> str:
    """Stable formatting for IDs: 1.5 -> '1p5', 2.0 -> '2p0'."""
    s = f"{float(v):.12g}"
    if "e" not in s and "." not in s:
        s = s + ".0"
    return s.replace("-", "m").replace(".", "p")


def infer_discounting_type(prtp: Any, eta: Any, dr: Any) -> str:
    prtp_ok = pd.notna(prtp)
    eta_ok = pd.notna(eta)
    dr_ok = pd.notna(dr)
    if prtp_ok and eta_ok:
        return "ramsey"
    if dr_ok:
        return "fixed"
    return "unknown"


def make_discount_id(discounting_type: str, prtp: Any, eta: Any, dr: Any) -> str:
    if discounting_type == "ramsey":
        prtp_s = _fmt_float_id(float(prtp)) if pd.notna(prtp) else "nan"
        eta_s = str(eta) if pd.notna(eta) else "nan"
        eta_s = eta_s.replace("-", "m").replace(".", "p")
        return f"ramsey_prtp{prtp_s}_eta{eta_s}"
    if discounting_type == "fixed":
        dr_s = _fmt_float_id(float(dr)) if pd.notna(dr) else "nan"
        return f"fixed_dr{dr_s}"
    return "unknown"


def _sample_indices(
    n: int,
    *,
    B: int,
    seed: int,
    weights: None | np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if weights is None:
        return rng.choice(np.arange(int(n)), size=int(B), replace=True)
    w = np.asarray(weights, dtype=float)
    if w.shape != (int(n),):
        raise ValueError("weights must have shape (len(scenarios_df),)")
    w = w / float(np.sum(w))
    return rng.choice(np.arange(int(n)), size=int(B), replace=True, p=w)


def build_scenario_table(
    cscc_csv: str = DEFAULT_CSCC_CSV,
    *,
    dmgfuncpar: str | Iterable[str],
    climate: Iterable[str] = ("expected", "uncertain"),
    value_col: str = "50%",
) -> pd.DataFrame:
    """Build a scenario-level GSCC table using the repo aggregation logic."""
    df = load_cscc(cscc_csv)
    df = df[df["ISO3"] != "WLD"].copy()

    dmg_set = set(_as_list_str(dmgfuncpar))
    df = df[df["dmgfuncpar"].isin(dmg_set)]
    df = df[df["climate"].isin({str(c) for c in climate})]

    scenarios_df = compute_gscc_per_scenario(df, value_col=value_col)
    if scenarios_df.empty:
        raise ValueError("No scenarios remain after filters")

    scenarios_df = scenarios_df.copy()
    for col in ["run", "climate", "SSP", "RCP", "dmgfuncpar"]:
        if col in scenarios_df.columns:
            scenarios_df[col] = scenarios_df[col].astype(str)

    scenarios_df["scenario_id"] = (
        scenarios_df["SSP"].astype(str) + "_" + scenarios_df["RCP"].astype(str)
    )
    scenarios_df["discounting_type"] = [
        infer_discounting_type(p, e, d)
        for p, e, d in zip(
            scenarios_df.get("prtp"),
            scenarios_df.get("eta"),
            scenarios_df.get("dr"),
            strict=False,
        )
    ]
    scenarios_df["discount_id"] = [
        make_discount_id(t, p, e, d)
        for t, p, e, d in zip(
            scenarios_df["discounting_type"],
            scenarios_df.get("prtp"),
            scenarios_df.get("eta"),
            scenarios_df.get("dr"),
            strict=False,
        )
    ]

    scenarios_df["gscc"] = pd.to_numeric(scenarios_df["gscc"], errors="coerce")
    if (~np.isfinite(scenarios_df["gscc"].to_numpy(dtype=float))).any():
        raise ValueError("Scenario table has non-finite gscc values")
    if (scenarios_df["gscc"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("Scenario table has non-positive gscc values; cannot take logs")

    scenarios_df["log_gscc"] = np.log(scenarios_df["gscc"].to_numpy(dtype=float))

    keep_front = [
        "gscc", "log_gscc", "run", "climate", "dmgfuncpar",
        "SSP", "RCP", "scenario_id", "discounting_type", "discount_id",
        "prtp", "eta", "dr", "n_iso3",
    ]
    keep_front_existing = [c for c in keep_front if c in scenarios_df.columns]
    remainder = [c for c in scenarios_df.columns if c not in keep_front_existing]
    return scenarios_df[keep_front_existing + remainder].reset_index(drop=True)


@dataclass(frozen=True)
class VariantOutputs:
    variant_name: str
    scenarios_df: pd.DataFrame
    draws: np.ndarray
    draw_indices: np.ndarray
    draws_df: pd.DataFrame


def draw_distribution(
    scenarios_df: pd.DataFrame,
    *,
    B: int,
    seed: int,
    sampling_scheme: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Draw a distribution from a scenario table and return (draws, idx, draws_df)."""
    scheme = str(sampling_scheme).lower()
    if scheme == "uniform":
        weights = None
    elif scheme in {"bertram_proxy", "bertram", "weighted"}:
        weights = bertram_proxy_weights(scenarios_df, run_col="run")
    else:
        raise ValueError("sampling_scheme must be 'uniform' or 'weighted'")

    idx = _sample_indices(len(scenarios_df), B=int(B), seed=int(seed), weights=weights)
    draws_df = scenarios_df.iloc[idx].reset_index(drop=True).copy()
    draws = draws_df["gscc"].to_numpy(dtype=float)

    draws_df.insert(0, "draw", np.arange(len(draws_df), dtype=int))
    draws_df.insert(1, "scenario_row", idx.astype(int))
    draws_df["log_gscc"] = np.log(draws)
    return draws, idx, draws_df


def compute_variant(
    spec: GSCCVariantSpec,
    *,
    cscc_csv: str = DEFAULT_CSCC_CSV,
    B: int,
    seed: int,
    climate: Iterable[str] = ("expected", "uncertain"),
    value_col: str = "50%",
) -> VariantOutputs:
    """Compute scenario table and draw distribution for one GSCC variant."""
    scenarios_df = build_scenario_table(
        cscc_csv,
        dmgfuncpar=sorted(spec.dmgfuncpar),
        climate=climate,
        value_col=value_col,
    )

    draws, idx, draws_df = draw_distribution(
        scenarios_df,
        B=int(B),
        seed=int(seed),
        sampling_scheme=spec.sampling_scheme,
    )

    required = [
        "gscc", "log_gscc", "run", "climate", "dmgfuncpar",
        "SSP", "RCP", "scenario_id", "discounting_type", "discount_id",
    ]
    missing = [c for c in required if c not in draws_df.columns]
    if missing:
        raise ValueError(f"draws_df missing required columns: {missing}")

    if (~np.isfinite(draws_df["log_gscc"].to_numpy(dtype=float))).any():
        raise ValueError("Non-finite log_gscc in draws_df")
    if (draws_df["gscc"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("Non-positive gscc in draws_df")

    for col in [
        "run", "climate", "dmgfuncpar", "SSP", "RCP",
        "scenario_id", "discounting_type", "discount_id",
    ]:
        draws_df[col] = draws_df[col].astype(str)

    return VariantOutputs(
        variant_name=spec.name,
        scenarios_df=scenarios_df,
        draws=draws,
        draw_indices=idx,
        draws_df=draws_df,
    )


def compute_variants(
    specs: Iterable[GSCCVariantSpec],
    *,
    cscc_csv: str = DEFAULT_CSCC_CSV,
    B: int,
    seed: int,
    climate: Iterable[str] = ("expected", "uncertain"),
    value_col: str = "50%",
) -> dict[str, VariantOutputs]:
    out: dict[str, VariantOutputs] = {}
    for spec in specs:
        out[spec.name] = compute_variant(
            spec,
            cscc_csv=cscc_csv,
            B=int(B),
            seed=int(seed),
            climate=climate,
            value_col=value_col,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Ridge decomposition
# (from new_test_gscc/decomposition.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DesignInfo:
    factors: list[str]
    baselines: dict[str, str]
    levels: dict[str, list[str]]
    col_slices: dict[str, slice]
    column_names: list[str]


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_design_matrix(
    df: pd.DataFrame,
    factors: list[str],
    *,
    design_info: DesignInfo | None = None,
) -> tuple[Any, DesignInfo]:
    """Build sparse one-hot design matrix with drop-first encoding."""
    try:
        from scipy import sparse  # type: ignore
    except Exception as e:
        raise ImportError("SciPy is required for sparse ridge fitting") from e

    n = int(len(df))
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    baselines: dict[str, str] = {} if design_info is None else dict(design_info.baselines)
    levels: dict[str, list[str]] = (
        {} if design_info is None else {k: list(v) for k, v in design_info.levels.items()}
    )
    col_slices: dict[str, slice] = {}
    column_names: list[str] = []

    col_offset = 0
    for factor in factors:
        if factor not in df.columns:
            raise ValueError(f"Missing factor column: {factor}")
        s = df[factor]
        if s.isna().any():
            raise ValueError(
                f"Factor '{factor}' contains NaNs; drop rows explicitly before fitting"
            )

        if design_info is None:
            levs = sorted(pd.unique(s.astype(str)))
            levels[factor] = levs
            baseline = levs[0] if levs else ""
            baselines[factor] = baseline
        else:
            if factor not in levels or factor not in baselines:
                raise ValueError(f"design_info missing factor '{factor}'")
            levs = levels[factor]
            baseline = baselines[factor]

        if len(levs) <= 1:
            col_slices[factor] = slice(col_offset, col_offset)
            continue

        code = pd.Categorical(s.astype(str), categories=levs, ordered=True).codes
        if np.any(code < 0):
            bad = pd.unique(s.astype(str)[code < 0])
            raise ValueError(
                f"Unseen levels in factor '{factor}': {sorted(map(str, bad))[:5]}"
            )

        nz = np.nonzero(code)[0]
        if nz.size:
            row_idx.extend(nz.tolist())
            col_idx.extend((col_offset + (code[nz] - 1)).tolist())
            data.extend([1.0] * int(nz.size))

        start = col_offset
        end = col_offset + (len(levs) - 1)
        col_slices[factor] = slice(start, end)
        column_names.extend([f"{factor}={lev}" for lev in levs[1:]])
        col_offset = end

    p_ = col_offset
    X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, p_), dtype=float)
    info = DesignInfo(
        factors=list(factors),
        baselines=baselines,
        levels=levels,
        col_slices=col_slices,
        column_names=column_names,
    )
    return X, info


def _ridge_solve_with_intercept(
    y: np.ndarray,
    X: Any,
    *,
    lam: float,
    max_iter: int = 2_000,
    atol: float = 1e-10,
    btol: float = 1e-10,
) -> tuple[float, np.ndarray, dict[str, Any] | None]:
    """Solve ridge with unpenalized intercept via augmented least squares."""
    if float(lam) < 0:
        raise ValueError("lam must be >= 0")

    try:
        from scipy import sparse  # type: ignore
        from scipy.sparse.linalg import lsqr  # type: ignore
    except Exception as e:
        raise ImportError("SciPy is required for ridge fitting") from e

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values")

    n = int(y.shape[0])
    p_ = int(X.shape[1])

    if p_ == 0:
        return float(np.mean(y)), np.zeros(0, dtype=float), None

    ones = sparse.csr_matrix(np.ones((n, 1), dtype=float))
    Z = sparse.hstack([ones, X], format="csr")

    pen = sparse.hstack(
        [
            sparse.csr_matrix((p_, 1), dtype=float),
            sparse.identity(p_, format="csr", dtype=float),
        ],
        format="csr",
    )
    A = sparse.vstack([Z, np.sqrt(float(lam)) * pen], format="csr")
    b = np.concatenate([y, np.zeros(p_, dtype=float)])

    sol = lsqr(A, b, atol=float(atol), btol=float(btol), iter_lim=int(max_iter))
    w = np.asarray(sol[0], dtype=float)
    info = {
        "istop": int(sol[1]),
        "itn": int(sol[2]),
        "r1norm": float(sol[3]),
        "r2norm": float(sol[4]),
    }
    return float(w[0]), w[1:], info


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")
    if y.size == 0:
        raise ValueError("Empty y")
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - (sse / sst) if sst > 0 else 0.0


def fit_ridge_decomposition(
    draws_df: pd.DataFrame,
    factors: list[str],
    lam: float,
    *,
    target_col: str = "log_gscc",
) -> dict[str, Any]:
    """Fit ridge decomposition model and return fitted components."""
    _require_columns(draws_df, [target_col] + list(factors))

    df = draws_df.loc[:, [target_col] + list(factors)].copy()
    if df.isna().any(axis=None):
        bad_cols = [c for c in df.columns if df[c].isna().any()]
        raise ValueError(f"NaNs present in required columns: {bad_cols}")

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in {target_col}")

    X, info = build_design_matrix(df, factors)
    theta_hat, beta_hat, lsqr_info = _ridge_solve_with_intercept(y, X, lam=float(lam))

    systematic_raw = (X @ beta_hat).astype(float)
    systematic_centered = systematic_raw - float(np.mean(systematic_raw))
    fitted = theta_hat + systematic_centered
    residual = y - fitted

    effects_tables: dict[str, pd.Series] = {}
    for factor in info.factors:
        levs = info.levels.get(factor, [])
        baseline = info.baselines.get(factor, levs[0] if levs else "")
        sl = info.col_slices.get(factor, slice(0, 0))
        coef_factor = beta_hat[sl] if sl.stop > sl.start else np.zeros(0, dtype=float)

        effects: dict[str, float] = {str(baseline): 0.0}
        for lev, c in zip(levs[1:], coef_factor, strict=False):
            effects[str(lev)] = float(c)
        effects_tables[factor] = pd.Series(effects, dtype=float).sort_index()

    r2 = r2_score(y, fitted)

    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    var_sys = float(np.var(systematic_centered, ddof=1)) if y.size > 1 else 0.0
    var_res = float(np.var(residual, ddof=1)) if y.size > 1 else 0.0
    structured_share = (var_sys / var_y) if var_y > 0 else 0.0
    residual_share = (var_res / var_y) if var_y > 0 else 0.0

    return {
        "target_col": target_col,
        "factors": list(factors),
        "lam": float(lam),
        "theta_hat": float(theta_hat),
        "beta_hat": beta_hat,
        "design_info": info,
        "systematic_raw": systematic_raw,
        "systematic_centered": systematic_centered,
        "systematic_mean_raw": float(np.mean(systematic_raw)),
        "residual": residual,
        "fitted": fitted,
        "effects_tables": effects_tables,
        "r2": float(r2),
        "var_y": var_y,
        "var_systematic": var_sys,
        "var_residual": var_res,
        "structured_var_share": float(structured_share),
        "residual_var_share": float(residual_share),
        "lsqr_info": lsqr_info,
    }


def apply_decomposition(
    fit_result: dict[str, Any],
    draws_df: pd.DataFrame,
    *,
    require_positive: bool = True,
) -> pd.DataFrame:
    """Apply a fitted decomposition and add adjusted columns.

    Adds:
    - systematic_centered
    - log_gscc_adj = log_gscc - systematic_centered
    - gscc_adj = exp(log_gscc_adj)
    """
    factors = list(fit_result.get("factors", []))
    target_col = str(fit_result.get("target_col", "log_gscc"))
    theta_hat = float(fit_result.get("theta_hat"))
    beta_hat = np.asarray(fit_result.get("beta_hat"), dtype=float)
    design_info = fit_result.get("design_info")
    if not isinstance(design_info, DesignInfo):
        raise ValueError("fit_result missing DesignInfo")

    _require_columns(draws_df, [target_col] + factors)

    df = draws_df.copy()
    X, _info2 = build_design_matrix(df, factors, design_info=design_info)
    if beta_hat.shape != (int(X.shape[1]),):
        raise ValueError("beta_hat shape does not match design matrix")

    systematic_raw = (X @ beta_hat).astype(float)
    systematic_centered = systematic_raw - float(np.mean(systematic_raw))

    log_raw = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(log_raw).all():
        raise ValueError(f"Non-finite values in {target_col} at apply time")

    log_adj = log_raw - systematic_centered
    gscc_adj = np.exp(log_adj)

    if (~np.isfinite(log_adj)).any() or (~np.isfinite(gscc_adj)).any():
        raise ValueError("Non-finite values produced by adjustment")
    if require_positive and (gscc_adj <= 0).any():
        raise ValueError("Non-positive gscc_adj produced by adjustment")

    df["systematic_centered"] = systematic_centered
    df["log_gscc_adj"] = log_adj
    df["gscc_adj"] = gscc_adj
    df["fitted_log"] = theta_hat + systematic_centered
    df["residual"] = log_raw - df["fitted_log"].to_numpy(dtype=float)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Robustness experiments
# (from new_test_gscc/robustness.py)
# ═══════════════════════════════════════════════════════════════════════════


def summarize_draws(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("No finite values")
    mean = float(np.mean(a))
    median = float(np.median(a))
    return {
        "B": float(a.size),
        "mean": mean,
        "median": median,
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p05": float(np.quantile(a, 0.05)),
        "p95": float(np.quantile(a, 0.95)),
        "mean_median_ratio": float(mean / median) if median != 0 else np.nan,
    }


def ridge_r2_from_labels(
    df: pd.DataFrame, *, factors: list[str], lam: float, target_col: str
) -> float:
    res = fit_ridge_decomposition(df, factors, lam, target_col=target_col)
    return float(res["r2"])


def leave_one_spec_out_influence(
    scenarios_df: pd.DataFrame,
    *,
    sampling_scheme: str,
    B: int,
    seed: int,
    group_col: str,
) -> pd.DataFrame:
    """Leave-one-category-out influence on distribution summary stats."""
    if scenarios_df.empty:
        raise ValueError("Empty scenarios_df")
    if group_col not in scenarios_df.columns:
        raise ValueError(f"Missing group_col: {group_col}")

    base_draws, _idx, _ddf = draw_distribution(
        scenarios_df,
        B=int(B),
        seed=int(seed),
        sampling_scheme=sampling_scheme,
    )
    base = summarize_draws(base_draws)

    rows: list[dict[str, Any]] = []
    levels = sorted(pd.unique(scenarios_df[group_col].astype(str)))
    for lev in levels:
        sub = scenarios_df[scenarios_df[group_col].astype(str) != str(lev)].copy()
        if sub.empty:
            continue
        draws, _i2, _df2 = draw_distribution(
            sub, B=int(B), seed=int(seed), sampling_scheme=sampling_scheme
        )
        s = summarize_draws(draws)
        rows.append(
            {
                "group_col": str(group_col),
                "left_out": str(lev),
                "n_scenarios": int(len(sub)),
                "delta_mean": float(s["mean"] - base["mean"]),
                "delta_median": float(s["median"] - base["median"]),
                "delta_p95": float(s["p95"] - base["p95"]),
                "delta_mean_pct": (
                    float((s["mean"] - base["mean"]) / base["mean"])
                    if base["mean"] != 0
                    else np.nan
                ),
                "delta_median_pct": (
                    float((s["median"] - base["median"]) / base["median"])
                    if base["median"] != 0
                    else np.nan
                ),
                "delta_p95_pct": (
                    float((s["p95"] - base["p95"]) / base["p95"])
                    if base["p95"] != 0
                    else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["group_col", "delta_p95_pct"], ascending=[True, False]
    ).reset_index(drop=True)


def lambda_stability_analysis(
    fit_df: pd.DataFrame,
    *,
    factors: list[str],
    lambda_grid: list[float],
    apply_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refit decomposition across lambdas and track stability metrics."""
    if apply_df is None:
        apply_df = fit_df
    rows: list[dict[str, Any]] = []

    for lam in lambda_grid:
        fit_raw = fit_ridge_decomposition(fit_df, factors, lam, target_col="log_gscc")

        df_fit_adj = apply_decomposition(fit_raw, fit_df)
        fit_adj = fit_ridge_decomposition(df_fit_adj, factors, lam, target_col="log_gscc_adj")

        df_apply_adj = apply_decomposition(fit_raw, apply_df)
        adj_stats = summarize_draws(df_apply_adj["gscc_adj"].to_numpy(dtype=float))
        raw_stats = summarize_draws(df_apply_adj["gscc"].to_numpy(dtype=float))

        rows.append(
            {
                "lam": float(lam),
                "r2_raw": float(fit_raw["r2"]),
                "r2_adj": float(fit_adj["r2"]),
                "r2_drop": float(fit_raw["r2"] - fit_adj["r2"]),
                "structured_var_share": float(fit_raw["structured_var_share"]),
                "residual_var_share": float(fit_raw["residual_var_share"]),
                "raw_mean": float(raw_stats["mean"]),
                "raw_median": float(raw_stats["median"]),
                "raw_p95": float(raw_stats["p95"]),
                "adj_mean": float(adj_stats["mean"]),
                "adj_median": float(adj_stats["median"]),
                "adj_p95": float(adj_stats["p95"]),
                "adj_mean_median_ratio": float(adj_stats["mean_median_ratio"]),
            }
        )

    return pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)


def select_lambda_star(stability_df: pd.DataFrame) -> float:
    """Deterministic lambda selection: maximize R2 drop, tie-break by smaller lambda."""
    needed = {"lam", "r2_drop"}
    if not needed.issubset(stability_df.columns):
        raise ValueError(
            f"stability_df missing columns: {sorted(needed.difference(stability_df.columns))}"
        )

    tmp = stability_df[["lam", "r2_drop"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if tmp.empty:
        raise ValueError("No finite lambda candidates")

    best_drop = float(tmp["r2_drop"].max())
    cand = tmp[tmp["r2_drop"] == best_drop].sort_values("lam", ascending=True)
    return float(cand.iloc[0]["lam"])


def effect_vector_correlation(
    fit_a: dict[str, Any],
    fit_b: dict[str, Any],
    *,
    factor: str,
) -> float:
    """Correlation of per-level effects for a factor between two fits."""
    ea = fit_a.get("effects_tables", {}).get(factor)
    eb = fit_b.get("effects_tables", {}).get(factor)
    if ea is None or eb is None:
        return np.nan
    sa = pd.Series(ea, dtype=float)
    sb = pd.Series(eb, dtype=float)
    common = sorted(set(sa.index).intersection(set(sb.index)))
    if len(common) < 2:
        return np.nan
    xa = sa.loc[common].to_numpy(dtype=float)
    xb = sb.loc[common].to_numpy(dtype=float)
    if np.std(xa) == 0 or np.std(xb) == 0:
        return np.nan
    return float(np.corrcoef(xa, xb)[0, 1])


def family_exclusion_analysis(
    fit_df: pd.DataFrame,
    *,
    factors: list[str],
    lam: float,
    exclude: list[str],
    apply_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop one family factor at a time and recompute key metrics."""
    if apply_df is None:
        apply_df = fit_df
    rows: list[dict[str, Any]] = []

    base_fit = fit_ridge_decomposition(fit_df, factors, lam, target_col="log_gscc")
    base_fit_adj_df = apply_decomposition(base_fit, fit_df)
    base_fit_adj = fit_ridge_decomposition(
        base_fit_adj_df, factors, lam, target_col="log_gscc_adj"
    )

    base_apply_adj = apply_decomposition(base_fit, apply_df)
    base_stats = summarize_draws(base_apply_adj["gscc_adj"].to_numpy(dtype=float))
    rows.append(
        {
            "excluded": "none",
            "r2_raw": float(base_fit["r2"]),
            "r2_adj": float(base_fit_adj["r2"]),
            "r2_drop": float(base_fit["r2"] - base_fit_adj["r2"]),
            "structured_var_share": float(base_fit["structured_var_share"]),
            "adj_mean": float(base_stats["mean"]),
            "adj_median": float(base_stats["median"]),
            "adj_p95": float(base_stats["p95"]),
        }
    )

    for f in exclude:
        keep = [x for x in factors if x != f]
        if not keep:
            continue
        fit_raw = fit_ridge_decomposition(fit_df, keep, lam, target_col="log_gscc")
        fit_adj_df = apply_decomposition(fit_raw, fit_df)
        fit_adj = fit_ridge_decomposition(fit_adj_df, keep, lam, target_col="log_gscc_adj")

        apply_adj = apply_decomposition(fit_raw, apply_df)
        stats = summarize_draws(apply_adj["gscc_adj"].to_numpy(dtype=float))
        rows.append(
            {
                "excluded": str(f),
                "r2_raw": float(fit_raw["r2"]),
                "r2_adj": float(fit_adj["r2"]),
                "r2_drop": float(fit_raw["r2"] - fit_adj["r2"]),
                "structured_var_share": float(fit_raw["structured_var_share"]),
                "adj_mean": float(stats["mean"]),
                "adj_median": float(stats["median"]),
                "adj_p95": float(stats["p95"]),
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Matplotlib plotting helpers
# (from new_test_gscc/plotting.py)
# ═══════════════════════════════════════════════════════════════════════════


def _top_levels_by_count(df: pd.DataFrame, col: str, top_k: int) -> list[str]:
    counts = df[col].astype(str).value_counts(dropna=False)
    return [str(x) for x in counts.head(int(top_k)).index.tolist()]


def bar_median_by(
    ax,
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str = "gscc",
    top_k: int | None = None,
    title: str | None = None,
) -> None:
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Missing required columns")

    d = df.copy()
    d[group_col] = d[group_col].astype(str)
    if top_k is not None:
        keep = set(_top_levels_by_count(d, group_col, int(top_k)))
        d = d[d[group_col].isin(keep)].copy()

    g = d.groupby(group_col, dropna=False)[value_col].median().sort_values(ascending=False)

    ax.bar(np.arange(len(g)), g.to_numpy(dtype=float))
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(g.index.tolist(), rotation=90)
    ax.set_ylabel(f"median({value_col})")
    if title:
        ax.set_title(title)


def heatmap_median_ssp_rcp(
    ax,
    df: pd.DataFrame,
    *,
    value_col: str = "gscc",
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    needed = {"SSP", "RCP", value_col}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(needed.difference(df.columns))}")

    piv = df.groupby(["SSP", "RCP"], dropna=False)[value_col].median().unstack("RCP")
    ssp = [str(x) for x in piv.index.tolist()]
    rcp = [str(x) for x in piv.columns.tolist()]
    mat = piv.to_numpy(dtype=float)

    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(rcp)))
    ax.set_xticklabels(rcp)
    ax.set_yticks(np.arange(len(ssp)))
    ax.set_yticklabels(ssp)
    ax.set_xlabel("RCP")
    ax.set_ylabel("SSP")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def influence_bars(
    ax,
    influence_df: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
    max_bars: int = 30,
) -> None:
    if influence_df.empty:
        ax.text(0.5, 0.5, "No influence results", ha="center", va="center")
        return
    if metric not in influence_df.columns:
        raise ValueError(f"Missing metric column: {metric}")
    d = influence_df.copy()
    d = d.sort_values(metric, ascending=False).head(int(max_bars))
    labels = d["left_out"].astype(str).tolist()
    vals = d[metric].to_numpy(dtype=float)
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)


def summary_stats_table(draws: np.ndarray) -> pd.DataFrame:
    a = np.asarray(draws, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("No finite draws")
    mean = float(np.mean(a))
    median = float(np.median(a))
    out = {
        "mean": mean,
        "median": median,
        "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p05": float(np.quantile(a, 0.05)),
        "p95": float(np.quantile(a, 0.95)),
        "mean_median_ratio": float(mean / median) if median != 0 else np.nan,
    }
    return pd.DataFrame([out])


def plot_raw_vs_adjusted_distributions(
    df: pd.DataFrame,
    *,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    needed = {"gscc", "gscc_adj", "log_gscc", "log_gscc_adj"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(needed.difference(df.columns))}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df["gscc"].to_numpy(dtype=float), bins=80, alpha=0.5, label="raw")
    axes[0].hist(df["gscc_adj"].to_numpy(dtype=float), bins=80, alpha=0.5, label="adjusted")
    axes[0].set_title(f"{title_prefix} levels histogram")
    axes[0].set_xlabel("GSCC")
    axes[0].legend()

    x1 = np.sort(df["log_gscc"].to_numpy(dtype=float))
    x2 = np.sort(df["log_gscc_adj"].to_numpy(dtype=float))
    y1 = np.linspace(0.0, 1.0, len(x1), endpoint=True)
    y2 = np.linspace(0.0, 1.0, len(x2), endpoint=True)
    axes[1].plot(x1, y1, label="raw")
    axes[1].plot(x2, y2, label="adjusted")
    axes[1].set_title(f"{title_prefix} log-space CDF")
    axes[1].set_xlabel("log(GSCC)")
    axes[1].legend()

    q = np.linspace(0.01, 0.99, 99)
    q_raw = np.quantile(df["log_gscc"].to_numpy(dtype=float), q)
    q_adj = np.quantile(df["log_gscc_adj"].to_numpy(dtype=float), q)
    axes[2].scatter(q_raw, q_adj, s=10)
    lo = float(min(q_raw.min(), q_adj.min()))
    hi = float(max(q_raw.max(), q_adj.max()))
    axes[2].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[2].set_title(f"{title_prefix} QQ (log)")
    axes[2].set_xlabel("raw quantiles")
    axes[2].set_ylabel("adjusted quantiles")

    plt.tight_layout()


def plot_lambda_stability(stability_df: pd.DataFrame, *, title_prefix: str = "") -> None:
    import matplotlib.pyplot as plt

    needed = {"lam", "structured_var_share", "r2_raw", "r2_adj", "adj_median", "adj_p95"}
    if not needed.issubset(stability_df.columns):
        raise ValueError(f"Missing columns: {sorted(needed.difference(stability_df.columns))}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    x = stability_df["lam"].to_numpy(dtype=float)

    axes[0, 0].plot(x, stability_df["structured_var_share"].to_numpy(dtype=float), marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title(f"{title_prefix} structured variance share")
    axes[0, 0].set_xlabel("lambda")

    axes[0, 1].plot(x, stability_df["r2_raw"].to_numpy(dtype=float), marker="o", label="R2 raw")
    axes[0, 1].plot(x, stability_df["r2_adj"].to_numpy(dtype=float), marker="o", label="R2 adjusted")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title(f"{title_prefix} label dependence (R2)")
    axes[0, 1].set_xlabel("lambda")
    axes[0, 1].legend()

    axes[1, 0].plot(x, stability_df["adj_median"].to_numpy(dtype=float), marker="o")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title(f"{title_prefix} adjusted median")
    axes[1, 0].set_xlabel("lambda")

    axes[1, 1].plot(x, stability_df["adj_p95"].to_numpy(dtype=float), marker="o")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(f"{title_prefix} adjusted p95")
    axes[1, 1].set_xlabel("lambda")

    plt.tight_layout()


def plot_estimand_cdfs(
    estimands: dict[str, np.ndarray],
    *,
    title: str = "GSCC estimand CDFs",
    xlabel: str = "GSCC (2015 USD / tCO2)",
    figsize: tuple[float, float] = (9, 5),
) -> None:
    """Overlay ECDFs for multiple estimand draws in a single figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for label, draws in estimands.items():
        xs = np.sort(np.asarray(draws, dtype=float))
        xs = xs[np.isfinite(xs)]
        ys = np.arange(1, xs.size + 1) / float(xs.size)
        ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_estimand_histograms(
    estimands: dict[str, np.ndarray],
    *,
    bins: int = 80,
    title: str = "GSCC estimand distributions",
    xlabel: str = "GSCC (2015 USD / tCO2)",
    figsize: tuple[float, float] = (9, 5),
) -> None:
    """Overlay density histograms for multiple estimand draws."""
    import matplotlib.pyplot as plt

    all_vals = np.concatenate(
        [np.asarray(d, dtype=float) for d in estimands.values()]
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    bin_edges = np.histogram_bin_edges(all_vals, bins=int(bins))

    fig, ax = plt.subplots(figsize=figsize)
    for label, draws in estimands.items():
        ax.hist(draws, bins=bin_edges, density=True, alpha=0.40, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Variant selection
# Same rule as lasttest.ipynb; reuses choose_best_baseline from src.utils.
# ═══════════════════════════════════════════════════════════════════════════

# Thresholds mirror those hard-coded in lasttest.ipynb.
_SCHEME_THRESHOLDS: dict[str, float] = {
    "med_pct_threshold": 0.05,
    "ks_threshold": 0.05,
    "overlap_threshold": 0.8,
}

_BASELINE_THRESHOLDS: dict[str, float] = {
    "baseline_close_ks": 0.05,
    "baseline_close_med_pct": 0.05,
    "mixed_nonhom_ks": 0.05,
    "mixed_nonhom_med_pct": 0.05,
    "between_overlap": 0.8,
}


def select_main_variant(variants: dict[str, "VariantOutputs"]) -> dict[str, Any]:
    """Select the main estimand using the same rule as lasttest.ipynb.

    Maps DEFAULT_VARIANTS onto roles:
      bootstrap_only_uniform   → "bootstrap"   (bootstrap-only, uniform)
      estimates_only_uniform   → "estimates"   (estimates-only, uniform)
      mixed_uniform            → "estimates"   (all dmgfunc, uniform)
      mixed_weighted           → "mixed"       (all dmgfunc, weighted)

    Uses ``choose_best_baseline`` from ``src.utils``, which implements the same
    pairwise KS / Δ-median / overlap comparisons documented in lasttest.ipynb.

    Returns
    -------
    dict with keys:
      selected        (variant name string)
      selected_role   ("bootstrap" | "estimates" | "mixed")
      rationale       (string)
      flags           (dict of boolean indicators)
      metrics_table   (list of dicts: pairwise metrics)
    """
    from src.utils import choose_best_baseline  # noqa: PLC0415

    v_names = [v.name for v in DEFAULT_VARIANTS]
    if len(v_names) < 3:
        first = v_names[0] if v_names else ""
        return {"selected": first, "rationale": "Only one variant; selected by default."}

    main_name, all_unif_name, bertram_name = v_names[0], v_names[1], v_names[2]

    reports = {
        "bootstrap": {"chosen_draws": variants[main_name].draws},
        "estimates": {"chosen_draws": variants[all_unif_name].draws},
        "mixed": {"chosen_draws": variants[bertram_name].draws},
    }

    decision = choose_best_baseline(reports, _BASELINE_THRESHOLDS)

    role_to_variant: dict[str, str] = {
        "bootstrap": main_name,
        "estimates": all_unif_name,
        "mixed": bertram_name,
    }
    selected_role = decision["selected_baseline"]
    selected_variant = role_to_variant[selected_role]

    return {
        "selected": selected_variant,
        "selected_role": selected_role,
        "rationale": decision["rationale"],
        "flags": decision["flags"],
        "metrics_table": decision["metrics_table"],
    }
