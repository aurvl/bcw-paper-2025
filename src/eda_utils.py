from __future__ import annotations

import json
from typing import Mapping, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from colorama import Fore, Style, init as colorama_init

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns
import statsmodels.api as sm

colorama_init(autoreset=True)


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _to_geom(val) -> BaseGeometry:
    """Convert WKT string, GeoJSON string, or shapely geometry to a shapely geometry."""
    if isinstance(val, BaseGeometry):
        return val
    try:
        return wkt.loads(val)
    except Exception:
        pass
    try:
        gj = json.loads(val)
        return shape(gj)
    except Exception:
        pass
    raise ValueError(f"Unrecognised geometry format: {type(val)}")


def build_world_geodataframe(
    shp_path: Union[str, Path],
    usa_split_csv: Union[str, Path],
) -> gpd.GeoDataFrame:
    """
    Load and clean the world shapefile, then replace the single USA polygon
    with the Alaska / contiguous-US split from *usa_split_csv*.

    Returns a GeoDataFrame with columns: ISO_A3, NAME, CONTINENT, geometry.
    """
    # 1) World shapefile, drop Antarctica, fix a few ISO codes
    wld = gpd.read_file(shp_path)
    wld = wld[wld['ISO_A3'] != 'ATA'].copy()
    wld.loc[wld['NAME'] == 'France',          'ISO_A3'] = 'FRA'
    wld.loc[wld['NAME'] == 'Somaliland',      'ISO_A3'] = 'SOM'
    wld.loc[wld['NAME'] == 'Northern Cyprus', 'ISO_A3'] = 'CYP'
    wld.loc[wld['NAME'] == 'Norway',          'ISO_A3'] = 'NOR'
    wld = wld[['ISO_A3', 'NAME', 'CONTINENT', 'geometry']]

    # 2) USA split (Alaska separate)
    saal = pd.read_csv(usa_split_csv)[['ISO_A3', 'NAME', 'CONTINENT', 'geometry']].copy()
    saal['geometry'] = saal['geometry'].apply(_to_geom)
    saal.loc[saal['NAME'] == 'Alaska', 'ISO_A3'] = 'US-AK'
    saal.loc[saal['ISO_A3'] == 'USA',  'NAME']   = 'United States'
    saal = gpd.GeoDataFrame(saal, geometry='geometry', crs=wld.crs)

    # 3) Merge: remove the single USA row, add the two split rows
    world = pd.concat(
        [wld[wld['ISO_A3'] != 'USA'], saal],
        ignore_index=True,
    )
    return gpd.GeoDataFrame(world, geometry='geometry', crs=wld.crs)


def _categorize(val: float, which: str = 'BCE') -> str:
    """Return the display category label for a sequestration value."""
    if which == 'BCE':
        if   0      <  val <  6e5:  
            return '0 – 0.6 MtC'
        elif 6e5   <= val < 12e5:   
            return '0.6 – 1.2 MtC'
        elif 12e5  <= val < 14e5:   
            return '1.2 – 1.4 MtC'
        elif val >= 14e5:           
            return '> 1.4 MtC'
        else:                       
            return 'No data available'
    elif which == 'BCP':
        if   0     <  val <  25e6:  
            return '0 – 25 MtC'
        elif 25e6 <= val <  4e7:    
            return '25 – 40 MtC'
        elif 4e7  <= val <  5e7:    
            return '40 – 50 MtC'
        elif val >= 5e7:            
            return '> 50 MtC'
        else:                       
            return 'No data available'
    else:
        raise ValueError("which must be 'BCE' or 'BCP'.")


def build_country_data(
    world: gpd.GeoDataFrame,
    plot_data: pd.DataFrame,
    var: str,
    which: str = 'BCE',
) -> gpd.GeoDataFrame:
    """
    Join *world* with *plot_data* on ISO, add a category column, and return
    a clean GeoDataFrame ready for plotting.

    Parameters
    ----------
    world     : output of build_world_geodataframe()
    plot_data : DataFrame with an 'ISO' column and at least *var*
    var       : column in plot_data to categorise (e.g. 'tot_uptake (tC)')
    which     : 'BCE' or 'BCP' — drives the category thresholds
    """
    dw = plot_data.copy()
    dw['cat'] = dw[var].apply(_categorize, which=which)

    country_data = (
        world.set_index('ISO_A3')
        .join(dw.set_index('ISO'))
        [['NAME', 'CONTINENT', 'geometry', var, 'cat']]
        .reset_index()
        .rename(columns={'ISO_A3': 'ISO'})
    )
    country_data = country_data.dropna(subset=['cat'])
    # keep only valid / non-empty geometries
    country_data = country_data[
        country_data.geometry.notnull() & country_data.geometry.is_valid
    ]
    return gpd.GeoDataFrame(country_data, geometry='geometry', crs=world.crs)


def complete_iso_and_continent(
    data: pd.DataFrame,
    iso_map: Mapping[str, str],
    continent_map: Mapping[str, str],
    *,
    country_col: str = "country_name",
    iso_col: str = "ISO",
    continent_col: str = "Continent",
    overrides: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    1) Creates a unique ISO for duplicates (non-NaN ISO) by suffixing -1, -2, ...
    2) Fills missing ISO values using iso_map (without suffix)
    3) Fills Continent (for rows where ISO was originally missing) using continent_map
    4) Replaces ISO with the unique ISO
    5) Applies final overrides
    """
    df = data.copy()

    # 1) Unique ISO for duplicates (only non-NaN ISO)
    df["_iso_unique"] = df[iso_col]
    mask_dup = df[iso_col].notna() & df.duplicated(iso_col, keep=False)
    if mask_dup.any():
        df.loc[mask_dup, "_iso_unique"] = (
            df.loc[mask_dup, iso_col].astype(str)
            + "-"
            + (df.loc[mask_dup].groupby(iso_col).cumcount() + 1).astype(str)
        )

    # 2) Complete rows without ISO (based on originally missing ISO)
    mask_iso_nan = df[iso_col].isna()
    if mask_iso_nan.any():
        # ISO from iso_map
        filled_iso = df.loc[mask_iso_nan, country_col].map(dict(iso_map))
        df.loc[mask_iso_nan, iso_col] = filled_iso

        # no suffix for these rows
        df.loc[mask_iso_nan, "_iso_unique"] = df.loc[mask_iso_nan, iso_col]

        # Continent from continent_map
        filled_cont = df.loc[mask_iso_nan, country_col].map(dict(continent_map))
        df.loc[mask_iso_nan, continent_col] = df.loc[mask_iso_nan, continent_col].where(
            filled_cont.isna(), filled_cont
        )

    # 3) ISO = iso_unique then cleanup
    df[iso_col] = df["_iso_unique"]
    df.drop(columns=["_iso_unique"], inplace=True)

    # 4) Final overrides
    if overrides:
        for cname, iso in overrides.items():
            df.loc[df[country_col] == cname, iso_col] = iso

    return df


def _safe_arr(series: pd.Series) -> np.ndarray:
    """Retourne un numpy array float en remplaçant les non numériques par np.nan."""
    try:
        return series.to_numpy(dtype=float)
    except Exception:
        return np.array([], dtype=float)

TITLE_W = 44   # largeur colonne bloc (nom + unit) — plus compacte
LBL_W   = 6    # largeur colonne label (std/mean/total)
VAL_W   = 22   # largeur colonne valeur (droite, suffisante pour "val ± se")
LINE_W  = TITLE_W + LBL_W + VAL_W + 6

# échelles demandées
SCALE_MAP = {
    "MtC": 1e6,
    "GtC": 1e9,
    "billion US$": 1e9,
    "trillion US$": 1e12
}

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

CONTINENT_PAL = {
    'Asia': '#ef233c',
    'Europe': '#ffea00',
    'Africa': '#4cc9f0',
    'Oceania': '#00509d',
    'Antarctica': '#6c757d',
    'Americas': '#bfd200',
    'Caribbean': '#80b918',
}

BCE_COLORMAP = {
    'No data available': '#D3D3D3',
    '0 – 0.6 MtC': '#ECFF90',
    '0.6 – 1.2 MtC': '#7EFD7E',
    '1.2 – 1.4 MtC': '#016D01',
    '> 1.4 MtC': '#0B3E36F0',
}

BCP_COLORMAP = {
    'No data available': '#D3D3D3',
    '0 – 25 MtC': '#ADE8F4',
    '25 – 40 MtC': '#00B4D8',
    '40 – 50 MtC': '#0077B6',
    '> 50 MtC': '#03045E',
}

GROUPS_COLORMAP = {
    'SIDS': 'tab:blue',
    'Developing economies': 'tab:orange',
    'Developed economies': 'tab:green',
    'LDCs': 'tab:red',
}

def _safe_series(data, col):
    return data.get(col, pd.Series(dtype=float))

def _to_num(v):
    try:
        vv = float(v)
    except Exception:
        return None
    if not np.isfinite(vv):
        return None
    return vv

def _fmt_num(v):
    """Format number with 2 decimals and thousands separator, or placeholder."""
    if v is None:
        return "-" * VAL_W
    s = f"{v:,.2f}"
    # if too long, keep right-most characters (preserve decimals)
    return s.rjust(VAL_W) if len(s) <= VAL_W else s[-VAL_W:]

def pretty_report(data: pd.DataFrame, use_color: bool = True):
    """
    Compact 3-col report with exact titles/units:
    - col1: block title with unit in parentheses (e.g. "Coastal BCEs Sequestration (MtC)")
    - col2: label (std:, mean:, total:)
    - col3: value with 2 decimals; totals show "val ± SE" if SE exists (no units in values)
    """
    if use_color:
        H = Fore.CYAN + Style.BRIGHT
        K = Fore.YELLOW + Style.BRIGHT
        V = Fore.GREEN + Style.BRIGHT
        R = Style.RESET_ALL
    else:
        H = K = V = R = ""
    
    sep = "─" * LINE_W
    print()
    print(H + "Blue Carbon Summary".center(LINE_W) + R)
    print(sep)

    # Blocks: (df_col, display_title_with_unit, scale_key, se_source_column_or_None)
    blocks = [
        ("uptake_total_mean", "Coastal BCEs Sequestration (MtC)", "MtC", "uptake_total_se"),
        ("cBCW", "Coastal BC Wealth (billion US$)", "billion US$", "cBCW_se"),
        ("BCP Seq (tC)", "Blue Carbon Pump Sequestration (GtC)", "GtC", None),  # SE computed below if desired
        ("oBCW", "Blue Carbon Pump Wealth (trillion US$)", "trillion US$", "oBCW_se"),
        ("Total BCseq", "Total Blue Carbon Sequestration (GtC)", "GtC", "Total BCseq_se"),
        ("Total BCW", "Total Blue Carbon Wealth (trillion US$)", "trillion US$", "Total BCW_se"),
    ]

    # compute SEs where applicable (use same logic as before)
    se_cache = {}
    # direct sum-of-sq approach for per-row SE columns
    for key in ["uptake_total_se", "cBCW_se", "oBCW_se", "Total BCseq_se", "Total BCW_se"]:
        arr = _safe_arr(data.get(key, pd.Series(dtype=float)))
        se_cache[key] = float(np.sqrt(np.nansum(arr ** 2))) if arr.size else None

    # special SE for 'BCP Seq (tC)' as earlier (std * sqrt(count)) if present
    try:
        bcp_series = _safe_series(data, "BCP Seq (tC)")
        se_cache["BCP Seq (tC)"] = float(bcp_series.std(ddof=1) * np.sqrt(int(bcp_series.count()))) if getattr(bcp_series, "size", 0) else None
    except Exception:
        se_cache["BCP Seq (tC)"] = None

    def get_series(col):
        return _safe_series(data, col)

    # print helper for each block
    for df_col, disp_title, scale_key, se_colname in blocks:
        series = get_series(df_col)
        std_v = _to_num(series.std(ddof=1))
        mean_v = _to_num(series.mean())
        sum_v = _to_num(series.sum())

        # get SE for totals from provided mapping if available
        se_total = None
        if se_colname:
            se_total = _to_num(se_cache.get(se_colname))
        else:
            se_total = _to_num(se_cache.get(df_col))  # fallback for BCP

        # scale factor
        scale = SCALE_MAP.get(scale_key, 1.0)

        # formatted strings (values shown WITHOUT units)
        std_str = _fmt_num(std_v / scale) if std_v is not None else _fmt_num(None)
        mean_str = _fmt_num(mean_v / scale) if mean_v is not None else _fmt_num(None)

        # total: show "val ± SE" if se_total exists, else show val only
        if sum_v is None:
            total_str = _fmt_num(None)
        else:
            val_display = f"{(sum_v/scale):,.2f}"
            if se_total is not None:
                se_display = f"{(se_total/scale):,.2f}"
                combined = f"{val_display} ± {se_display}"
                total_str = combined.rjust(VAL_W) if len(combined) <= VAL_W else combined[-VAL_W:]
            else:
                total_str = f"{(sum_v/scale):,.2f}".rjust(VAL_W)

        # title centered vertically on the middle line (line index 1)
        title_lines = ["", disp_title, ""]
        labels = ["std:", "mean:", "total:"]

        for i in range(3):
            left = title_lines[i].ljust(TITLE_W)
            lbl = labels[i].rjust(LBL_W)
            if i == 0:
                val = std_str
            elif i == 1:
                val = mean_str
            else:
                val = total_str
            print(f"{K}{left}{R} {lbl} {V}{val}{R}")

        print(sep)

    print()
    
    
def blue_carbon_report(sum_salt, sum_seag, sum_mang, sum_plank,
                       glob_share_salt, glob_share_seag, glob_share_mang, glob_share_plank,
                       bce_share_salt, bce_share_seag, bce_share_mang,
                       total_sum, SE, cmol, GSCC):

    # ---------- helpers ----------
    def MtC(x): return x / 1e6
    def GtC(x): return x / 1e9
    def BUSD(x): return (x / 1e9) * cmol * GSCC

    # ---------- BCE total ----------
    sum_bce = sum_salt + sum_seag + sum_mang

    # ---------- text ----------
    text = f"""
BLUE CARBON SEQUESTRATION REPORT
--------------------------------
******************************
COASTAL BLUE CARBON ECOSYSTEMS
******************************

Salt marshes :
  {MtC(sum_salt):.2f} MtC yr⁻¹
  {glob_share_salt:.1f}% of global blue carbon ({bce_share_salt:.1f}% of BCEs)
  ≈ {BUSD(sum_salt):.2f} billion US$ yr⁻¹

Seagrasses :
  {MtC(sum_seag):.2f} MtC yr⁻¹
  {glob_share_seag:.1f}% of global blue carbon ({bce_share_seag:.1f}% of BCEs)
  ≈ {BUSD(sum_seag):.2f} billion US$ yr⁻¹

Mangroves :
  {MtC(sum_mang):.2f} MtC yr⁻¹
  {glob_share_mang:.1f}% of global blue carbon ({bce_share_mang:.1f}% of BCEs)
  ≈ {BUSD(sum_mang):.2f} billion US$ yr⁻¹

Combined coastal ecosystems :
  {MtC(sum_bce):.2f} MtC yr⁻¹
  {glob_share_salt + glob_share_seag + glob_share_mang:.0f}% of global blue carbon
  ≈ {BUSD(sum_bce):.2f} billion US$ yr⁻¹

******************************
BIOLOGICAL CARBON PUMP
******************************

Phytoplankton export :
  {GtC(sum_plank):.2f} GtC yr⁻¹
  {glob_share_plank:.0f}% of global blue carbon
  ≈ {BUSD(sum_plank):.2f} billion US$ yr⁻¹

******************************
TOTAL BLUE CARBON
******************************

Total sequestration :
  {GtC(total_sum):.3f} ± {GtC(SE):.3f} GtC yr⁻¹

Estimated global value :
  ≈ {BUSD(total_sum):.2f} billion US$ yr⁻¹
"""

    print(text)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def format_log_axis_plain(axis='y'):
    """Format a matplotlib log-scale axis to show plain (non-scientific) tick labels."""
    ax = plt.gca()
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.ticklabel_format(axis='y', style='plain')
    elif axis == 'x':
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        plt.ticklabel_format(axis='x', style='plain')


def rgb_to_hex(rgb):
    """Convert an (r, g, b) tuple with values in [0, 1] to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )


def categorize_bcw_pcap(i):
    """Categorize a BCW per-capita value (in US$) into display bins."""
    if 0 <= i < 5:
        return '0 - 5 US$'
    elif 5 <= i < 50:
        return '5 - 50 US$'
    elif 50 <= i < 500:
        return '50 - 500 US$'
    elif 500 <= i:
        return '> 500 US$'
    else:
        return 'No data available'


def circle_area_plot(
    ecosystem_totals: pd.Series, diameters: list = None, colors: list = None, label_colors: list = None
) -> None:
    """
    Plot a circle area diagram based on values from a pandas Series.

    Args:
        ecosystem_totals (pd.Series): Series where index are labels (e.g. ecosystem names) and values are numeric (e.g. carbon uptake).
        diameters (list, optional): Diameters for the circles (must match the length of ecosystem_totals). If None, scaled automatically.
        colors (list, optional): Fill colors for the circles.
        label_colors (list, optional): Text label colors for ecosystem names.

    Returns:
        None
    """

    try:
        n = len(ecosystem_totals)
        if n == 0:
            print("Nothing to plot: empty series.")
            return

        # Default diameters based on relative size if not provided
        if diameters is None:
            max_val = ecosystem_totals.max()
            diameters = [4 + 16 * (val / max_val)**0.5 for val in ecosystem_totals]  # square root scaling

        # Fallback colors if not enough provided
        if colors is None or len(colors) < n:
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum', 'skyblue']
            colors = (colors * (n // len(colors) + 1))[:n]

        if label_colors is None or len(label_colors) < n:
            label_colors = ['black'] * n

        labels = ecosystem_totals.index.tolist()
        uptake_values = ecosystem_totals.values.tolist()

        # Positioning: even spacing based on max diameter
        circle_radii = [d / 2 for d in diameters]
        border_spacing = 1.5

        # Calcul des positions des centres des cercles
        positions = []
        current_x = 0

        for i, radius in enumerate(circle_radii):
            if i == 0:
                current_x = radius
            else:
                previous_radius = circle_radii[i - 1]
                current_x += previous_radius + radius + border_spacing
            positions.append(current_x)
        # Create plot
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(3 * n, 12))
        for x, diameter, color, text_color, label, value in zip(positions, diameters, colors, label_colors, labels, uptake_values):
            circle = plt.Circle((x, 5), diameter / 2, color=color, alpha=0.7)
            ax.add_artist(circle)
            font_size = max(15, min(25, diameter * 2.5))
            if value < 1e3:
                ax.text(x, 5, f"{value:.2f}\nMtC yr⁻¹", fontsize=font_size,
                        ha='center', va='center', color='black')
            else:
                ax.text(x, 5, f"{value/1e3:.2f}\nGtC yr⁻¹", fontsize=font_size,
                        ha='center', va='center', color='black')
            ax.text(x, 5 - diameter / 2 - 0.7, label, fontsize=19, ha='center', va='center', color=text_color)

        ax.set_aspect('equal')
        ax.set_xlim(-2, max(positions) + border_spacing * 4)
        ax.set_ylim(-10, 18)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


def regplotter(dtf, var, varname, title, xlab, yvar='cBCW', col="blue"):
    """Log-log scatter + OLS regression using log-transformed columns."""
    df = dtf.copy()
    df = df[(df[var] > 0) & (df[yvar] > 0)].copy()
    df[varname] = np.log(df[var])
    df['log_cBCW'] = np.log(df[yvar])
    df = df.dropna(subset=[varname, 'log_cBCW'])
    X = df[varname]
    y = df['log_cBCW']
    X = sm.add_constant(X)  # intercept

    model = sm.OLS(y, X).fit()
    a_hat = model.params['const']   # Intercept (a^)
    b_hat = model.params[varname]   # Coefficient (b^)
    R2 = model.rsquared             # Coefficient R²

    equation = f"$y = {a_hat:.2f} + {b_hat:.2f}x$ | $R² = {R2:.2f}$"
    print(f"Regression equation: {equation}")
    plt.figure(figsize=(10, 4))
    sns.scatterplot(data=df, x=varname, y='log_cBCW', alpha=0.6, color=col)
    sns.regplot(x=varname, y='log_cBCW', data=df, scatter=False, color='black', line_kws={"linewidth": 2})
    plt.xlabel(f'log({xlab})')
    plt.ylabel(f'log({yvar})')
    plt.title(title, loc='left', fontweight='bold', fontsize=14, pad=14)
    plt.legend(labels=['Data point', f'Equation: {equation}'], loc='best')
    plt.grid(alpha=0.3)
    sns.despine()
    plt.show()


def regplotter_log_axes(dtf, var, varname, title, xlab, yvar='BCW'):
    """Scatter + OLS regression displayed on log-log axes (data not pre-transformed)."""
    df = dtf.copy()
    df = df[(df[var] > 0) & (df[yvar] > 0)].copy()
    df = df.dropna(subset=[var, yvar])

    X = np.log(df[var])
    y = np.log(df[yvar])
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    a_hat = model.params['const']
    b_hat = model.params[var]
    R2 = model.rsquared

    equation = f"$y = {a_hat:.2f} + {b_hat:.2f}x$ | $R² = {R2:.2f}$"
    print(f"Regression equation: {equation}")

    plt.figure(figsize=(10, 4))
    sns.scatterplot(data=df, x=var, y=yvar, alpha=0.6)
    sns.regplot(data=df, x=var, y=yvar, scatter=False, color='red', line_kws={"linewidth": 2})
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(xlab)
    plt.ylabel(yvar)
    plt.title(title, fontweight='bold', fontsize=14, pad=14)
    plt.legend(labels=['Data point', f'Equation: {equation}'], loc='best')
    plt.grid(alpha=0.3, which='both')
    sns.despine()
    plt.show()


def mapper_bce(dtfm: pd.DataFrame, var: str, ecosystem: str, palet: dict,
               title: str, subtitle: str, wld: gpd.GeoDataFrame) -> None:
    """
    Display a world choropleth for a BCE variable, colouring countries by
    ecosystem-specific sequestration bins.

    Parameters
    ----------
    dtfm      : DataFrame with columns 'ISO' and `var`
    var       : column to categorise
    ecosystem : 'Mangroves', 'Saltmarshes', 'Seagrasses', 'areas', or 'Total BCW uptake'
    palet     : matplotlib colormap name (str) used by generate_col_map
    title     : main figure title
    subtitle  : sub-title line
    wld       : world GeoDataFrame from build_world_geodataframe()
    """

    def categorize_bce(value: float, ecosystem: str) -> str:
        """Categorise a value according to the specified ecosystem's thresholds."""
        if ecosystem == 'Mangroves':
            if 0 < value < 2e5:
                return '0 – 0.2 MtC'
            elif 2e5 <= value < 4e5:
                return '0.2 – 0.4 MtC'
            elif 4e5 <= value < 8e5:
                return '0.4 – 0.8 MtC'
            elif value >= 8e5:
                return '> 0.8 MtC'
            else:
                return 'No data available'
        elif ecosystem == 'Saltmarshes':
            if 0 < value < 1e5:
                return '0 – 0.1 MtC'
            elif 1e5 <= value < 2e5:
                return '0.1 – 0.2 MtC'
            elif 2e5 <= value < 5e5:
                return '0.2 – 0.5 MtC'
            elif value >= 5e5:
                return '> 0.5 MtC'
            else:
                return 'No data available'
        elif ecosystem == 'Seagrasses':
            if 0 < value < 1e5:
                return '0 – 0.1 MtC'
            elif 1e5 <= value < 1e6:
                return '0.1 – 1 MtC'
            elif 1e6 <= value < 2e6:
                return '1 – 2 MtC'
            elif value >= 2e6:
                return '> 2 MtC'
            else:
                return 'No data available'
        elif ecosystem == 'areas':
            if 0 < value < 1e3:
                return '0 – 1,000 km²'
            elif 1e3 <= value < 2e3:
                return '1,000 – 2,000 km²'
            elif 2e3 <= value < 5e3:
                return '2,000 – 5,000 km²'
            elif value >= 5e3:
                return '> 5,000 km²'
        elif ecosystem == 'Total BCW uptake':
            if 0 < value < 5e6:
                return '0 - 5 MtC'
            elif 5e6 <= value < 10e6:
                return '5 – 10 MtC'
            elif 10e6 <= value < 30e6:
                return '10 – 30 MtC'
            elif value >= 30e6:
                return '> 30 MtC'
        return 'No data available'

    def generate_col_map(df: pd.DataFrame, palette: str = 'Greens') -> dict:
        """Generate a color dict for each category in df['cat']."""
        categories = sorted(df['cat'].dropna().unique().tolist())
        if 'No data available' in categories:
            categories.remove('No data available')
        categories = ['No data available'] + categories

        cmap = plt.get_cmap(palette)
        colors = [mcolors.to_hex(cmap(i / max(len(categories) - 2, 1))) for i in range(len(categories) - 1)]

        col_map = {'No data available': '#D3D3D3'}
        for cat, color in zip(categories[1:], colors):
            col_map[cat] = color
        return col_map

    # Chargement des frontières pays
    world = wld[['ISO_A3', 'NAME', 'CONTINENT', 'geometry']]
    world = world[world['NAME'] != 'Antarctica']

    # Catégorisation de la variable
    df = dtfm.copy()
    df['cat'] = df[var].apply(lambda x: categorize_bce(x, ecosystem=ecosystem))

    # Jointure avec la carte
    country_data = world.set_index('ISO_A3').join(df.set_index('ISO'))
    country_data = country_data[['NAME', 'CONTINENT', 'geometry', var, 'cat']].reset_index()
    country_data = country_data.rename(columns={'ISO_A3': 'ISO'})
    country_data['cat'] = country_data['cat'].fillna('No data available')

    # Affichage de la carte
    fig, ax = plt.subplots(figsize=(10, 8))
    legend_elements = []
    col_map = generate_col_map(country_data, palette=palet)

    for category, color in col_map.items():
        subset = country_data[country_data['cat'] == category]
        subset = subset[subset.geometry.is_valid & subset.geometry.notnull()]
        if not subset.empty:
            subset.plot(ax=ax, color=color, label=category, edgecolor='black', linewidth=0.5)
            legend_elements.append(Patch(color=color, label=category))

    fig.text(0.05, 0.8, title, fontsize=24, fontweight='bold', va='top', ha='left')
    fig.text(0.051, 0.76, subtitle, fontsize=15, va='top', ha='left')

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.1),
        loc='lower center',
        fontsize=12,
        frameon=False,
        ncol=3
    )

    ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def map_and_barplot(df: pd.DataFrame,
                    var: str,
                    group_col: str,
                    thresholds: list,
                    labels: list,
                    colors: list,
                    title: str,
                    subtitle: str,
                    xlabel_bar: str,
                    fig_size_map=(14, 10),
                    fig_size_bar=(5, 2),
                    cmap_palette='Blues',
                    wld=None) -> None:
    """
    Display a choropleth map and a horizontal bar chart for a continuous variable.

    Parameters
    ----------
    df          : DataFrame with columns ['ISO', var, group_col]
    var         : column to map
    group_col   : column used for bar-chart grouping
    thresholds  : ascending break-points for category bins
    labels      : category labels (len == len(thresholds) + 1)
    colors      : hex colors per category (same order as labels)
    title       : main map title
    subtitle    : map sub-title
    xlabel_bar  : x-axis label for the bar chart
    fig_size_map: figure size for the map
    fig_size_bar: figure size for the bar chart
    cmap_palette: seaborn palette for the bar chart
    wld         : world GeoDataFrame from build_world_geodataframe()
    """
    if wld is None:
        raise ValueError("wld (world GeoDataFrame) must be provided.")

    # 1. Catégorisation
    def categorize(value):
        if pd.isna(value):
            return 'No data available'
        for i, t in enumerate(thresholds):
            if value < t:
                return labels[i]
        return labels[-1]

    df = df.copy()
    df['cat'] = df[var].apply(categorize)

    # 2. Charger la carte
    world = wld[['ISO_A3', 'NAME', 'CONTINENT', 'geometry']]
    world = world[world['ISO_A3'] != 'ATA']  # Exclure l'Antarctique

    # 3. Fusionner
    by_country = df[['ISO', var, 'cat', group_col]]
    country_data = world.set_index('ISO_A3').join(by_country.set_index('ISO'))
    country_data = country_data.reset_index().rename(columns={'ISO_A3': 'ISO'})
    country_data['cat'] = country_data['cat'].fillna('No data available')

    # 4. Barplot moyen par groupe
    sq = df.replace(0, np.nan).groupby(group_col)[var].mean().reset_index().dropna().sort_values(by=var, ascending=False)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=fig_size_bar)
    sns.barplot(data=sq, x=var, y=group_col, palette=cmap_palette, width=0.8, ci=False, edgecolor='black')
    plt.xlabel(xlabel_bar)
    plt.ylabel('')
    plt.xscale("log")
    format_log_axis_plain('x')
    sns.despine()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.grid(True, axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()

    # 5. Carte
    fig, ax = plt.subplots(figsize=fig_size_map)
    legend_patches = []

    color_map = dict(zip(labels, colors))
    color_map['No data available'] = '#d3d3d3'

    for cat, color in color_map.items():
        subset = country_data[country_data['cat'] == cat]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.3)
            legend_patches.append(Patch(color=color, label=cat))

    fig.text(0.05, 0.83, title, fontsize=27, fontweight='bold', va='top', ha='left')
    fig.text(0.051, 0.794, subtitle, fontsize=18, va='top', ha='left')

    plt.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.1),
               loc='lower center', ncol=3, fontsize=13,
               title='', frameon=False)

    ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def wind_rose(df: pd.DataFrame, segment_col: str = 'Segment', label_col: str = 'Label',
              value_col: str = 'Value', category_col: str = None, n_top: int = 10,
              category_colors: dict = None, figsize: tuple = (18, 18)) -> None:
    """
    Generate a radial bar chart (wind-rose style) showing the top n entities per
    segment, coloured by category.

    Parameters
    ----------
    df              : DataFrame with the data
    segment_col     : column for the outer segments (e.g. continent)
    label_col       : column with entity labels
    value_col       : numeric column to plot (bar height)
    category_col    : column used for bar colours
    n_top           : number of top entities per segment
    category_colors : dict mapping category → colour
    figsize         : figure size in inches
    """
    continents = df[segment_col].unique()
    n_segments = len(continents)
    angles = np.linspace(0, 2 * np.pi, n_segments + 1, endpoint=True)

    # Couleurs automatiques si category_colors n'est pas fourni
    if category_col is not None:
        unique_categories = df[category_col].unique()
        if category_colors is None:
            cmap = plt.get_cmap('tab10')
            category_colors = {cat: cmap(i % 10) for i, cat in enumerate(unique_categories)}
    else:
        category_colors = None

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    sns.set_theme(style="whitegrid")

    segment_width = 2 * np.pi / n_segments
    ax.set_theta_offset(np.pi / 2 - segment_width / 2)

    # Lignes radiales (séparation des continents)
    for angle in angles[:-1]:
        ax.axvline(x=angle, color='black', lw=2)

    width = (2 * np.pi / n_segments) / n_top  # Largeur de chaque barre

    for i, segment in enumerate(continents):
        angles_i = np.linspace(angles[i], angles[i + 1], n_top + 1)
        segment_data = df[df[segment_col] == segment].sort_values(by=value_col, ascending=False).head(n_top)

        for j in range(len(segment_data)):
            row = segment_data.iloc[j]
            label = row[label_col]
            value = row[value_col]
            if category_col is not None:
                category = row[category_col]
                color = category_colors.get(category, '#333333')
            else:
                color = '#1f77b4'  # couleur par défaut

            ax.bar(
                angles_i[j] + width / 2,
                value,
                width=width,
                bottom=0,
                color=color,
                alpha=0.7,
                edgecolor='white',
                linewidth=2
            )
            ax.text(
                angles_i[j] + width / 2,
                value + 1,
                f"{label}",
                fontsize=15,
                fontweight='bold',
                ha='center',
                va='bottom'
            )

    ax.set_yscale("symlog")
    ax.set_xticks(angles[:-1] + (angles[1] - angles[0]) / 2)
    ax.set_xticklabels(continents, va='bottom', fontsize=25, fontweight='bold')
    ax.grid(axis='x', visible=False)

    # Légende
    if category_col is not None:
        legend_handles = [
            mpatches.Patch(color=category_colors[cat], label=cat)
            for cat in unique_categories
        ]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=15)
    ax.grid(True, linestyle="-", alpha=0.7, axis='y')

    plt.show()


def stats_for_group(df: pd.DataFrame, group_name: str,
                    num_cols: list, conversion_factors: dict) -> pd.DataFrame:
    """
    Compute mean / median / std for numeric columns in df, applying unit conversions.

    Parameters
    ----------
    df                  : country-level DataFrame for the group
    group_name          : label to insert in the 'Groups' column
    num_cols            : list of column names to summarise
    conversion_factors  : dict mapping column name → multiplicative factor

    Returns
    -------
    DataFrame with columns: Groups, Number of countries, Stat, <num_cols...>
    """
    df_conv = df.copy()
    for col, factor in conversion_factors.items():
        if col in df_conv:
            df_conv[col] = df_conv[col] * factor

    n_countries = df_conv['ISO'].nunique()

    stats = df_conv[num_cols].agg(['mean', 'median', 'std']).round(3)
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'Stat'}, inplace=True)

    stats.insert(0, 'Groups', group_name)
    stats.insert(1, 'Number of countries', n_countries)
    return stats


# ---------------------------------------------------------------------------
# EEZ choropleth
# ---------------------------------------------------------------------------

def plot_eez_choropleth(
    eez_gdf,
    data_df=None,
    value_col="Total BCEs Area (km\u00b2)",
    merge_on="TERRITORY1",
    world_gdf=None,
    bins=None,
    palette="cubehelix",
    projection="ESRI:54030",
    figsize=(14, 6),
    linewidth=0.2,
    missing_color="#f0f0f0",
    legend_fontsize=11,
    show_legend=True,
    title="",
    dissolve_by=None,
    clip_land=False,
):
    """
    Plot a choropleth over EEZ polygons only (not land).

    Only EEZ polygons are colored (not land). The world/land layer is drawn as a
    flat grey background at a lower z-order; no value shading is ever applied to it.

    Parameters
    ----------
    eez_gdf      : GeoDataFrame or path (str/Path) to EEZ shapefile
    data_df      : DataFrame that contains *merge_on* and *value_col* (used when
                   *value_col* is absent from the EEZ GeoDataFrame)
    value_col    : column to colour the EEZ polygons by
    merge_on     : join key between EEZ and data_df (default "TERRITORY1")
    world_gdf    : optional GeoDataFrame or path to land boundaries (background only)
    bins         : bin edges for discrete classification; continuous colorbar if None
    palette      : matplotlib / seaborn colormap name
    projection   : display CRS; source data kept at EPSG:4326 (default Robinson)
    figsize      : figure size tuple
    linewidth    : EEZ polygon edge width
    missing_color: fill for EEZ rows with no data
    legend_fontsize: font size for legend labels
    show_legend  : draw legend or not
    title        : optional figure title
    dissolve_by  : if set to the same value as *merge_on*, dissolve EEZ polygons
                   by that column before plotting (aggregates multi-row territories)
    clip_land    : if True and world_gdf provided, subtract land polygons from EEZ
                   display layer so only truly maritime areas are colored

    Returns
    -------
    fig, ax, eez_disp
        eez_disp is the reprojected GeoDataFrame used for plotting (inspect .crs,
        .total_bounds, or merged values directly from the caller).
    """
    # ------------------------------------------------------------------ #
    # 1. Load EEZ geodataframe (work on a copy – never modify the original)
    # ------------------------------------------------------------------ #
    if isinstance(eez_gdf, (str, Path)):
        eez_path = Path(eez_gdf)
        if not eez_path.exists():
            raise FileNotFoundError(
                f"EEZ shapefile not found: {eez_path}\n"
                "Check that the file exists under data_source/shp/eez/."
            )
        eez = gpd.read_file(eez_path)
    else:
        eez = eez_gdf.copy()

    # ------------------------------------------------------------------ #
    # 2. CRS – assume EPSG:4326 if missing
    # ------------------------------------------------------------------ #
    if eez.crs is None:
        eez = eez.set_crs("EPSG:4326")

    # ------------------------------------------------------------------ #
    # 3. Merge tabular data if value_col is absent from the GeoDataFrame
    # ------------------------------------------------------------------ #
    if value_col not in eez.columns:
        if (
            data_df is not None
            and merge_on in data_df.columns
            and value_col in data_df.columns
        ):
            eez = eez.merge(
                data_df[[merge_on, value_col]].drop_duplicates(subset=[merge_on]),
                left_on=merge_on,
                right_on=merge_on,
                how="left",
            )
        else:
            print(
                f"[plot_eez_choropleth] WARNING: '{value_col}' not found in EEZ "
                "GeoDataFrame and could not be merged from data_df. "
                "All EEZ polygons will be shown as 'No data'."
            )
            eez[value_col] = np.nan

    # ------------------------------------------------------------------ #
    # 4. Optional dissolve (aggregate multi-row territories before reproject)
    # ------------------------------------------------------------------ #
    if dissolve_by is not None and dissolve_by in eez.columns:
        numeric_cols = eez.select_dtypes(include="number").columns.tolist()
        agg = {c: "sum" for c in numeric_cols}
        eez = eez.dissolve(by=dissolve_by, aggfunc=agg).reset_index()

    # ------------------------------------------------------------------ #
    # 5. Fix invalid geometries before reprojection
    # ------------------------------------------------------------------ #
    invalid = ~eez.geometry.is_valid
    if invalid.any():
        eez.loc[invalid, "geometry"] = eez.loc[invalid, "geometry"].buffer(0)

    # ------------------------------------------------------------------ #
    # 6. Reproject for display only – original `eez` stays in EPSG:4326
    # ------------------------------------------------------------------ #
    try:
        eez_disp = eez.to_crs(projection)
    except Exception as exc:
        raise RuntimeError(
            f"[plot_eez_choropleth] Reprojection of EEZ to {projection!r} failed: {exc}"
        ) from exc

    # ------------------------------------------------------------------ #
    # 7. Load and reproject land background (never colored by value)
    # ------------------------------------------------------------------ #
    world_disp = None
    if world_gdf is not None:
        if isinstance(world_gdf, (str, Path)):
            wgdf = gpd.read_file(Path(world_gdf))
        else:
            wgdf = world_gdf.copy()
        if wgdf.crs is None:
            wgdf = wgdf.set_crs("EPSG:4326")
        try:
            world_disp = wgdf.to_crs(projection)
        except Exception as exc:
            print(
                f"[plot_eez_choropleth] WARNING: land background reprojection failed "
                f"({exc}). Skipping background."
            )
            world_disp = None

    # ------------------------------------------------------------------ #
    # 8. Optional: clip land from EEZ display layer
    # ------------------------------------------------------------------ #
    if clip_land and world_disp is not None:
        try:
            land_union = world_disp.geometry.unary_union
            eez_disp = eez_disp.copy()
            eez_disp["geometry"] = eez_disp.geometry.difference(land_union)
        except Exception as exc:
            print(
                f"[plot_eez_choropleth] WARNING: clip_land difference failed ({exc}). "
                "Proceeding without land clipping."
            )

    # ------------------------------------------------------------------ #
    # 9. Discrete classification (pd.cut, no zero class; NaN -> "No data")
    # ------------------------------------------------------------------ #
    values = pd.to_numeric(eez_disp[value_col], errors="coerce")

    if bins is not None:
        raw_cats = pd.cut(values, bins=bins, include_lowest=False)

        # Human-readable labels e.g. "101\u20131,000 km\u00b2"
        bin_labels = []
        for iv in raw_cats.cat.categories:
            lo = int(iv.left) + 1
            hi = int(iv.right)
            bin_labels.append(f"{lo:,}\u2013{hi:,} km\u00b2")

        cat_series = raw_cats.cat.rename_categories(bin_labels).astype(object)
        no_data_mask = values.isna() | raw_cats.isna()
        cat_series[no_data_mask] = "No data"

        n_cats = len(bin_labels)
        if palette == "cubehelix":
            rgb_list = sns.cubehelix_palette(n_cats, start=0.5, rot=-0.5, as_cmap=False)
            cat_colors = {
                lbl: mcolors.to_hex(rgb_list[i]) for i, lbl in enumerate(bin_labels)
            }
        else:
            cmap_obj = plt.get_cmap(palette, n_cats)
            cat_colors = {
                lbl: mcolors.to_hex(cmap_obj(i / max(n_cats - 1, 1)))
                for i, lbl in enumerate(bin_labels)
            }
        cat_colors["No data"] = missing_color

        eez_disp = eez_disp.copy()
        eez_disp["_cat"] = cat_series.values

    # ------------------------------------------------------------------ #
    # 10. Figure and axes
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)

    # Land background at zorder=0 – flat grey only, never shaded by value
    if world_disp is not None:
        world_disp.plot(
            ax=ax, color="#efefef", edgecolor="#cccccc", linewidth=0.3, zorder=0
        )

    # EEZ polygons at zorder=2 (above land)
    if bins is not None:
        legend_patches = []
        for cat_label, color in cat_colors.items():
            subset = eez_disp[eez_disp["_cat"] == cat_label]
            if not subset.empty:
                subset.plot(
                    ax=ax, color=color,
                    edgecolor="white", linewidth=linewidth, zorder=2,
                )
            legend_patches.append(
                Patch(facecolor=color, edgecolor="#888888", linewidth=0.4, label=cat_label)
            )

        if show_legend:
            legend = ax.legend(
                handles=legend_patches,
                loc="lower left",
                bbox_to_anchor=(0.01, 0.01),
                fontsize=legend_fontsize,
                frameon=True,
                title=value_col,
                title_fontsize=legend_fontsize,
            )
            legend.get_frame().set_linewidth(0.6)
            legend.get_frame().set_edgecolor("#666666")
    else:
        # Continuous colorbar fallback
        eez_disp.plot(
            column=value_col,
            ax=ax,
            cmap=palette,
            legend=show_legend,
            missing_kwds={"color": missing_color, "label": "No data"},
            edgecolor="white",
            linewidth=linewidth,
            zorder=2,
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()

    return fig, ax, eez_disp
