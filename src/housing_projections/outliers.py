# housing_projections/outliers.py

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN, INFER_YEARS

# ── Detection ─────────────────────────────────────────────────────────────────

def _find_outliers(gdf, max_plausible=2000, min_plausible=-500,
                  discrepancy_threshold=500):
    """
    Find suspicious observations in planning and BEN data.

    Two categories with different severity:

    HARD outliers (excluded by default):
      - Values above max_plausible or below min_plausible in either source
        These are likely data entry errors and implausible under any
        reasonable interpretation.

    SOFT outliers (flagged but retained by default):
      - Large discrepancy between planning and BEN in a single LSOA-year
        where one source is near zero. These may reflect genuine temporal
        lags, missing data, or spatial misallocation rather than errors.

    Parameters
    ----------
    gdf                   : GeoDataFrame
    max_plausible         : float
    min_plausible         : float
    discrepancy_threshold : float

    Returns
    -------
    pd.DataFrame with columns:
        lsoa_idx, lsoa_id, year, source, value, reason, severity
        where severity is 'hard' or 'soft'
    """
    records = []

    P = gdf[INFER_COLS_PLAN].values
    E = gdf[INFER_COLS_BEN].values

    id_col = next(
        (c for c in ['LSOA11CD', 'lsoa11cd', 'geo_code', 'code', 'lsoa_code']
         if c in gdf.columns),
        None
    )

    for i in range(len(gdf)):
        lsoa_id = gdf.iloc[i][id_col] if id_col else i

        for t, yr in enumerate(INFER_YEARS):
            p_val = P[i, t]
            e_val = E[i, t]

            # Hard: out of plausible range
            for val, source in [(p_val, 'planning'), (e_val, 'ben')]:
                if val > max_plausible:
                    records.append({
                        'lsoa_idx': i,
                        'lsoa_id':  lsoa_id,
                        'year':     yr,
                        'source':   source,
                        'value':    val,
                        'reason':   f'above max_plausible ({max_plausible})',
                        'severity': 'hard',
                    })
                elif val < min_plausible:
                    records.append({
                        'lsoa_idx': i,
                        'lsoa_id':  lsoa_id,
                        'year':     yr,
                        'source':   source,
                        'value':    val,
                        'reason':   f'below min_plausible ({min_plausible})',
                        'severity': 'hard',
                    })

            # Soft: large discrepancy with one source near zero
            discrepancy = abs(p_val - e_val)
            if discrepancy > discrepancy_threshold:
                if abs(p_val) < 10 or abs(e_val) < 10:
                    records.append({
                        'lsoa_idx': i,
                        'lsoa_id':  lsoa_id,
                        'year':     yr,
                        'source':   'both',
                        'value':    discrepancy,
                        'reason':   f'large discrepancy ({discrepancy:.0f}) '
                                    f'with one source near zero — may reflect '
                                    f'lag, missing data, or misallocation',
                        'severity': 'soft',
                    })

    return pd.DataFrame(records)


def _get_hard_outlier_lsoa_indices(outlier_df):
    """Return unique LSOA indices with hard outliers only."""
    hard = outlier_df[outlier_df['severity'] == 'hard']
    return sorted(hard['lsoa_idx'].unique().tolist())


def _get_soft_outlier_lsoa_indices(outlier_df):
    """Return unique LSOA indices with soft outliers only (excluding hard)."""
    hard_idx = set(_get_hard_outlier_lsoa_indices(outlier_df))
    soft     = outlier_df[outlier_df['severity'] == 'soft']
    return sorted(set(soft['lsoa_idx'].unique().tolist()) - hard_idx)


# ── Analysis ──────────────────────────────────────────────────────────────────

def _analyse_outliers(gdf, outlier_df, verbose=True):
    """
    Print a summary of detected outliers broken down by severity.

    Returns
    -------
    dict with summary statistics
    """
    hard = outlier_df[outlier_df['severity'] == 'hard']
    soft = outlier_df[outlier_df['severity'] == 'soft']

    result = {
        'hard': {
            'n_lsoas':      hard['lsoa_idx'].nunique(),
            'n_lsoa_years': len(hard),
            'by_source':    hard.groupby('source').size(),
            'by_year':      hard.groupby('year').size(),
            'details':      hard,
        },
        'soft': {
            'n_lsoas':      soft['lsoa_idx'].nunique(),
            'n_lsoa_years': len(soft),
            'by_source':    soft.groupby('source').size(),
            'by_year':      soft.groupby('year').size(),
            'details':      soft,
        },
    }

    if verbose:
        print("\n── Outlier summary ───────────────────────────────────────────")
        print("\n  HARD outliers (excluded by default — likely data errors):")
        print(f"    Flagged LSOAs:      {result['hard']['n_lsoas']} / {len(gdf)}")
        print(f"    Flagged LSOA-years: {result['hard']['n_lsoa_years']}")
        if result['hard']['n_lsoas'] > 0:
            print(f"    By source:\n{hard.groupby('source').size().to_string()}")
            print(f"    By year:\n{hard.groupby('year').size().to_string()}")
            print(f"    Details:\n{hard.to_string()}")

        print("\n  SOFT outliers (retained — may reflect lag/missing data):")
        print(f"    Flagged LSOAs:      {result['soft']['n_lsoas']} / {len(gdf)}")
        print(f"    Flagged LSOA-years: {result['soft']['n_lsoa_years']}")
        if result['soft']['n_lsoas'] > 0:
            print(f"    By source:\n{soft.groupby('source').size().to_string()}")
            print(f"    By year:\n{soft.groupby('year').size().to_string()}")

    return result


def plot_hard_outlier_areas(gdf, outlier_df, n_cols=3):
    """Plot time series for hard-outlier LSOAs only."""
    plot_outlier_areas(gdf, outlier_df, severity='hard', n_cols=n_cols)


def plot_soft_outlier_areas(gdf, outlier_df, n_cols=3):
    """Plot time series for soft-outlier LSOAs only."""
    plot_outlier_areas(gdf, outlier_df, severity='soft', n_cols=n_cols)


def plot_outlier_areas(gdf, outlier_df, severity='both', n_cols=3):
    """
    Plot planning and BEN time series for flagged areas.

    Parameters
    ----------
    severity : 'hard', 'soft', or 'both'
    """
    if severity == 'hard':
        indices = _get_hard_outlier_lsoa_indices(outlier_df)
        sev_label = 'hard'
    elif severity == 'soft':
        indices = _get_soft_outlier_lsoa_indices(outlier_df)
        sev_label = 'soft'
    else:
        indices   = sorted(outlier_df['lsoa_idx'].unique().tolist())
        sev_label = 'hard and soft'

    if len(indices) == 0:
        print(f"No {sev_label} outlier areas to plot.")
        return None, None

    P      = gdf[INFER_COLS_PLAN].values
    E      = gdf[INFER_COLS_BEN].values
    id_col = next(
        (c for c in ['LSOA11CD', 'lsoa11cd', 'geo_code', 'code', 'lsoa_code']
         if c in gdf.columns),
        None
    )

    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()

    for ax, idx in zip(axes, indices):
        lsoa_id = gdf.iloc[idx][id_col] if id_col else idx
        flags   = outlier_df[outlier_df['lsoa_idx'] == idx]

        ax.plot(INFER_YEARS, P[idx], color='steelblue',
                marker='s', linewidth=1.0, label='Planning')
        ax.plot(INFER_YEARS, E[idx], color='coral',
                marker='^', linewidth=1.0, label='BEN')
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

        # Mark flagged years — hard in red, soft in orange
        for _, flag in flags.iterrows():
            colour = 'red' if flag['severity'] == 'hard' else 'orange'
            ax.axvline(flag['year'], color=colour,
                       linewidth=0.8, linestyle='--', alpha=0.6)

        severities  = flags['severity'].unique()
        sev_str     = ' + '.join(severities)
        ax.set_title(f'{lsoa_id}  [{sev_str}]\n'
                     f'{flags["reason"].iloc[0]}', fontsize=7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Net dwelling change')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7)

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    plt.suptitle(f'Flagged outlier areas ({sev_label})')
    plt.tight_layout()
    return fig, axes


# ── Exclusion ─────────────────────────────────────────────────────────────────

def _exclude_hard_outlier_lsoas(gdf, outlier_df, verbose=True):
    """
    Return a copy of gdf with hard outlier LSOAs removed.
    Soft outlier LSOAs are retained.

    Parameters
    ----------
    gdf        : GeoDataFrame
    outlier_df : pd.DataFrame — output of find_outliers()
    verbose    : bool

    Returns
    -------
    GeoDataFrame with hard outlier LSOAs removed and index reset
    """
    hard_indices = _get_hard_outlier_lsoa_indices(outlier_df)
    soft_indices = _get_soft_outlier_lsoa_indices(outlier_df)

    mask     = ~pd.Series(range(len(gdf))).isin(hard_indices)
    gdf_clean = gdf[mask.values].reset_index(drop=True)

    if verbose:
        print("\n── Outlier exclusion ─────────────────────────────────────────")
        print(f"  Removed {len(hard_indices)} hard outlier LSOAs")
        print(f"  Retained {len(soft_indices)} soft outlier LSOAs "
              f"(may reflect lag or missing data)")
        print(f"  Remaining: {len(gdf_clean)} / {len(gdf)} LSOAs")

    return gdf_clean


def apply_outlier_exclusion(gdf, max_plausible=2000, min_plausible=-500,
                             discrepancy_threshold=500, verbose=True):
    """
    Run outlier detection and exclude hard outlier LSOAs.
    Soft outliers are flagged but retained.

    Call this explicitly in your notebook after load_data() if you want
    to remove hard outliers before modelling.

    Parameters
    ----------
    gdf                   : GeoDataFrame — output of load_data()
    max_plausible         : float
    min_plausible         : float
    discrepancy_threshold : float
    verbose               : bool

    Returns
    -------
    tuple of (GeoDataFrame, pd.DataFrame)
        cleaned GeoDataFrame and full outlier_df for inspection
    """
    outlier_df = _find_outliers(
        gdf,
        max_plausible         = max_plausible,
        min_plausible         = min_plausible,
        discrepancy_threshold = discrepancy_threshold,
    )

    _analyse_outliers(gdf, outlier_df, verbose=verbose)

    gdf_clean = _exclude_hard_outlier_lsoas(gdf, outlier_df, verbose=verbose)

    if verbose:
        print("\n   To inspect flagged areas further:")
        print("     from housing_projections.outliers import "
              "plot_outlier_areas, plot_outlier_map")
        print("     plot_outlier_areas(gdf, outlier_df, severity='hard')")
        print("     plot_outlier_areas(gdf, outlier_df, severity='soft')")
        print("     plot_outlier_map(gdf, outlier_df)")

    return gdf_clean, outlier_df



def plot_outlier_map(gdf, outlier_df):
    """
    Plot flagged outlier areas on a map of all LSOAs.

    Hard outliers shown in red, soft outliers in orange,
    clean areas in light grey.

    Parameters
    ----------
    gdf        : GeoDataFrame — full dataset (before exclusion)
    outlier_df : pd.DataFrame — output of find_outliers()
    """
    hard_indices = set(_get_hard_outlier_lsoa_indices(outlier_df))
    soft_indices = set(_get_soft_outlier_lsoa_indices(outlier_df))

    def classify(idx):
        if idx in hard_indices:
            return 'hard'
        elif idx in soft_indices:
            return 'soft'
        return 'clean'

    gdf_plot          = gdf.copy()
    gdf_plot['flag']  = [classify(i) for i in range(len(gdf))]

    colour_map = {
        'clean': '#d9d9d9',
        'soft':  'orange',
        'hard':  'red',
    }
    gdf_plot['colour'] = gdf_plot['flag'].map(colour_map)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot clean areas first, then soft, then hard so flagged areas sit on top
    for flag in ['clean', 'soft', 'hard']:
        subset = gdf_plot[gdf_plot['flag'] == flag]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colour_map[flag],
                        edgecolor='white', linewidth=0.2)

    # Legend
    patches = [
        mpatches.Patch(color='#d9d9d9', label=f'Clean ({(gdf_plot["flag"]=="clean").sum()})'),
        mpatches.Patch(color='orange',  label=f'Soft outlier ({len(soft_indices)})'),
        mpatches.Patch(color='red',     label=f'Hard outlier ({len(hard_indices)})'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=9)
    ax.set_axis_off()
    ax.set_title('Flagged outlier LSOAs')
    plt.tight_layout()
