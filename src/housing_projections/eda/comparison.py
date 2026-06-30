# housing_projections/eda/comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from housing_projections.config import INFER_COLS_PLAN, INFER_COLS_BEN, INFER_YEARS


# ── Census stock overview ─────────────────────────────────────────────────────

def plot_census_stocks(gdf,
                       col_2011='dwellings_2011',
                       col_2021='dwellings_2021'):
    """
    Overview of census dwelling stocks in 2011 and 2021 and the
    distribution of intercensal change.
    """
    D = gdf[col_2021].values - gdf[col_2011].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 2011 stock distribution
    axes[0].hist(gdf[col_2011].values, bins=100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Dwellings')
    axes[0].set_title('2011 dwelling stock')
    axes[0].spines[['top', 'right']].set_visible(False)

    # 2021 stock distribution
    axes[1].hist(gdf[col_2021].values, bins=100, color='coral', alpha=0.7)
    axes[1].set_xlabel('Dwellings')
    axes[1].set_title('2021 dwelling stock')
    axes[1].spines[['top', 'right']].set_visible(False)

    # Intercensal change distribution
    clip = np.quantile(np.abs(D), 0.99)
    axes[2].hist(D, bins=100, color='steelblue', alpha=0.7,
                 range=(-clip, clip))
    axes[2].axvline(0,        color='black', linewidth=0.8)
    axes[2].axvline(D.mean(), color='red',   linewidth=0.8,
                    linestyle='--', label=f'mean={D.mean():.1f}')
    axes[2].axvline(np.median(D), color='darkred', linewidth=0.8,
                    linestyle=':',  label=f'median={np.median(D):.1f}')
    axes[2].set_xlabel('Net dwelling change')
    axes[2].set_title('Intercensal change (2011-2021)')
    axes[2].spines[['top', 'right']].set_visible(False)
    axes[2].legend(fontsize=8)

    plt.suptitle('Census dwelling stocks overview')
    plt.tight_layout()
    plt.show()


def plot_stock_scatter(gdf, col_2011='dwellings_2011',
                        col_2021='dwellings_2021'):
    """
    Scatter of 2011 vs 2021 dwelling stock with 1:1 line.
    Areas above the line gained dwellings, below lost.
    """
    x = gdf[col_2011].values
    y = gdf[col_2021].values

    r, _   = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.2, s=5, color='steelblue')

    lims    = [min(x.min(), y.min()), max(x.max(), y.max())]
    x_range = np.linspace(lims[0], lims[1], 100)

    ax.plot(lims,    lims,
            color='black', linestyle='--', linewidth=0.8, label='1:1')
    ax.plot(x_range, slope * x_range + intercept,
            color='red',   linestyle='-',  linewidth=0.8,
            label=f'fitted (slope={slope:.3f})')

    ax.set_xlabel('Dwelling stock 2011')
    ax.set_ylabel('Dwelling stock 2021')
    ax.set_title(f'Census stock 2011 vs 2021 (r={r:.3f})')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── Cumulative flow vs intercensal change ─────────────────────────────────────

def plot_cumulative_vs_intercensal(gdf, cols, labels,
                                    exact_col='intercensal_change'):
    """
    For each cumulative flow estimate (sum of annual P or E),
    scatter against exact intercensal change with 1:1 line,
    fitted regression, and summary statistics.

    Parameters
    ----------
    gdf       : GeoDataFrame
    cols      : list of str — cumulative estimate columns
    labels    : list of str — display names
    exact_col : str — column with exact intercensal change (census diff)
    """
    n     = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col, label in zip(axes, cols, labels):
        x = gdf[exact_col].values
        y = gdf[col].values

        mask    = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        r, p_val                      = stats.pearsonr(x_clean, y_clean)
        r2_1to1                       = 1 - np.sum((y_clean - x_clean)**2) / \
                                            np.sum((y_clean - y_clean.mean())**2)
        slope, intercept, r_fit, _, _ = stats.linregress(x_clean, y_clean)
        r2_fitted                     = r_fit ** 2

        ax.scatter(x_clean, y_clean, alpha=0.2, s=5, color='steelblue')

        lims    = [min(x_clean.min(), y_clean.min()),
                   max(x_clean.max(), y_clean.max())]
        x_range = np.linspace(lims[0], lims[1], 100)

        ax.plot(lims,    lims,
                color='black', linestyle='--', linewidth=0.8, label='1:1')
        ax.plot(x_range, slope * x_range + intercept,
                color='red',   linestyle='-',  linewidth=0.8,
                label=f'fitted (slope={slope:.2f}, intercept={intercept:.1f})')

        stats_text = (
            f'r={r:.3f} (p={p_val:.1e})\n'
            f'R²(1:1)={r2_1to1:.3f}\n'
            f'R²(fit)={r2_fitted:.3f}\n'
            f'slope={slope:.3f}\n'
            f'intercept={intercept:.2f}\n'
            f'N={mask.sum()}'
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Intercensal change (exact census diff)')
        ax.set_ylabel(f'Cumulative {label}')
        ax.set_title(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7)

    plt.suptitle('Cumulative flow estimates vs exact intercensal change')
    plt.tight_layout()
    plt.show()


# ── Direct P vs E comparison ──────────────────────────────────────────────────

def compute_overall_correlation(gdf, verbose=True):
    """
    Compute overall correlation between planning and BEN across all
    areas and years simultaneously, plus per-area correlation distribution.

    Returns
    -------
    dict with keys 'overall_r', 'overall_p', 'per_area_corr'
    """
    P = gdf[INFER_COLS_PLAN].values
    E = gdf[INFER_COLS_BEN].values

    overall_r, overall_p = stats.pearsonr(P.ravel(), E.ravel())

    per_area_corr = pd.Series([
        stats.pearsonr(P[i], E[i])[0]
        for i in range(len(gdf))
    ])

    result = {
        'overall_r':     overall_r,
        'overall_p':     overall_p,
        'per_area_corr': per_area_corr,
    }

    if verbose:
        print(f"\n── Overall planning vs BEN correlation ──────────────────────")
        print(f"  Overall (flattened):    r={overall_r:.3f} (p={overall_p:.1e})")
        print(f"  Mean per-area:          r={per_area_corr.mean():.3f}")
        print(f"  Median per-area:        r={per_area_corr.median():.3f}")
        print(f"  Std per-area:           {per_area_corr.std():.3f}")
        print(f"  % with r > 0.5:         "
              f"{(per_area_corr > 0.5).mean()*100:.1f}%")

    return result


def plot_per_area_correlation(gdf):
    """
    Histogram of per-area correlations between planning and BEN.
    """
    P = gdf[INFER_COLS_PLAN].values
    E = gdf[INFER_COLS_BEN].values

    per_area_corr = pd.Series([
        stats.pearsonr(P[i], E[i])[0]
        for i in range(len(gdf))
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_area_corr, bins=50, density=True,
            color='steelblue', alpha=0.7)
    ax.axvline(0,                      color='black',   linewidth=0.8)
    ax.axvline(per_area_corr.mean(),   color='red',     linewidth=0.8,
               linestyle='--', label=f'mean={per_area_corr.mean():.3f}')
    ax.axvline(per_area_corr.median(), color='darkred', linewidth=0.8,
               linestyle=':',  label=f'median={per_area_corr.median():.3f}')
    ax.set_xlabel('Per-area Pearson r (planning vs BEN)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of per-area annual correlations between '
                 'planning and BEN')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_annual_p_vs_e(gdf, n_cols=5):
    """
    Year-by-year scatter of planning vs BEN with 1:1 line and correlation.
    """
    n_years = len(INFER_YEARS)
    n_rows  = int(np.ceil(n_years / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()

    for ax, yr, col_p, col_e in zip(
        axes, INFER_YEARS, INFER_COLS_PLAN, INFER_COLS_BEN
    ):
        p = gdf[col_p].values
        e = gdf[col_e].values

        mask    = np.isfinite(p) & np.isfinite(e)
        p_clean = p[mask]
        e_clean = e[mask]

        r, _  = stats.pearsonr(p_clean, e_clean)
        clip  = np.quantile(
            np.abs(np.concatenate([p_clean, e_clean])), 0.99)

        ax.scatter(p_clean, e_clean, alpha=0.2, s=3, color='steelblue')
        ax.plot([-clip, clip], [-clip, clip],
                color='black', linestyle='--', linewidth=0.8)
        ax.set_xlim(-clip, clip)
        ax.set_ylim(-clip, clip)
        ax.set_title(f'{yr}  (r={r:.2f})', fontsize=8)
        ax.set_xlabel('Planning', fontsize=7)
        ax.set_ylabel('BEN',      fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[n_years:]:
        ax.set_visible(False)

    plt.suptitle('Annual planning vs BEN by year')
    plt.tight_layout()
    plt.show()