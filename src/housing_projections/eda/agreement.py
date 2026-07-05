import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate

from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN, INFER_YEARS

# ── Summary statistics ────────────────────────────────────────────────────────

def compute_agreement_stats(gdf, verbose=True):
    """
    Compute summary statistics on agreement between planning and BEN
    at the total and annual level.

    Returns
    -------
    dict with keys:
        total_corr       : Pearson r between cumulative sums
        total_bias       : mean(sum_plan - sum_ben)
        total_mae        : mean |sum_plan - sum_ben|
        pct_same_sign    : % LSOAs where both sources agree on net direction
        pct_close_total  : % LSOAs where |sum_plan - sum_ben| < threshold
        annual_corr_mean : mean per-LSOA correlation of annual series
        annual_corr_dist : pd.Series of per-LSOA annual correlations
    """
    P     = gdf[INFER_COLS_PLAN].values   # (n_areas, n_years)
    E     = gdf[INFER_COLS_BEN].values

    sum_p = P.sum(axis=1)
    sum_e = E.sum(axis=1)
    diff  = sum_p - sum_e

    # Total level agreement
    total_corr, _  = stats.pearsonr(sum_p, sum_e)
    total_bias     = diff.mean()
    total_mae      = np.abs(diff).mean()
    pct_same_sign  = np.mean(np.sign(sum_p) == np.sign(sum_e)) * 100
    pct_close      = np.mean(np.abs(diff) < 20) * 100

    # Per-area annual correlation
    annual_corrs = pd.Series([
        stats.pearsonr(P[i], E[i])[0]
        for i in range(len(gdf))
    ])

    result = {
        'total_corr':       total_corr,
        'total_bias':       total_bias,
        'total_mae':        total_mae,
        'pct_same_sign':    pct_same_sign,
        'pct_close_total':  pct_close,
        'annual_corr_mean': annual_corrs.mean(),
        'annual_corr_dist': annual_corrs,
    }

    if verbose:
        print("\n── Planning vs BEN agreement ─────────────────────────────────")
        print("\n  Cumulative totals:")
        print(f"    Pearson r:          {total_corr:.3f}")
        print(f"    Mean bias (P - E):  {total_bias:.2f}")
        print(f"    MAE:                {total_mae:.2f}")
        print(f"    Same sign:          {pct_same_sign:.1f}%")
        print(f"    |diff| < 20:        {pct_close:.1f}%")
        print("\n  Annual series (per-LSOA correlation):")
        print(f"    Mean:               {annual_corrs.mean():.3f}")
        print(f"    Median:             {annual_corrs.median():.3f}")
        print(f"    Std:                {annual_corrs.std():.3f}")
        print(f"    % with r > 0.5:     {(annual_corrs > 0.5).mean()*100:.1f}%")
        print(f"    % with r < 0:       {(annual_corrs < 0).mean()*100:.1f}%")

    return result


def classify_lsoas(gdf, close_threshold=20):
    """
    Classify each LSOA into agreement categories based on cumulative totals
    and annual correlation.
    """
    P     = gdf[INFER_COLS_PLAN].values
    E     = gdf[INFER_COLS_BEN].values

    sum_p = P.sum(axis=1)
    sum_e = E.sum(axis=1)
    diff  = sum_p - sum_e

    annual_corrs = np.array([
        stats.pearsonr(P[i], E[i])[0]
        if not (np.all(P[i] == 0) or np.all(E[i] == 0))
        else 0.0
        for i in range(len(gdf))
    ])

    close      = np.abs(diff) < close_threshold
    same_sign  = np.sign(sum_p) == np.sign(sum_e)
    correlated = annual_corrs > 0.5

    # Use explicit if/elif logic per row rather than nested np.where
    categories = []
    for c, s, r in zip(close, same_sign, correlated):
        if c and r:
            categories.append('agree_total_agree_annual')
        elif c and not r:
            categories.append('agree_total_lag')
        elif not c and s:
            categories.append('disagree_total_same_sign')
        else:
            categories.append('disagree_total_diff_sign')

    return pd.DataFrame({
        'lsoa_idx':    np.arange(len(gdf)),
        'sum_plan':    sum_p,
        'sum_ben':     sum_e,
        'diff':        diff,
        'annual_corr': annual_corrs,
        'category':    categories,
    })


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_total_agreement(gdf, stats_dict=None):
    """
    Four-panel overview of total agreement between planning and BEN.
    """
    P     = gdf[INFER_COLS_PLAN].values
    E     = gdf[INFER_COLS_BEN].values
    D     = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values

    sum_p = P.sum(axis=1)
    sum_e = E.sum(axis=1)
    diff  = sum_p - sum_e

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── 1. Scatter: sum_plan vs sum_ben ────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(sum_p, sum_e, alpha=0.2, s=5, color='steelblue')
    lims = [min(sum_p.min(), sum_e.min()), max(sum_p.max(), sum_e.max())]
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=0.8)
    r, _ = stats.pearsonr(sum_p, sum_e)
    ax.set_xlabel('Cumulative planning')
    ax.set_ylabel('Cumulative BEN')
    ax.set_title(f'Cumulative totals: planning vs BEN (r={r:.3f})')
    ax.spines[['top', 'right']].set_visible(False)

    # ── 2. Distribution of diff ────────────────────────────────────────────
    ax = axes[0, 1]
    clip = np.percentile(np.abs(diff), 99)
    ax.hist(diff, bins=100, density=True, color='steelblue', alpha=0.7,
            range=(-clip, clip))
    ax.axvline(0,           color='black', linewidth=0.8)
    ax.axvline(diff.mean(), color='red',   linewidth=0.8, linestyle='--',
               label=f'mean={diff.mean():.1f}')
    ax.axvline(np.median(diff), color='darkred', linewidth=0.8, linestyle=':',
               label=f'median={np.median(diff):.1f}')
    ax.set_xlabel('sum_plan - sum_ben')
    ax.set_title('Distribution of cumulative difference')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    # ── 3. Both vs census diff ─────────────────────────────────────────────
    ax = axes[1, 0]
    clip = np.percentile(np.abs(np.concatenate([sum_p, sum_e, D])), 99)
    ax.scatter(D, sum_p, alpha=0.2, s=5, color='steelblue', label='Planning')
    ax.scatter(D, sum_e, alpha=0.2, s=5, color='coral',     label='BEN')
    ax.plot([-clip, clip], [-clip, clip], color='black',
            linestyle='--', linewidth=0.8)
    ax.set_xlim(-clip, clip)
    ax.set_ylim(-clip, clip)
    ax.set_xlabel('Census diff (D)')
    ax.set_ylabel('Cumulative estimate')
    ax.set_title('Planning and BEN vs census diff')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    # ── 4. Distribution of annual correlations ─────────────────────────────
    ax = axes[1, 1]
    annual_corrs = pd.Series([
        stats.pearsonr(P[i], E[i])[0]
        for i in range(len(gdf))
    ])
    ax.hist(annual_corrs, bins=50, density=True,
            color='steelblue', alpha=0.7)
    ax.axvline(0,                   color='black', linewidth=0.8)
    ax.axvline(annual_corrs.mean(), color='red',   linewidth=0.8,
               linestyle='--', label=f'mean={annual_corrs.mean():.3f}')
    ax.axvline(annual_corrs.median(), color='darkred', linewidth=0.8,
               linestyle=':', label=f'median={annual_corrs.median():.3f}')
    ax.set_xlabel('Per-LSOA annual correlation')
    ax.set_title('Distribution of annual correlations')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle('Planning vs BEN agreement overview')
    plt.tight_layout()
    return fig, axes


def plot_category_breakdown(gdf, classification_df):
    """
    Bar chart of LSOA classification categories with counts and percentages.
    """
    counts = classification_df['category'].value_counts()
    pcts   = counts / len(classification_df) * 100

    colours = {
        'agree_total_agree_annual': 'steelblue',
        'agree_total_lag':          'coral',
        'disagree_total_same_sign': 'orange',
        'disagree_total_diff_sign': 'red',
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        range(len(counts)),
        counts.values,
        color=[colours.get(c, 'grey') for c in counts.index],
        alpha=0.8,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([c.replace('_', '\n') for c in counts.index], fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title('LSOA classification by planning/BEN agreement')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{pct:.1f}%', ha='center', fontsize=8)

    plt.tight_layout()
    return fig, ax


def plot_category_examples(gdf, classification_df, n_per_category=3):
    """
    Plot planning and BEN time series for example LSOAs from each category.
    """
    P      = gdf[INFER_COLS_PLAN].values
    E      = gdf[INFER_COLS_BEN].values
    D      = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values

    categories = classification_df['category'].unique()
    n_cats     = len(categories)

    fig, axes = plt.subplots(n_cats, n_per_category,
                             figsize=(6 * n_per_category, 4 * n_cats))

    for row, cat in enumerate(categories):
        cat_df  = classification_df[classification_df['category'] == cat]
        # Sample evenly across the range of |diff| within each category
        sample  = cat_df.iloc[
            np.linspace(0, len(cat_df) - 1, n_per_category).astype(int)
        ]

        for col, (_, row_data) in enumerate(sample.iterrows()):
            ax  = axes[row, col]
            idx = int(row_data['lsoa_idx'])

            ax.plot(INFER_YEARS, P[idx], color='steelblue',
                    marker='s', linewidth=1.0, alpha=0.8, label='Planning')
            ax.plot(INFER_YEARS, E[idx], color='coral',
                    marker='^', linewidth=1.0, alpha=0.8, label='BEN')
            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

            ax.set_title(
                f'{cat.replace("_", " ")}\n'
                f'sum_P={row_data["sum_plan"]:.0f}  '
                f'sum_E={row_data["sum_ben"]:.0f}  '
                f'D={D[idx]:.0f}  '
                f'r={row_data["annual_corr"]:.2f}',
                fontsize=7
            )
            ax.set_xlabel('Year')
            ax.set_ylabel('Net change')
            ax.spines[['top', 'right']].set_visible(False)

            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.suptitle('Example LSOAs by agreement category')
    plt.tight_layout()
    return fig, axes


def plot_lag_candidates(gdf, classification_df, n_examples=6):

    P      = gdf[INFER_COLS_PLAN].values
    E      = gdf[INFER_COLS_BEN].values
    D      = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values

    lag_df = classification_df[
        classification_df['category'] == 'agree_total_lag'
    ].copy()

    if len(lag_df) == 0:
        print("No lag candidates found.")
        return None, None

    lag_df  = lag_df.reindex(
        lag_df['diff'].abs().sort_values().index
    ).head(n_examples)

    n_years   = len(INFER_YEARS)
    lag_years = np.arange(-(n_years - 1), n_years)

    # Two rows per example — time series on top, cross-corr below
    fig, axes = plt.subplots(
        n_examples * 2, 1,
        figsize=(10, 4 * n_examples),
        gridspec_kw={'height_ratios': [2, 1] * n_examples}
    )

    for i, (_, row_data) in enumerate(lag_df.iterrows()):
        ax_ts   = axes[i * 2]
        ax_xcorr = axes[i * 2 + 1]

        idx  = int(row_data['lsoa_idx'])
        p    = P[idx]
        e    = E[idx]

        p_std = p.std()
        e_std = e.std()
        p_n   = (p - p.mean()) / (p_std if p_std > 0 else 1)
        e_n   = (e - e.mean()) / (e_std if e_std > 0 else 1)

        xcorr    = correlate(p_n, e_n, mode='full') / n_years
        peak_lag = lag_years[np.argmax(np.abs(xcorr))]

        # Time series
        ax_ts.plot(INFER_YEARS, p, color='steelblue', marker='s',
                   linewidth=1.0, alpha=0.8, label='Planning')
        ax_ts.plot(INFER_YEARS, e, color='coral', marker='^',
                   linewidth=1.0, alpha=0.8, label='BEN')
        ax_ts.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax_ts.set_ylabel('Net change')
        ax_ts.set_xticks(INFER_YEARS)
        ax_ts.set_title(
            f'LSOA {idx}  D={D[idx]:.0f}  '
            f'sum_P={row_data["sum_plan"]:.0f}  '
            f'sum_E={row_data["sum_ben"]:.0f}  '
            f'peak lag={peak_lag} yrs',
            fontsize=8
        )
        ax_ts.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax_ts.legend(fontsize=7)

        # Cross-correlation on its own axis with its own x range
        ax_xcorr.bar(lag_years, xcorr, color='grey', alpha=0.5, width=0.4)
        ax_xcorr.axvline(0, color='black', linewidth=0.8)
        ax_xcorr.axhline(0, color='black', linewidth=0.5)
        ax_xcorr.set_xlim(-(n_years - 1) - 0.5, (n_years - 1) + 0.5)
        ax_xcorr.set_xticks(lag_years)
        ax_xcorr.set_xlabel('Lag (years, positive = planning leads BEN)')
        ax_xcorr.set_ylabel('Cross-corr')
        ax_xcorr.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Lag candidates — totals agree but annual series do not')
    plt.tight_layout()
    return fig, axes


def plot_sign_disagreements(gdf, classification_df, n_examples=6):
    """
    Plot LSOAs where planning and BEN disagree on net direction.
    These are the most problematic cases for modelling.
    """
    P   = gdf[INFER_COLS_PLAN].values
    E   = gdf[INFER_COLS_BEN].values
    D   = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values

    sign_df = classification_df[
        classification_df['category'] == 'disagree_total_diff_sign'
    ].copy()

    if len(sign_df) == 0:
        print("No sign disagreement cases found.")
        return None, None

    # Sort by absolute diff — worst disagreements first
    sign_df = sign_df.reindex(
        sign_df['diff'].abs().sort_values(ascending=False).index
    ).head(n_examples)

    n_cols  = 3
    n_rows  = int(np.ceil(len(sign_df) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()

    for ax, (_, row_data) in zip(axes, sign_df.iterrows()):
        idx = int(row_data['lsoa_idx'])

        ax.plot(INFER_YEARS, P[idx], color='steelblue', marker='s',
                linewidth=1.0, alpha=0.8, label='Planning')
        ax.plot(INFER_YEARS, E[idx], color='coral', marker='^',
                linewidth=1.0, alpha=0.8, label='BEN')
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.axhline(D[idx] / len(INFER_YEARS), color='green',
                   linewidth=0.8, linestyle='--', alpha=0.5,
                   label=f'D/n ({D[idx]/len(INFER_YEARS):.1f})')

        ax.set_title(
            f'LSOA {idx}  D={D[idx]:.0f}\n'
            f'sum_P={row_data["sum_plan"]:.0f}  '
            f'sum_E={row_data["sum_ben"]:.0f}',
            fontsize=8
        )
        ax.set_xlabel('Year')
        ax.set_ylabel('Net change')
        ax.spines[['top', 'right']].set_visible(False)

        if idx == int(sign_df.iloc[0]['lsoa_idx']):
            ax.legend(fontsize=7)

    for ax in axes[len(sign_df):]:
        ax.set_visible(False)

    plt.suptitle('Sign disagreements — planning and BEN disagree on net direction')
    plt.tight_layout()
    return fig, axes


# ── Full agreement analysis ───────────────────────────────────────────────────

def full_agreement_analysis(gdf, close_threshold=20, n_examples=3,
                             verbose=True):
    """
    Run the full agreement analysis pipeline.

    Parameters
    ----------
    gdf             : GeoDataFrame
    close_threshold : float — threshold for classifying totals as close
    n_examples      : int — examples per category in plots
    verbose         : bool

    Returns
    -------
    dict with keys 'stats' and 'classification'
    """
    stats_dict      = compute_agreement_stats(gdf, verbose=verbose)
    classification  = classify_lsoas(gdf, close_threshold=close_threshold)

    if verbose:
        print("\n── LSOA classification ───────────────────────────────────────")
        counts = classification['category'].value_counts()
        pcts   = counts / len(classification) * 100
        for cat, count in counts.items():
            print(f"  {cat:40s}: {count:5d} ({pcts[cat]:.1f}%)")

    plot_total_agreement(gdf, stats_dict)
    plot_category_breakdown(gdf, classification)
    plot_category_examples(gdf, classification, n_per_category=n_examples)
    plot_lag_candidates(gdf, classification)
    plot_sign_disagreements(gdf, classification)

    return {
        'stats':          stats_dict,
        'classification': classification,
    }
