"""
Model sensitivity diagnostics: how much do z estimates depend on model choice?
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "compute_z_model_sensitivity",
    "compute_model_agreement_matrix",
    "compute_z_ensemble",
    "compute_decomposed_uncertainty",
    "plot_z_sensitivity_map",
    "plot_model_agreement_matrix",
    "plot_z_range_distribution",
    "plot_sensitivity_vs_disagreement",
    "plot_ensemble_mean_map",
    "plot_estimate_vs_uncertainty",
]


# ── Compute functions ─────────────────────────────────────────────────────────

def compute_z_model_sensitivity(traces):
    """
    Compute per-LSOA z sensitivity: how much do posterior mean z estimates
    vary across models?

    Parameters
    ----------
    traces : dict[str, az.InferenceData]
        Keyed by model name. Each must have a 'z' variable in posterior
        of shape (chain, draw, n_areas, n_years).

    Returns
    -------
    summary : pd.DataFrame
        One row per LSOA. Columns: z_mean_{model} for each model, plus
        z_mean_across_models, z_std_across_models, z_range_across_models.
        Each z statistic is the mean over all years.
    long_form : pd.DataFrame
        Long-form with columns: model, lsoa_idx, year, z_mean.
    """
    z_means = {}
    for name, trace in traces.items():
        z_post = trace.posterior['z'].values            # (chain, draw, n_areas, n_years)
        z_means[name] = z_post.mean(axis=(0, 1))        # (n_areas, n_years)

    model_names = list(z_means)
    n_areas, n_years = next(iter(z_means.values())).shape

    # Long form
    rows = []
    for name, zm in z_means.items():
        for lsoa in range(n_areas):
            for t in range(n_years):
                rows.append({'model': name, 'lsoa_idx': lsoa, 'year': t, 'z_mean': zm[lsoa, t]})
    long_form = pd.DataFrame(rows)

    # Summary: per-LSOA, mean over years
    z_mat = np.stack([z_means[m].mean(axis=1) for m in model_names], axis=1)   # (n_areas, n_models)
    summary = pd.DataFrame({f'z_mean_{m}': z_mat[:, i] for i, m in enumerate(model_names)})
    summary['z_mean_across_models'] = z_mat.mean(axis=1)
    summary['z_std_across_models']  = z_mat.std(axis=1)
    summary['z_range_across_models'] = z_mat.max(axis=1) - z_mat.min(axis=1)

    return summary, long_form


def compute_model_agreement_matrix(traces):
    """
    Compute pairwise correlation matrix of flattened z posterior means.

    Parameters
    ----------
    traces : dict[str, az.InferenceData]

    Returns
    -------
    pd.DataFrame  — model × model correlation matrix, symmetric, diagonal=1.
    """
    z_flat = {}
    for name, trace in traces.items():
        z_post = trace.posterior['z'].values
        z_flat[name] = z_post.mean(axis=(0, 1)).ravel()

    df = pd.DataFrame(z_flat)
    return df.corr()


def compute_z_ensemble(traces, comparison_df=None):
    """
    Compute LOO-stacking-weighted ensemble z posterior mean.

    Parameters
    ----------
    traces        : dict[str, az.InferenceData]
    comparison_df : pd.DataFrame or None
        ArviZ LOO comparison table (output of compute_model_comparison).
        If None, all models receive equal weight.

    Returns
    -------
    np.ndarray  — shape (n_areas, n_years), weighted average z mean.
    """
    model_names = list(traces)

    if comparison_df is not None and 'weight' in comparison_df.columns:
        weights = {name: float(comparison_df.loc[name, 'weight'])
                   if name in comparison_df.index else 0.0
                   for name in model_names}
    else:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    total = sum(weights.values())
    ensemble = None

    for name, trace in traces.items():
        w = weights.get(name, 0.0) / total if total > 0 else 0.0
        z_mean = trace.posterior['z'].values.mean(axis=(0, 1))
        if ensemble is None:
            ensemble = w * z_mean
        else:
            ensemble += w * z_mean

    return ensemble


def compute_decomposed_uncertainty(traces, comparison_df=None, ci=0.9, lsoa_codes=None):
    """
    Decompose per-LSOA z uncertainty into within-model and between-model components.

    Within-model uncertainty captures sampling variance given one model's assumptions.
    Between-model uncertainty captures how much the estimate itself shifts across models.
    Together they give a more honest picture of total uncertainty than any single model
    credible interval alone.

    Parameters
    ----------
    traces        : dict[str, az.InferenceData]
    comparison_df : pd.DataFrame or None — LOO comparison table; used for stacking weights.
                    If None, equal weights are used.
    ci            : float — credible interval width, e.g. 0.9 for 90%
    lsoa_codes    : array-like or None — LSOA21CD codes, one per area row

    Returns
    -------
    pd.DataFrame with one row per LSOA and columns:
        lsoa_idx, (lsoa_code if provided),
        z_ensemble_mean      — LOO-stacking weighted posterior mean
        z_within_uncertainty — mean posterior SD across models (within-model)
        z_between_uncertainty — std of posterior means across models (between-model)
        z_total_uncertainty  — sqrt(within² + between²)
        z_ci_lo, z_ci_hi     — ensemble-mean ± z_total_uncertainty * z_factor
        confidence_tier      — 'High' / 'Medium' / 'Low'
    """
    model_names = list(traces)

    if comparison_df is not None and 'weight' in comparison_df.columns:
        raw_weights = {n: float(comparison_df.loc[n, 'weight'])
                       if n in comparison_df.index else 0.0
                       for n in model_names}
    else:
        raw_weights = {n: 1.0 / len(model_names) for n in model_names}
    total_w = sum(raw_weights.values()) or 1.0
    weights = {n: raw_weights[n] / total_w for n in model_names}

    # Per-model posterior means and SDs, averaged over years → (n_areas,)
    z_post_means = {}   # posterior mean per area per model
    z_post_sds   = {}   # posterior SD per area per model

    for name, trace in traces.items():
        z = trace.posterior['z'].values           # (chains, draws, n_areas, n_years)
        flat = z.reshape(-1, z.shape[2], z.shape[3])  # (S, n_areas, n_years)
        z_post_means[name] = flat.mean(axis=0).mean(axis=1)   # (n_areas,)
        z_post_sds[name]   = flat.std(axis=0).mean(axis=1)    # (n_areas,)

    n_areas = next(iter(z_post_means.values())).shape[0]

    # Ensemble mean: LOO-stacking weighted average of posterior means
    ensemble_mean = np.zeros(n_areas)
    for name in model_names:
        ensemble_mean += weights[name] * z_post_means[name]

    # Within-model uncertainty: weighted mean of per-model posterior SDs
    within = np.zeros(n_areas)
    for name in model_names:
        within += weights[name] * z_post_sds[name]

    # Between-model uncertainty: std of posterior means across models
    means_mat = np.stack([z_post_means[n] for n in model_names], axis=1)  # (n_areas, n_models)
    between = means_mat.std(axis=1)

    total = np.sqrt(within**2 + between**2)

    # Approximate CI using normal quantile (posterior is roughly normal for large n)
    from scipy.stats import norm
    z_factor = norm.ppf((1 + ci) / 2)
    ci_lo = ensemble_mean - z_factor * total
    ci_hi = ensemble_mean + z_factor * total

    # Confidence tier: coefficient of variation of total uncertainty relative to |ensemble mean|
    # Low CV → High confidence. Thresholds chosen so roughly 1/3 of LSOAs fall in each tier.
    cv = total / (np.abs(ensemble_mean) + 1e-6)
    tier = np.where(cv < np.percentile(cv, 33), 'High',
           np.where(cv < np.percentile(cv, 67), 'Medium', 'Low'))

    df = pd.DataFrame({
        'lsoa_idx':             np.arange(n_areas),
        'z_ensemble_mean':      ensemble_mean,
        'z_within_uncertainty': within,
        'z_between_uncertainty': between,
        'z_total_uncertainty':  total,
        f'z_ci{int(ci*100)}_lo': ci_lo,
        f'z_ci{int(ci*100)}_hi': ci_hi,
        'confidence_tier':      tier,
    })

    if lsoa_codes is not None:
        df.insert(0, 'lsoa_code', np.asarray(lsoa_codes)[:n_areas])

    return df


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_z_sensitivity_map(gdf, sensitivity_df, col='z_std_across_models',
                           title='', figsize=(10, 8)):
    """
    Choropleth map of z model sensitivity.

    Parameters
    ----------
    gdf            : GeoDataFrame  — must be same length/order as sensitivity_df
    sensitivity_df : pd.DataFrame  — output of compute_z_model_sensitivity()[0]
    col            : str           — column to plot (default z_std_across_models)
    title          : str
    figsize        : tuple
    """
    plot_gdf = gdf.copy()
    plot_gdf[col] = sensitivity_df[col].values

    fig, ax = plt.subplots(figsize=figsize)
    plot_gdf.plot(column=col, ax=ax, cmap='YlOrRd', legend=True,
                  legend_kwds={'label': col, 'shrink': 0.6})
    ax.set_title(title or f'Model sensitivity: {col}')
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def plot_model_agreement_matrix(corr_df, title='Model-to-model z correlation'):
    """
    Annotated heatmap of model-to-model z posterior mean correlation.

    Parameters
    ----------
    corr_df : pd.DataFrame  — output of compute_model_agreement_matrix
    title   : str
    """
    fig, ax = plt.subplots(figsize=(max(5, len(corr_df) * 0.8 + 1),
                                     max(4, len(corr_df) * 0.8)))
    im = ax.imshow(corr_df.values, vmin=0.9, vmax=1.0, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Pearson r')

    models = list(corr_df.columns)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f'{corr_df.values[i, j]:.3f}', ha='center',
                    va='center', fontsize=7, color='black')

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_z_range_distribution(sensitivity_df, title=''):
    """
    Histogram of per-LSOA z_range_across_models (max - min across model z means).

    Parameters
    ----------
    sensitivity_df : pd.DataFrame  — output of compute_z_model_sensitivity()[0]
    title          : str
    """
    col   = 'z_range_across_models'
    vals  = sensitivity_df[col].dropna()
    p50   = vals.median()
    p90   = vals.quantile(0.90)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=60, color='steelblue', alpha=0.7, density=True)
    ax.axvline(p50, color='black', linestyle='--', linewidth=1,
               label=f'median={p50:.2f}')
    ax.axvline(p90, color='red',   linestyle='--', linewidth=1,
               label=f'90th pct={p90:.2f}')
    ax.set_xlabel('Range of z means across models (dwellings / year)')
    ax.set_ylabel('Density')
    ax.set_title(title or 'Per-LSOA model sensitivity: range of z posterior means')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_sensitivity_vs_disagreement(sensitivity_df, gdf,
                                      plan_cols, ben_cols, title=''):
    """
    Scatter: mean absolute source disagreement vs z model sensitivity.
    Reveals whether areas where PLD and BEN disagree are also areas
    where model choice matters most.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame  — output of compute_z_model_sensitivity()[0]
    gdf            : GeoDataFrame  — same length/order as sensitivity_df
    plan_cols      : list[str]     — planning data columns
    ben_cols       : list[str]     — BEN data columns
    title          : str
    """
    disagreement = (gdf[plan_cols].values - gdf[ben_cols].values)
    mean_abs_disagree = np.abs(disagreement).mean(axis=1)

    sens = sensitivity_df['z_std_across_models'].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_abs_disagree, sens, alpha=0.3, s=8, color='steelblue')

    # Bin means
    bins = np.percentile(mean_abs_disagree, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_idx = np.digitize(mean_abs_disagree, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    bin_means   = [sens[bin_idx == i].mean() for i in range(len(bins) - 1) if (bin_idx == i).any()]
    bin_centers = [bin_centers[i] for i in range(len(bins) - 1) if (bin_idx == i).any()]
    ax.plot(bin_centers, bin_means, color='red', linewidth=2, label='bin mean')

    ax.set_xlabel('Mean |PLD − BEN| per LSOA (dwellings / year)')
    ax.set_ylabel('z std across model posteriors (dwellings / year)')
    ax.set_title(title or 'Source disagreement vs model sensitivity')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_ensemble_mean_map(gdf, uncertainty_df, figsize=(14, 6)):
    """
    Side-by-side choropleth maps of ensemble mean z and total uncertainty.

    Parameters
    ----------
    gdf            : GeoDataFrame — same length/order as uncertainty_df
    uncertainty_df : pd.DataFrame — output of compute_decomposed_uncertainty
    figsize        : tuple
    """
    plot_gdf = gdf.copy().iloc[:len(uncertainty_df)]
    plot_gdf['z_ensemble_mean']     = uncertainty_df['z_ensemble_mean'].values
    plot_gdf['z_total_uncertainty'] = uncertainty_df['z_total_uncertainty'].values

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_gdf.plot(column='z_ensemble_mean', ax=axes[0], cmap='RdBu_r', legend=True,
                  legend_kwds={'label': 'dwellings / year', 'shrink': 0.6})
    axes[0].set_title('Ensemble mean z (dwelling delivery rate)')
    axes[0].set_axis_off()

    plot_gdf.plot(column='z_total_uncertainty', ax=axes[1], cmap='YlOrRd', legend=True,
                  legend_kwds={'label': 'dwellings / year', 'shrink': 0.6})
    axes[1].set_title('Total uncertainty (√within² + between²)')
    axes[1].set_axis_off()

    plt.tight_layout()
    return fig


def plot_estimate_vs_uncertainty(uncertainty_df, title=''):
    """
    Scatter of ensemble mean z vs total uncertainty, coloured by confidence tier.

    Shows which LSOAs have high estimates but also high uncertainty, and which
    are reliably estimated.

    Parameters
    ----------
    uncertainty_df : pd.DataFrame — output of compute_decomposed_uncertainty
    title          : str
    """
    tier_colours = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: point estimate vs total uncertainty, coloured by tier
    for tier, colour in tier_colours.items():
        mask = uncertainty_df['confidence_tier'] == tier
        axes[0].scatter(
            uncertainty_df.loc[mask, 'z_ensemble_mean'],
            uncertainty_df.loc[mask, 'z_total_uncertainty'],
            alpha=0.4, s=8, color=colour, label=tier,
        )
    axes[0].set_xlabel('Ensemble mean z (dwellings / year)')
    axes[0].set_ylabel('Total uncertainty (dwellings / year)')
    axes[0].set_title(title or 'Estimate vs uncertainty by confidence tier')
    axes[0].legend(title='Confidence', fontsize=9)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Right: within vs between decomposition scatter
    axes[1].scatter(
        uncertainty_df['z_within_uncertainty'],
        uncertainty_df['z_between_uncertainty'],
        alpha=0.3, s=8, color='steelblue',
    )
    lim = max(uncertainty_df['z_within_uncertainty'].max(),
              uncertainty_df['z_between_uncertainty'].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel('Within-model uncertainty (posterior SD)')
    axes[1].set_ylabel('Between-model uncertainty (std of means)')
    axes[1].set_title('Uncertainty decomposition\n(above diagonal = model choice dominates)')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return fig
