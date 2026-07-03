import numpy as np
import matplotlib.pyplot as plt

from housing_projections.diagnostics import (
    full_diagnostics,
    compute_lag_weights,
    compute_lag_residuals,
    compute_spatial_misallocation_stats,
    _check_census_constraint,
    _check_calibration,
    _check_morans_i,
)
from housing_projections.plots.core import (
    plot_prior_predictive,
    plot_parameter_trace,
    plot_posterior_predictive,
    plot_sample_areas,
    plot_residual_analysis,
    plot_uncertainty_vs_disagreement,
    plot_residuals_by_year,
    plot_residuals_vs_D,
)
from housing_projections.plots.model import (
    plot_lag_weights,
    plot_lag_residuals,
    plot_lag_residuals_by_year,
    plot_lag_effect,
    plot_lag_shift,
    plot_missingness_posterior,
    plot_zero_inflation_check,
    plot_zero_residuals,
    plot_missing_statistics,
    plot_missingness_effect_on_z,
    plot_negative_tail_comparison,
    plot_twocomp_diagnostics,
    plot_spatial_diagnostics,
)

__all__ = ["full_report", "run_comparison_reports"]

# ── Model diagnostic registry ─────────────────────────────────────────────────
# Maps model name to list of extra diagnostic functions to call after
# the standard suite. Each function takes (trace, data, title, model).
# Add new models here — no changes needed elsewhere.

def plot_lag_diagnostics(trace, data, title='M3', model=None):
    """
    Orchestrate all M3 lag diagnostic plots.
    Computes lag weights and residuals via diagnostics, then passes
    pre-computed results to the individual plot functions.
    """
    if 'lambda_weights' not in trace.posterior:
        print(f"{title}: lambda_weights fixed — skipping lag diagnostics")
        return
    lag_results = compute_lag_weights(trace, verbose=True)
    resids      = compute_lag_residuals(trace, data)
    plot_lag_weights(lag_results, title=title)
    plot_lag_residuals(resids, title=title)
    plot_lag_residuals_by_year(resids, title=title)
    plot_lag_effect(trace, data, title=title)
    plot_lag_shift(trace, data, title=title)


def plot_missingness_diagnostics(trace, data, title='M4', model=None,
                                  trace_before=None, post_pred_before=None,
                                  post_pred_after=None):
    """
    Orchestrate all M4 missingness diagnostic plots.
    Computes lag residuals via diagnostics, then passes pre-computed
    results to plot functions. Pass trace_before and post_pred_before
    for M3 vs M4 comparison plots.
    """
    lambda_weights = getattr(model, 'lambda_weights', None)
    plot_missingness_posterior(trace, title=title)
    plot_zero_inflation_check(trace, data, title=title)
    resids = compute_lag_residuals(trace, data,
                                               lambda_weights=lambda_weights)
    plot_zero_residuals(resids, data['P_obs'], title=title)
    plot_missing_statistics(trace, data, title=title)

    if trace_before is not None:
        plot_missingness_effect_on_z(trace_before, trace, data, title=title)

    if post_pred_before is not None and post_pred_after is not None:
        plot_negative_tail_comparison(post_pred_before, post_pred_after, data, title=title)


def plot_spatial_diagnostics_report(trace, data, title='', model=None):
    stats_dict = compute_spatial_misallocation_stats(trace, data)
    plot_spatial_diagnostics(stats_dict, title=title)


MODEL_DIAGNOSTICS = {
    'M3':  [plot_lag_diagnostics],
    'M4':  [plot_lag_diagnostics,
            plot_missingness_diagnostics],
    'M5':  [plot_lag_diagnostics,
            plot_missingness_diagnostics],
    'M5b': [plot_lag_diagnostics,
            plot_missingness_diagnostics,
            lambda trace, data, title='', model=None: plot_twocomp_diagnostics(trace, data, title=title)],
    'M6':  [plot_missingness_diagnostics,
            plot_spatial_diagnostics_report],
}


# ── Model comparison registry ─────────────────────────────────────────────────
# Maps tuple of model names to comparison report function.

def get_model_comparisons():
    return {
        ('M3', 'M4'):  missingness_comparison_report,
        ('M4', 'M5'):  missingness_comparison_report,
        ('M5', 'M5b'): missingness_comparison_report,
        ('M5', 'M6'):  spatial_misallocation_comparison,
    }


def full_report(trace, data, post_pred, prior=None,
                model=None, title='', random_state=None):
    """
    Full diagnostic report for a single model.
    Model-specific diagnostics are looked up from MODEL_DIAGNOSTICS registry.
    """
    t = title or (model.name if model is not None else '')

    # Prior predictive
    if prior is not None:
        z_prior = prior.prior['z'].values
        print(f"\nPrior predictive summary:")
        print(f"  z mean:     {z_prior.mean():.3f}")
        print(f"  z sd:       {z_prior.std():.3f}")
        print(f"  z 99th:     {np.percentile(z_prior, 99):.3f}")
        print(f"  z 1st:      {np.percentile(z_prior,  1):.3f}")
        print(f"  P(|z| < 3): {(np.abs(z_prior) < 3).mean():.3f}")
        plot_prior_predictive(prior, data, title=t)

    # Sampling diagnostics
    full_diagnostics(trace, data, model=model, verbose=True)

    # Parameter traces
    if model is not None:
        plot_parameter_trace(trace, model.var_names, title=t)

    # Posterior predictive vs observed
    plot_posterior_predictive(post_pred, data, title=t)

    # Sample area plots
    plot_sample_areas(trace, data, title=t, random_state=random_state)

    # Residual analysis
    plot_residual_analysis(trace, data, title=t)

    # Uncertainty vs disagreement
    z_post = trace.posterior['z'].values
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_uncertainty_vs_disagreement(
        ax, z_post, data['P_obs'], data['E_obs'])
    plt.suptitle(f'{t} — uncertainty vs source disagreement')
    plt.tight_layout()
    plt.show()

    # Census constraint check
    census    = _check_census_constraint(trace, data, verbose=True)
    z_sums    = z_post.sum(axis=-1).reshape(-1, data['n_areas'])
    residuals = (z_sums - data['D'][None, :]).ravel()
    fig, ax   = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=100, density=True, color='steelblue', alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('z sum - D')
    ax.set_title(f'{t} — census constraint violations (posterior)')
    ax.text(0.05, 0.95,
            f'mean={census["mean_violation"]:.3f}\n'
            f'max={census["max_violation"]:.3f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

    # Residuals by year
    z_mean_post = z_post.mean(axis=(0, 1))
    resid_plan  = data['P_obs'] - z_mean_post
    resid_ben   = data['E_obs'] - z_mean_post

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, resid, label in zip(axes, [resid_plan, resid_ben],
                                 ['Planning', 'BEN']):
        plot_residuals_by_year(ax, resid, label)
    plt.suptitle(f'{t} — mean residuals by year')
    plt.tight_layout()
    plt.show()

    # Residuals vs census diff
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, resid, label in zip(axes, [resid_plan, resid_ben],
                                 ['Planning', 'BEN']):
        plot_residuals_vs_D(ax, resid, data['D'], label)
    plt.suptitle(f'{t} — residuals vs census diff')
    plt.tight_layout()
    plt.show()

    # Calibration and Moran's I
    _check_calibration(trace, data, verbose=True)
    _check_morans_i(trace, data, verbose=True)

    # Model-specific diagnostics from registry
    if model is not None:
        for fn in MODEL_DIAGNOSTICS.get(model.name, []):
            fn(trace, data, title=t, model=model)


def run_comparison_reports(models, traces, data, post_preds):
    """
    Run all applicable model comparison reports based on which
    models have been sampled. Looks up from MODEL_COMPARISONS registry.
    """
    sampled = set(models.keys())
    for (name_before, name_after), report_fn in get_model_comparisons().items():
        if name_before in sampled and name_after in sampled:
            print(f"\nRunning comparison: {name_before} vs {name_after}")
            report_fn(
                traces[name_before], traces[name_after], data,
                post_preds[name_before], post_preds[name_after],
                title=f'{name_before} vs {name_after}'
            )


# ── missingness comparisons ──────────────────────────────────────────────────────
def missingness_comparison_report(trace_before, trace_after, data,
                                   post_pred_before, post_pred_after,
                                   title=''):
    # Extract model names from title e.g. 'M3 vs M4' -> 'M3', 'M4'
    parts        = title.split(' vs ') if ' vs ' in title else ['before', 'after']
    label_before = parts[0].strip()
    label_after  = parts[1].strip()

    plot_negative_tail_comparison(
        post_pred_before, post_pred_after, data,
        title=title,
        label_before=label_before,
        label_after=label_after)

    plot_missingness_effect_on_z(
        trace_before, trace_after, data,
        title=title,
        label_before=label_before,
        label_after=label_after)


def spatial_misallocation_comparison(trace_m5, trace_m6, data,
                                      post_pred_m5, post_pred_m6,
                                      title='M5 vs M6'):
    """
    Compare M5 and M6 to show the effect of adding spatial misallocation.
    """
    parts        = title.split(' vs ')
    label_before = parts[0].strip()
    label_after  = parts[1].strip()

    # ── Alpha spatial posterior ───────────────────────────────────────────
    stats_m6 = compute_spatial_misallocation_stats(
        trace_m6, data)
    plot_spatial_diagnostics(stats_m6, title=label_after)

    # ── Moran's I on planning residuals ───────────────────────────────────
    print(f"\nMoran's I on planning residuals — {label_before}:")
    _check_morans_i(trace_m5, data, verbose=True)

    print(f"\nMoran's I on planning residuals — {label_after}:")
    _check_morans_i(trace_m6, data, verbose=True)

    # ── Change in z inference ─────────────────────────────────────────────
    plot_missingness_effect_on_z(
        trace_m5, trace_m6, data,
        title=title,
        label_before=label_before,
        label_after=label_after)

    # ── Negative tail ─────────────────────────────────────────────────────
    plot_negative_tail_comparison(
        post_pred_m5, post_pred_m6, data,
        title=title,
        label_before=label_before,
        label_after=label_after)

    # ── Spatial distribution of z changes ────────────────────────────────
    z_m5 = trace_m5.posterior['z'].values.mean(axis=(0, 1))
    z_m6 = trace_m6.posterior['z'].values.mean(axis=(0, 1))
    diff  = (z_m6 - z_m5).mean(axis=1)   # mean change per area

    gdf   = data['gdf'].copy()
    gdf['z_diff'] = diff

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmax = np.abs(diff).quantile(0.95) if hasattr(diff, 'quantile') \
           else np.percentile(np.abs(diff), 95)

    gdf.plot(column='z_diff', ax=axes[0], cmap='RdBu',
             vmin=-vmax, vmax=vmax, legend=True)
    axes[0].set_title(f'Mean change in z: {label_after} - {label_before}')
    axes[0].set_axis_off()

    axes[1].hist(diff, bins=50, color='steelblue', alpha=0.7, density=True)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].axvline(diff.mean(), color='red', linestyle='--',
                    linewidth=0.8, label=f'mean={diff.mean():.3f}')
    axes[1].set_xlabel('Mean change in z per area')
    axes[1].set_title('Distribution of z changes')
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].legend(fontsize=8)

    plt.suptitle(f'{title} — spatial effect on z inference')
    plt.tight_layout()
    plt.show()