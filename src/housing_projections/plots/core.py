import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing_projections.config import COLOURS, INFER_YEARS


def select_sample_areas(D, n_sample=6, random_state=None):
    """
    Select n_sample areas spanning the range of census differences.
    Picks randomly from candidates near each quantile for variety.
    Pass random_state for reproducibility, or None for a different
    selection each call.
    """
    if random_state is None:
        random_state = int(time.time())

    rng = np.random.default_rng(random_state)

    quantiles  = np.linspace(0, 1, n_sample)
    D_series   = pd.Series(D)
    candidates = [
        D_series.sub(D_series.quantile(q)).abs().nsmallest(5).index.tolist()
        for q in quantiles
    ]

    return [int(rng.choice(c)) for c in candidates]


# ══════════════════════════════════════════════════════════════════════════════
# Primitives — single panel building blocks
# ══════════════════════════════════════════════════════════════════════════════

def plot_z_area(ax, z_post, idx, infer_years=INFER_YEARS,
                P_obs=None, E_obs=None, D=None, n_years=10,
                show_legend=False, alpha_ci=0.9):
    """
    Plot posterior z for a single area with optional observations and baselines.

    Parameters
    ----------
    ax          : matplotlib axis
    z_post      : (chains, draws, n_areas, n_years)
    idx         : int — area index
    P_obs       : (n_areas, n_years) — planning observations, optional
    E_obs       : (n_areas, n_years) — BEN observations, optional
    D           : (n_areas,) — census differences, optional
    n_years     : int
    show_legend : bool
    alpha_ci    : float — credible interval level (0.9 = 90%)
    """
    lo_pct = 100 * (1 - alpha_ci) / 2
    hi_pct = 100 * (1 - (1 - alpha_ci) / 2)

    z_area      = z_post[:, :, idx, :]
    z_mean      = z_area.mean(axis=(0, 1))
    z_lo        = np.percentile(z_area, lo_pct, axis=(0, 1))
    z_hi        = np.percentile(z_area, hi_pct, axis=(0, 1))
    z_sums      = z_area.sum(axis=-1).reshape(-1)
    z_sum_mean  = z_sums.mean()
    z_sum_lo    = np.percentile(z_sums, lo_pct)
    z_sum_hi    = np.percentile(z_sums, hi_pct)

    ax.plot(infer_years, z_mean, color=COLOURS['z'], marker='o',
            linewidth=1.5, label='Posterior mean z')
    ax.fill_between(infer_years, z_lo, z_hi,
                    alpha=0.2, color=COLOURS['z'],
                    label=f'{int(alpha_ci*100)}% CI')

    if P_obs is not None:
        ax.plot(infer_years, P_obs[idx], color=COLOURS['planning'],
                marker='s', alpha=0.7, linewidth=1.0, label='Planning')

    if E_obs is not None:
        ax.plot(infer_years, E_obs[idx], color=COLOURS['ben'],
                marker='^', alpha=0.7, linewidth=1.0, label='BEN')

    if D is not None:
        ax.axhline(D[idx] / n_years, color=COLOURS['baseline'],
                   linewidth=0.8, linestyle='--', alpha=0.5,
                   label=f'D/n ({D[idx]/n_years:.1f})')
        ax.axhline(z_sum_mean / n_years, color=COLOURS['posterior'],
                   linewidth=1.2, linestyle='-.',
                   label=f'Post. sum/n ({z_sum_mean:.0f})')

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_title(f'LSOA {idx}  (D={D[idx]:.0f}  '
                 f'post. sum={z_sum_mean:.0f} '
                 f'[{z_sum_lo:.0f}, {z_sum_hi:.0f}])'
                 if D is not None else f'LSOA {idx}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net dwelling change')
    ax.spines[['top', 'right']].set_visible(False)

    if show_legend:
        ax.legend(fontsize=7)

    return ax


def plot_predictive_distribution(ax, pred, obs, label, xlim=(-150, 150)):
    """
    Plot posterior/prior predictive vs observed for one source.

    Parameters
    ----------
    ax    : matplotlib axis
    pred  : array — predictive samples, any shape (will be flattened)
    obs   : array — observed values, any shape (will be flattened)
    label : str
    xlim  : tuple
    """
    ax.hist(pred.reshape(-1), bins=200, density=True, alpha=0.5,
            color='steelblue', label='Predictive')
    ax.hist(obs.reshape(-1),  bins=200, density=True, alpha=0.5,
            color='coral',     label='Observed')
    ax.set_xlim(xlim)
    ax.set_xlabel('Net dwelling change')
    ax.set_title(label)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()

    return ax


def plot_residuals_by_year(ax, residuals, label, infer_years=INFER_YEARS):
    """
    Plot mean residuals by year for one source.

    Parameters
    ----------
    ax         : matplotlib axis
    residuals  : (n_areas, n_years)
    label      : str
    """
    ax.plot(infer_years, residuals.mean(axis=0), marker='o',
            color=COLOURS['planning'] if 'plan' in label.lower()
            else COLOURS['ben'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Mean residuals by year: {label}')
    ax.spines[['top', 'right']].set_visible(False)

    return ax


def plot_residuals_vs_D(ax, residuals, D, label):
    """
    Plot mean residuals vs census diff for one source.

    Parameters
    ----------
    ax        : matplotlib axis
    residuals : (n_areas, n_years)
    D         : (n_areas,)
    label     : str
    """
    ax.scatter(D, residuals.mean(axis=1), alpha=0.3, s=5,
               color=COLOURS['planning'] if 'plan' in label.lower()
               else COLOURS['ben'])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Census diff (D)')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Residuals vs census diff: {label}')
    ax.spines[['top', 'right']].set_visible(False)

    return ax


def plot_uncertainty_vs_disagreement(ax, z_post, P_obs, E_obs,
                                      threshold=5, alpha_ci=0.9):
    """
    Plot posterior CI width vs source disagreement.

    Parameters
    ----------
    ax        : matplotlib axis
    z_post    : (chains, draws, n_areas, n_years)
    P_obs     : (n_areas, n_years)
    E_obs     : (n_areas, n_years)
    threshold : float — only show points where at least one source > threshold
    alpha_ci  : float
    """
    lo_pct = 100 * (1 - alpha_ci) / 2
    hi_pct = 100 * (1 - (1 - alpha_ci) / 2)

    z_lo   = np.percentile(z_post, lo_pct, axis=(0, 1))
    z_hi   = np.percentile(z_post, hi_pct, axis=(0, 1))

    ci_width     = z_hi - z_lo
    disagreement = np.abs(P_obs - E_obs)
    mask         = (np.abs(P_obs) > threshold) | (np.abs(E_obs) > threshold)

    ax.scatter(disagreement[mask], ci_width[mask],
               alpha=0.2, s=3, color='steelblue')
    ax.set_xlabel('|Planning - BEN|')
    ax.set_ylabel(f'{int(alpha_ci*100)}% CI width')
    ax.set_title(f'Uncertainty vs source disagreement (|obs|>{threshold})')
    ax.spines[['top', 'right']].set_visible(False)

    return ax


def plot_parameter_posteriors(ax, trace, var_names, colors=None):
    """
    Plot posterior distributions for scalar parameters on a single axis.

    Parameters
    ----------
    ax        : matplotlib axis
    trace     : az.InferenceData
    var_names : list of str
    colors    : list of str, optional
    """
    if colors is None:
        colors = plt.cm.tab10.colors

    for var, color in zip(var_names, colors):
        samples = trace.posterior[var].values.ravel()
        ax.hist(samples, bins=50, density=True, alpha=0.5,
                color=color, label=var)

    ax.set_xlabel('Value')
    ax.set_title('Posterior: parameters')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    return ax


# ══════════════════════════════════════════════════════════════════════════════
# Higher-level composite plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_areas(trace, data, n_sample=6, title='',
                      infer_years=INFER_YEARS, alpha_ci=0.9,
                      random_state=None):
    """
    Plot posterior z vs observations for a sample of areas spanning
    the range of census differences.
    """
    z_post = trace.posterior['z'].values
    D      = data['D']
    P_obs  = data['P_obs']
    E_obs  = data['E_obs']

    sample_idx = select_sample_areas(D, n_sample=n_sample,
                                     random_state=random_state)

    nrows = int(np.ceil(n_sample / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))

    for i, (ax, idx) in enumerate(zip(axes.ravel(), sample_idx)):
        plot_z_area(ax, z_post, idx,
                    infer_years=infer_years,
                    P_obs=P_obs, E_obs=E_obs, D=D,
                    n_years=data['n_years'],
                    show_legend=(i == 0),
                    alpha_ci=alpha_ci)

    for ax in axes.ravel()[n_sample:]:
        ax.set_visible(False)

    plt.suptitle(f'{title} — posterior z vs planning and BEN')
    plt.tight_layout()

    return fig, axes


def plot_posterior_predictive(post_pred, data, title=''):
    """
    Plot posterior predictive vs observed for planning and BEN.
    """
    P_post = post_pred.posterior_predictive['P_like'].values
    E_post = post_pred.posterior_predictive['E_like'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_predictive_distribution(axes[0], P_post, data['P_obs'], 'Planning')
    plot_predictive_distribution(axes[1], E_post, data['E_obs'], 'BEN')

    plt.suptitle(f'{title} — posterior predictive vs observed')
    plt.tight_layout()

    return fig, axes


def plot_prior_predictive(prior, data, title=''):
    """
    Plot prior predictive vs observed for planning and BEN.
    """
    P_prior = prior.prior_predictive['P_like'].values
    E_prior = prior.prior_predictive['E_like'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_predictive_distribution(axes[0], P_prior, data['P_obs'],
                                 'Planning (prior predictive)')
    plot_predictive_distribution(axes[1], E_prior, data['E_obs'],
                                 'BEN (prior predictive)')

    plt.suptitle(f'{title} — prior predictive vs observed')
    plt.tight_layout()

    return fig, axes


def plot_residual_analysis(trace, data, title='', quantile_clip=0.99):
    """
    Six-panel residual analysis treating planning and BEN symmetrically:
    - Residual distributions (clipped to quantile_clip for readability)
    - Mean and median residuals by year
    - Residuals vs census diff

    Parameters
    ----------
    trace         : az.InferenceData
    data          : dict
    title         : str
    quantile_clip : float — clip x-axis to this quantile for distribution plots
    """
    z_post      = trace.posterior['z'].values
    z_mean_post = z_post.mean(axis=(0, 1))

    resid_plan  = data['P_obs'] - z_mean_post
    resid_ben   = data['E_obs'] - z_mean_post

    sources = [
        (resid_plan, 'Planning', COLOURS['planning']),
        (resid_ben,  'BEN',      COLOURS['ben']),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # ── Row 1: residual distributions ─────────────────────────────────────
    for ax, (resid, label, color) in zip(axes[0], sources):
        clip = np.quantile(np.abs(resid), quantile_clip)
        ax.hist(resid.ravel(), bins=100, density=True,
                color=color, alpha=0.7,
                range=(-clip, clip))
        ax.axvline(0,               color='black', linewidth=0.8)
        ax.axvline(resid.mean(),    color='red',   linewidth=0.8,
                   linestyle='--', label=f'mean={resid.mean():.2f}')
        ax.axvline(np.median(resid), color='darkred', linewidth=0.8,
                   linestyle=':',  label=f'median={np.median(resid):.2f}')
        ax.set_title(f'Residuals: {label}')
        ax.set_xlabel('Observed - posterior mean z')
        ax.set_ylabel('Density')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    # ── Row 2: mean and median residuals by year ───────────────────────────
    for ax, (resid, label, color) in zip(axes[1], sources):
        mean_by_year   = resid.mean(axis=0)
        median_by_year = np.median(resid, axis=0)
        se_by_year     = resid.std(axis=0) / np.sqrt(data['n_areas'])

        ax.plot(INFER_YEARS, mean_by_year,
                marker='o', color=color, linewidth=1.5, label='Mean')
        ax.fill_between(
            INFER_YEARS,
            mean_by_year - se_by_year,
            mean_by_year + se_by_year,
            alpha=0.2, color=color, label='±1 SE'
        )
        ax.plot(INFER_YEARS, median_by_year,
                marker='o', color=color, linewidth=1.5,
                linestyle=':', label='Median')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'Mean and median residuals by year: {label}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Residual')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    # ── Row 3: residuals vs census diff ────────────────────────────────────
    for ax, (resid, label, color) in zip(axes[2], sources):
        mean_resid   = resid.mean(axis=1)
        median_resid = np.median(resid, axis=1)
        clip         = np.quantile(np.abs(mean_resid), quantile_clip)

        ax.scatter(data['D'], mean_resid,
                   alpha=0.3, s=5, color=color,  label='Mean')
        ax.scatter(data['D'], median_resid,
                   alpha=0.3, s=5, color='black', label='Median',
                   marker='x')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylim(-clip, clip)
        ax.set_title(f'Mean and median residuals vs census diff: {label}')
        ax.set_xlabel('Census diff (D)')
        ax.set_ylabel('Residual')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    plt.suptitle(f'{title} — residual analysis')
    plt.tight_layout()

    return fig, axes


def plot_parameter_trace(trace, var_names, title=''):
    """
    Plot trace and posterior for scalar parameters.
    """
    axes = az.plot_trace(trace, var_names=var_names)
    fig  = plt.gcf()
    plt.suptitle(f'{title} — parameter traces')
    plt.tight_layout()
    return fig, axes
