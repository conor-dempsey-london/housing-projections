"""Model-specific diagnostic plots — one section per model variant."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from housing_projections.config import INFER_YEARS


# ── M3 — spatial lag ──────────────────────────────────────────────────────────

def plot_lag_weights(lag_results, title=''):
    """
    Plot posterior distributions and bar chart of M3 lag weights.

    Parameters
    ----------
    lag_results : dict returned by diagnostics.compute_lag_weights
    """
    means       = lag_results['means']
    lo          = lag_results['lo']
    hi          = lag_results['hi']
    n_lags      = lag_results['n_lags']
    lags        = list(range(n_lags))
    lambda_flat = lag_results['lambda_flat']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for k in range(n_lags):
        ax.hist(lambda_flat[:, k], bins=50, density=True,
                alpha=0.6, label=f'lag {k} (mean={means[k]:.3f})')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Density')
    ax.set_title('Posterior lag weight distributions')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(lags, means, color='steelblue', alpha=0.7)
    ax.errorbar(lags, means,
                yerr=[means - lo, hi - means],
                fmt='none', color='black', capsize=4)
    ax.set_xlabel('Lag (years)')
    ax.set_ylabel('Posterior mean weight')
    ax.set_title(f'Estimated planning lag weights\n'
                 f'implied mean lag = {lag_results["implied_mean_lag"]:.2f} years')
    ax.set_xticks(lags)
    ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle(f'{title} — lag weights')
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_lag_residuals(resids, title=''):
    """
    Compare planning residuals with and without lag correction.

    Parameters
    ----------
    resids : dict returned by diagnostics.compute_lag_residuals
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, label in zip(
        axes,
        ['no_lag', 'with_lag'],
        ['Without lag (z)', 'With lag (P_mean)']
    ):
        resid = resids[key]
        clip  = np.quantile(np.abs(resid), 0.99)
        ax.hist(resid.ravel(), bins=100, density=True,
                color='steelblue', alpha=0.7, range=(-clip, clip))
        ax.axvline(0,               color='black',   linewidth=0.8)
        ax.axvline(resid.mean(),    color='red',     linewidth=0.8,
                   linestyle='--', label=f'mean={resid.mean():.2f}')
        ax.axvline(np.median(resid), color='darkred', linewidth=0.8,
                   linestyle=':',  label=f'median={np.median(resid):.2f}')
        ax.set_title(f'Planning residuals: {label}')
        ax.set_xlabel('Observed - predicted')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    plt.suptitle(f'{title} — lag residuals')
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_lag_residuals_by_year(resids, title=''):
    """
    Compare mean planning residuals by year with and without lag correction.

    Parameters
    ----------
    resids : dict returned by diagnostics.compute_lag_residuals
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(INFER_YEARS, resids['no_lag'].mean(axis=0),
            marker='o', color='steelblue', label='Without lag')
    ax.plot(INFER_YEARS, resids['with_lag'].mean(axis=0),
            marker='o', color='coral',     label='With lag')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean residual')
    ax.set_title('Mean planning residuals by year')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_lag_effect(trace, data, n_sample=6, title='M3'):
    """
    For a sample of areas, plot observed planning/BEN, posterior mean z,
    and the posterior mean lagged planning prediction (P_mean).
    """
    z_post      = trace.posterior['z'].values
    lambda_post = trace.posterior['lambda_weights'].values

    z_mean      = z_post.mean(axis=(0, 1))
    lambda_mean = lambda_post.mean(axis=(0, 1))
    n_lags      = len(lambda_mean)
    n_years     = data['n_years']

    P_mean_post = np.zeros_like(z_mean)
    for t in range(n_years):
        for k in range(n_lags):
            t_src               = max(t - k, 0)
            P_mean_post[:, t]  += lambda_mean[k] * z_mean[:, t_src]

    D          = data['D']
    sample_idx = [
        pd.Series(D).sub(pd.Series(D).quantile(q)).abs().idxmin()
        for q in np.linspace(0, 1, n_sample)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, (ax, idx) in enumerate(zip(axes.ravel(), sample_idx)):
        z_area      = z_post[:, :, idx, :]
        z_mean_area = z_area.mean(axis=(0, 1))
        z_lo        = np.percentile(z_area, 5,  axis=(0, 1))
        z_hi        = np.percentile(z_area, 95, axis=(0, 1))

        ax.plot(INFER_YEARS, data['P_obs'][idx], color='steelblue',
                marker='s', alpha=0.5, linewidth=1.0, linestyle='--',
                label='P observed')
        ax.plot(INFER_YEARS, data['E_obs'][idx], color='coral',
                marker='^', alpha=0.5, linewidth=1.0, linestyle='--',
                label='E observed')
        ax.plot(INFER_YEARS, z_mean_area, color='black',
                marker='o', linewidth=1.5, label='Posterior mean z')
        ax.fill_between(INFER_YEARS, z_lo, z_hi,
                        alpha=0.15, color='black', label='90% CI')
        ax.plot(INFER_YEARS, P_mean_post[idx], color='steelblue',
                marker='s', linewidth=1.5, linestyle='-',
                label='P_mean (lagged)')
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.axhline(D[idx] / n_years, color='green', linewidth=0.8,
                   linestyle='--', alpha=0.5,
                   label=f'D/n ({D[idx]/n_years:.1f})')

        ax.set_title(f'LSOA {idx}  (D={D[idx]:.0f})', fontsize=8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Net dwelling change')
        ax.spines[['top', 'right']].set_visible(False)

        if i == 0:
            ax.legend(fontsize=7)

    plt.suptitle(f'{title} — lag effect on planning prediction\n'
                 f'implied mean lag = '
                 f'{sum(k * lambda_mean[k] for k in range(n_lags)):.2f} years')
    plt.tight_layout()
    plt.show()


def plot_lag_shift(trace, data, title='M3'):
    """
    For each year, show how much the lag shifts the planning prediction
    relative to z — i.e. the difference between P_mean and z.
    """
    z_post      = trace.posterior['z'].values
    lambda_post = trace.posterior['lambda_weights'].values

    z_mean      = z_post.mean(axis=(0, 1))
    lambda_mean = lambda_post.mean(axis=(0, 1))
    n_lags      = len(lambda_mean)
    n_years     = data['n_years']

    P_mean_post = np.zeros_like(z_mean)
    for t in range(n_years):
        for k in range(n_lags):
            t_src              = max(t - k, 0)
            P_mean_post[:, t] += lambda_mean[k] * z_mean[:, t_src]

    shift = P_mean_post - z_mean

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.boxplot([shift[:, t] for t in range(n_years)],
               positions=INFER_YEARS, widths=0.6, showfliers=False)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('P_mean - z (lag shift)')
    ax.set_title('Distribution of lag shift by year')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    ax.plot(INFER_YEARS, shift.mean(axis=0),
            marker='o', color='steelblue', label='Mean shift')
    ax.fill_between(INFER_YEARS,
                    np.percentile(shift, 5,  axis=0),
                    np.percentile(shift, 95, axis=0),
                    alpha=0.2, color='steelblue', label='90% CI')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('P_mean - z (lag shift)')
    ax.set_title('Mean lag shift by year')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle(f'{title} — temporal redistribution from lag model')
    plt.tight_layout()
    plt.show()


# ── M4 — missingness / zero-inflation ─────────────────────────────────────────

def plot_missingness_posterior(trace, title=''):
    """
    Posterior distribution of missingness parameters.
    Handles both symmetric (pi_miss) and asymmetric
    (pi_miss_pos, pi_miss_neg) parameterisations.
    """
    has_symmetric  = 'pi_miss'     in trace.posterior
    has_asymmetric = 'pi_miss_pos' in trace.posterior

    if has_asymmetric:
        params = [
            ('pi_miss_pos', 7, 3, 'completions'),
            ('pi_miss_neg', 8, 2, 'demolitions'),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    elif has_symmetric:
        params = [('pi_miss', 2, 8, 'all observations')]
        fig, axes = plt.subplots(1, 1, figsize=(7, 4))
        axes = [axes]
    else:
        print(f"No missingness parameters found in trace for {title}")
        return

    for ax, (param, prior_alpha, prior_beta, label) in zip(axes, params):
        post = trace.posterior[param].values.ravel()
        ax.hist(post, bins=50, density=True, color='steelblue',
                alpha=0.7, label='Posterior')
        x     = np.linspace(0, 1, 200)
        prior = stats.beta.pdf(x, prior_alpha, prior_beta)
        ax.plot(x, prior, color='red', linestyle='--', linewidth=1.0,
                label=f'Prior Beta({prior_alpha},{prior_beta})')
        ax.axvline(post.mean(), color='black', linestyle='--',
                   linewidth=0.8, label=f'mean={post.mean():.3f}')
        ax.set_xlabel(param)
        ax.set_title(f'{label}')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    plt.suptitle(f'{title} — missingness parameter posteriors')
    plt.tight_layout()
    plt.show()


def plot_zero_inflation_check(trace, data, title=''):
    """
    Compare observed zero frequency vs model-predicted zero frequency
    in planning data, broken down by sign of posterior z.
    """
    z_post      = trace.posterior['z'].values
    z_mean_post = z_post.mean(axis=(0, 1))
    P_obs       = data['P_obs']
    is_zero_obs = np.abs(P_obs) < 1e-6

    if 'pi_miss' in trace.posterior:
        pi_pred            = float(trace.posterior['pi_miss'].values.ravel().mean())
        pred_zero_rate_pos = pi_pred
        pred_zero_rate_neg = pi_pred
    elif 'pi_miss_pos' in trace.posterior:
        pred_zero_rate_pos = float(
            trace.posterior['pi_miss_pos'].values.ravel().mean())
        pred_zero_rate_neg = float(
            trace.posterior['pi_miss_neg'].values.ravel().mean())
    else:
        print(f"No missingness parameters found in trace for {title}")
        return

    pos_mask          = z_mean_post > 0
    neg_mask          = z_mean_post < 0
    obs_zero_rate_pos = is_zero_obs[pos_mask].mean()
    obs_zero_rate_neg = is_zero_obs[neg_mask].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    x       = np.arange(2)
    width   = 0.35

    ax.bar(x - width / 2,
           [obs_zero_rate_pos,  obs_zero_rate_neg],
           width=width, color=['steelblue', 'coral'],
           alpha=0.7, label='Observed')
    ax.bar(x + width / 2,
           [pred_zero_rate_pos, pred_zero_rate_neg],
           width=width, color=['steelblue', 'coral'],
           alpha=0.4, label='Predicted')

    ax.set_xticks(x)
    ax.set_xticklabels(['z > 0 (completions)', 'z < 0 (demolitions)'])
    ax.set_ylabel('Fraction of zero observations')
    ax.set_title(f'{title} — observed vs predicted zero rate by sign of z')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    for i, val in enumerate([obs_zero_rate_pos, obs_zero_rate_neg]):
        ax.text(i - width / 2, val + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)
    for i, val in enumerate([pred_zero_rate_pos, pred_zero_rate_neg]):
        ax.text(i + width / 2, val + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"Observed zero rate where z>0:  {obs_zero_rate_pos:.3f}")
    print(f"Observed zero rate where z<0:  {obs_zero_rate_neg:.3f}")
    print(f"Predicted zero rate where z>0: {pred_zero_rate_pos:.3f}")
    print(f"Predicted zero rate where z<0: {pred_zero_rate_neg:.3f}")


def plot_missingness_effect_on_z(
        trace_before,
        trace_after,
        data,
        title='',
        label_before='Before',
        label_after='After'):
    """
    For areas where planning shows zeros but BEN shows non-zero,
    compare posterior z between two models.
    """
    P_obs = data['P_obs']
    E_obs = data['E_obs']

    plan_zero_ben_nonzero = (
        (np.abs(P_obs) < 1e-6) & (np.abs(E_obs) > 5)
    )
    area_mask = plan_zero_ben_nonzero.any(axis=1)

    z_before = trace_before.posterior['z'].values.mean(axis=(0, 1))
    z_after  = trace_after.posterior['z'].values.mean(axis=(0, 1))
    diff     = z_after[area_mask] - z_before[area_mask]

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(z_before[area_mask].ravel(), z_after[area_mask].ravel(),
               alpha=0.2, s=5, color='steelblue')
    lims = [min(z_before[area_mask].min(), z_after[area_mask].min()),
            max(z_before[area_mask].max(), z_after[area_mask].max())]
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f'Posterior mean z ({label_before})')
    ax.set_ylabel(f'Posterior mean z ({label_after})')
    ax.set_title('z comparison: planning=0, BEN≠0 areas')
    ax.spines[['top', 'right']].set_visible(False)

    ax   = axes[1]
    clip = np.quantile(np.abs(diff), 0.99)
    ax.hist(diff.ravel(), bins=100, density=True,
            color='steelblue', alpha=0.7, range=(-clip, clip))
    ax.axvline(0,           color='black', linewidth=0.8)
    ax.axvline(diff.mean(), color='red',   linestyle='--',
               linewidth=0.8, label=f'mean={diff.mean():.2f}')
    ax.set_xlabel(f'z({label_after}) - z({label_before})')
    ax.set_title(f'Change in posterior z from {label_before} to {label_after}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle(f'{title} — missingness effect on z inference')
    plt.tight_layout()
    plt.show()


def plot_zero_residuals(resids, P_obs, title=''):
    """
    Residuals for zero vs non-zero planning observations.

    Parameters
    ----------
    resids : dict returned by diagnostics.compute_lag_residuals
    P_obs  : (n_areas, n_years)
    """
    resid_plan = resids['with_lag']
    is_zero    = np.abs(P_obs) < 1e-6

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, mask, label in zip(
        axes,
        [is_zero, ~is_zero],
        ['Zero planning observations', 'Non-zero planning observations']
    ):
        resid_subset = resid_plan[mask]
        clip         = np.quantile(np.abs(resid_subset), 0.99)
        ax.hist(resid_subset, bins=100, density=True,
                color='steelblue', alpha=0.7, range=(-clip, clip))
        ax.axvline(0,                       color='black',   linewidth=0.8)
        ax.axvline(resid_subset.mean(),     color='red',     linestyle='--',
                   linewidth=0.8, label=f'mean={resid_subset.mean():.2f}')
        ax.axvline(np.median(resid_subset), color='darkred', linestyle=':',
                   linewidth=0.8,
                   label=f'median={np.median(resid_subset):.2f}')
        ax.set_xlabel('P_obs - P_mean')
        ax.set_title(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    plt.suptitle(f'{title} — residuals for zero vs non-zero planning observations')
    plt.tight_layout()
    plt.show()


def plot_negative_tail_comparison(
        post_pred_before,
        post_pred_after,
        data,
        title='',
        label_before='Before',
        label_after='After'):
    """
    Compare the negative tail of the planning predictive distribution
    between two models, and against observed data.
    """
    P_obs         = data['P_obs'].ravel()
    P_pred_before = post_pred_before.posterior_predictive['P_like'].values.reshape(-1)
    P_pred_after  = post_pred_after.posterior_predictive['P_like'].values.reshape(-1)

    clip = 20

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(P_obs,         bins=50, density=True, alpha=0.5,
            color='black',     label='Observed',    range=(-clip, clip))
    ax.hist(P_pred_before, bins=50, density=True, alpha=0.4,
            color='steelblue', label=label_before,  range=(-clip, clip))
    ax.hist(P_pred_after,  bins=50, density=True, alpha=0.4,
            color='coral',     label=label_after,   range=(-clip, clip))
    ax.set_xlabel('Planning observation')
    ax.set_title('Full distribution')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax      = axes[1]
    clip_neg = np.percentile(np.abs(P_obs[P_obs < 0]), 99) if (P_obs < 0).any() else 50
    ax.hist(P_obs[P_obs < 0],                     bins=50, density=True, alpha=0.5,
            color='black',     label='Observed',   range=(-clip_neg, 0))
    ax.hist(P_pred_before[P_pred_before < 0],     bins=50, density=True, alpha=0.4,
            color='steelblue', label=label_before, range=(-clip_neg, 0))
    ax.hist(P_pred_after[P_pred_after < 0],       bins=50, density=True, alpha=0.4,
            color='coral',     label=label_after,  range=(-clip_neg, 0))
    ax.set_xlabel('Planning observation (negative only)')
    ax.set_title('Negative tail')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle(f'{title} — negative tail of planning distribution')
    plt.tight_layout()
    plt.show()

    print(f"Fraction negative — observed:      {(P_obs < 0).mean():.4f}")
    print(f"Fraction negative — {label_before} predictive: "
          f"{(P_pred_before < 0).mean():.4f}")
    print(f"Fraction negative — {label_after} predictive: "
          f"{(P_pred_after < 0).mean():.4f}")


def plot_missing_statistics(trace, data, title=''):
    """
    Statistics of planning observations predicted to be missing.
    """
    z_post      = trace.posterior['z'].values
    z_mean_post = z_post.mean(axis=(0, 1))

    if 'pi_miss' in trace.posterior:
        pi_post = float(trace.posterior['pi_miss'].values.ravel().mean())
    elif 'pi_miss_pos' in trace.posterior:
        pi_pos  = float(trace.posterior['pi_miss_pos'].values.ravel().mean())
        pi_neg  = float(trace.posterior['pi_miss_neg'].values.ravel().mean())
        pi_post = (pi_pos + pi_neg) / 2
    else:
        print(f"No missingness parameters found in trace for {title}")
        return

    P_obs   = data['P_obs']
    is_zero = np.abs(P_obs) < 1e-6

    n_zeros_per_area = is_zero.sum(axis=1)
    expected_missing = n_zeros_per_area * pi_post
    total_missing    = expected_missing.sum()

    z_at_zeros    = z_mean_post[is_zero]
    z_pos_missing = z_at_zeros[z_at_zeros > 0]
    z_neg_missing = z_at_zeros[z_at_zeros < 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(n_zeros_per_area, bins=range(0, data['n_years'] + 2),
            color='steelblue', alpha=0.7, density=True)
    ax.axvline(n_zeros_per_area.mean(), color='red', linestyle='--',
               linewidth=0.8, label=f'mean={n_zeros_per_area.mean():.1f}')
    ax.set_xlabel('Number of zero planning observations per LSOA')
    ax.set_ylabel('Density')
    ax.set_title('Zeros per LSOA')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(expected_missing, bins=50, color='coral', alpha=0.7, density=True)
    ax.axvline(expected_missing.mean(), color='red', linestyle='--',
               linewidth=0.8, label=f'mean={expected_missing.mean():.1f}')
    ax.set_xlabel('Expected missing observations per LSOA')
    ax.set_ylabel('Density')
    ax.set_title(f'Expected missing per LSOA (pi_miss={pi_post:.3f})\n'
                 f'Total expected missing: {total_missing:.0f} / '
                 f'{is_zero.sum()} zeros')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax   = axes[1, 0]
    clip = np.quantile(np.abs(z_at_zeros), 0.99)
    ax.hist(z_at_zeros, bins=100, density=True,
            color='steelblue', alpha=0.7, range=(-clip, clip))
    ax.axvline(0,               color='black', linewidth=0.8)
    ax.axvline(z_at_zeros.mean(), color='red', linestyle='--',
               linewidth=0.8, label=f'mean={z_at_zeros.mean():.2f}')
    ax.set_xlabel('Posterior mean z where P_obs = 0')
    ax.set_ylabel('Density')
    ax.set_title('True change inferred where planning shows zero')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax    = axes[1, 1]
    total = len(z_at_zeros)
    n_pos  = (z_at_zeros > 0).sum()
    n_neg  = (z_at_zeros < 0).sum()
    n_zero = (z_at_zeros == 0).sum()

    bars = ax.bar(
        ['z > 0\n(missing completion)',
         'z ≈ 0\n(genuine zero)',
         'z < 0\n(missing demolition)'],
        [n_pos / total, n_zero / total, n_neg / total],
        color=['steelblue', 'grey', 'coral'],
        alpha=0.7
    )
    ax.set_ylabel('Fraction of zero observations')
    ax.set_title('Inferred reason for zero planning observations')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, [n_pos / total, n_zero / total, n_neg / total]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)

    plt.suptitle(f'{title} — missing planning observation statistics')
    plt.tight_layout()
    plt.show()

    print(f"\n── Missing observation summary ──────────────────────────────")
    print(f"  Total zero planning observations:    {is_zero.sum():,}")
    print(f"  Expected missing (pi_miss={pi_post:.3f}): {total_missing:.0f}")
    print(f"  Of zeros, z>0 (missing completion):  "
          f"{n_pos:,} ({n_pos/total*100:.1f}%)")
    print(f"  Of zeros, z≈0 (genuine zero):        "
          f"{n_zero:,} ({n_zero/total*100:.1f}%)")
    print(f"  Of zeros, z<0 (missing demolition):  "
          f"{n_neg:,} ({n_neg/total*100:.1f}%)")
    print(f"\n  Mean z where planning=0:             {z_at_zeros.mean():.2f}")
    print(f"  Mean z where planning=0 and z>0:     "
          f"{z_pos_missing.mean():.2f}" if len(z_pos_missing) > 0
          else "  No positive z at zeros")
    print(f"  Mean z where planning=0 and z<0:     "
          f"{z_neg_missing.mean():.2f}" if len(z_neg_missing) > 0
          else "  No negative z at zeros")


# ── M5b — two-component observation noise ─────────────────────────────────────

def plot_twocomp_diagnostics(trace, data, title='M5b'):
    """
    M5b-specific diagnostics — two-component observation noise.
    """
    w_tight_post = trace.posterior['w_tight'].values.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax    = axes[0]
    ax.hist(w_tight_post, bins=50, density=True,
            color='steelblue', alpha=0.7, label='Posterior')
    x     = np.linspace(0, 1, 200)
    prior = stats.beta.pdf(x, 9, 1)
    ax.plot(x, prior, color='red', linestyle='--',
            linewidth=1.0, label='Prior Beta(9,1)')
    ax.axvline(w_tight_post.mean(), color='black', linestyle='--',
               linewidth=0.8, label=f'mean={w_tight_post.mean():.3f}')
    ax.set_xlabel('w_tight')
    ax.set_title('Weight on tight observation component')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax    = axes[1]
    x     = np.linspace(-60, 60, 500)
    tight = stats.t.pdf(x, df=4, loc=0, scale=0.5)
    loose = stats.t.pdf(x, df=4, loc=0, scale=20.0)
    mix   = w_tight_post.mean() * tight + (1 - w_tight_post.mean()) * loose
    ax.plot(x, tight, color='steelblue', linewidth=1.0,
            label=f'Tight (sigma=0.5, w={w_tight_post.mean():.2f})')
    ax.plot(x, loose, color='coral',     linewidth=1.0,
            label=f'Loose (sigma=20, w={1-w_tight_post.mean():.2f})')
    ax.plot(x, mix,   color='black',     linewidth=1.5,
            linestyle='--', label='Mixture')
    ax.set_xlabel('Residual')
    ax.set_title('Implied observation noise distribution')
    ax.set_xlim(-30, 30)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle(f'{title} — two-component observation noise')
    plt.tight_layout()
    plt.show()

    print(f"\nw_tight posterior: mean={w_tight_post.mean():.3f}  "
          f"std={w_tight_post.std():.3f}")
    print(f"Implied fraction misallocated: {1-w_tight_post.mean():.3f}")


# ── M6 — spatial misallocation ────────────────────────────────────────────────

def plot_spatial_diagnostics(stats_dict, title='M6'):
    """
    M6-specific diagnostics — receives pre-computed stats dict
    from diagnostics.compute_spatial_misallocation_stats.
    """
    alpha_post = stats_dict['alpha_post']
    z_flat     = stats_dict['z_flat']
    z_lag      = stats_dict['z_lag']

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax    = axes[0]
    ax.hist(alpha_post, bins=50, density=True,
            color='steelblue', alpha=0.7, label='Posterior')
    x     = np.linspace(0, 1, 200)
    prior = stats.beta.pdf(x, 1, 19)
    ax.plot(x, prior, color='red', linestyle='--',
            linewidth=1.0, label='Prior Beta(1,19)')
    ax.axvline(alpha_post.mean(), color='black', linestyle='--',
               linewidth=0.8,
               label=f'mean={alpha_post.mean():.3f}')
    ax.set_xlabel('alpha_spatial')
    ax.set_title('Misallocation probability')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    ax   = axes[1]
    ax.scatter(z_flat, z_lag, alpha=0.2, s=5, color='steelblue')
    lims = [min(z_flat.min(), z_lag.min()),
            max(z_flat.max(), z_lag.max())]
    ax.plot(lims, lims, color='black', linestyle='--',
            linewidth=0.8, label='1:1')
    ax.set_xlabel('Posterior mean z')
    ax.set_ylabel('Spatial lag of z (neighbour mean)')
    ax.set_title('z vs spatial lag of z')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)

    plt.suptitle(f'{title} — spatial misallocation diagnostics')
    plt.tight_layout()
    plt.show()

    print(f"\nalpha_spatial posterior:")
    print(f"  mean={stats_dict['alpha_mean']:.4f}  "
          f"std={stats_dict['alpha_std']:.4f}  "
          f"90% CI=[{stats_dict['alpha_lo']:.4f}, "
          f"{stats_dict['alpha_hi']:.4f}]")
