import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

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
                show_legend=False, alpha_ci=0.9, lsoa_codes=None,
                resp_noise_P=None, resp_noise_E=None):
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
    lsoa_codes  : full list of LSOA21CD strings (e.g. trace's 'area' coord),
                  optional — if given, the panel title shows the actual
                  LSOA code (e.g. "E01033711") instead of the bare
                  positional index.
    resp_noise_P, resp_noise_E : (n_areas, n_years) posterior mean
                  P(this cell is noise), optional — for models with the
                  noise-mixture likelihood (e.g. AZ3's resp_noise_P/E).
                  When given, P_obs/E_obs markers are colour-coded by this
                  value (green=signal, red=noise) instead of drawn in a
                  single flat colour, so which specific spikes the model
                  is discounting as noise is visible directly on the plot
                  rather than needing a separate lookup.
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
        if resp_noise_P is not None:
            ax.plot(infer_years, P_obs[idx], color=COLOURS['planning'],
                    alpha=0.3, linewidth=1.0, zorder=2)
            ax.scatter(infer_years, P_obs[idx], c=resp_noise_P[idx],
                      cmap='RdYlGn_r', vmin=0, vmax=1, marker='s', s=45,
                      edgecolor='black', linewidth=0.5, zorder=3,
                      label='Planning (colour = P(noise))')
        else:
            ax.plot(infer_years, P_obs[idx], color=COLOURS['planning'],
                    marker='s', alpha=0.7, linewidth=1.0, label='Planning')

    if E_obs is not None:
        if resp_noise_E is not None:
            ax.plot(infer_years, E_obs[idx], color=COLOURS['ben'],
                    alpha=0.3, linewidth=1.0, zorder=2)
            ax.scatter(infer_years, E_obs[idx], c=resp_noise_E[idx],
                      cmap='RdYlGn_r', vmin=0, vmax=1, marker='^', s=45,
                      edgecolor='black', linewidth=0.5, zorder=3,
                      label='BEN (colour = P(noise))')
        else:
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
    label = lsoa_codes[idx] if lsoa_codes is not None else f'LSOA {idx}'
    ax.set_title(f'{label}  (D={D[idx]:.0f}  '
                 f'post. sum={z_sum_mean:.0f} '
                 f'[{z_sum_lo:.0f}, {z_sum_hi:.0f}])'
                 if D is not None else label)
    ax.set_xlabel('Year')
    ax.set_ylabel('Net dwelling change')
    ax.spines[['top', 'right']].set_visible(False)

    if show_legend:
        ax.legend(fontsize=7)

    return ax


_SCENARIO_COLOURS = ['#377eb8', '#4daf4a', '#984ea3']


def plot_z_area_modes(ax, z_post, idx, infer_years=INFER_YEARS,
                      P_obs=None, E_obs=None, D=None, n_years=10,
                      show_legend=False, lsoa_codes=None,
                      resp_noise_P=None, resp_noise_E=None,
                      n_clusters=2, alpha_ci=0.9, min_cluster_frac=0.03,
                      seed=0):
    """
    Mode-decomposed alternative to plot_z_area, for areas where z's
    posterior is genuinely multimodal (see diagnostics.detect_z_multimodality).

    A single mean+CI band, as plot_z_area draws, silently assumes z's
    marginal posterior is roughly unimodal. Found this assumption
    genuinely fails for a large fraction of cells in models with a
    noise-mixture likelihood (see docs/az-family-work-plan.md Phase 3):
    scanning AZ3, 34.6% of all LSOA-years and 89% of areas have at least
    one multimodal year, e.g. E01002702's 2013 cell is ~50% mass at ~0
    and ~50% spread across 30-150 -- its posterior MEAN (34.5) sits in
    the low-density valley BETWEEN those, representing neither
    explanation well, and its 90% CI band consequently spans a range
    that's individually implausible under either explanation on its own.

    This clusters entire posterior DRAWS (the full per-draw n_years
    vector for this area, not each year independently) into n_clusters
    groups via k-means, then plots each cluster's own mean+CI band as a
    separate "scenario", labelled with its share of the posterior. Whole
    -draw clustering (rather than per-cell) is deliberate: it captures
    that a mode is a COHERENT year-by-year story (e.g. "2013 and 2014
    are both ~0, with the freed-up total concentrated in 2016/2019/2021"
    is one scenario), not an independent choice per year -- exactly
    matching how a user correctly diagnosed this case by proposing a
    specific whole-row alternative, not just "z[2013] should be lower".

    Each cluster's per-draw z vectors individually still sum to D exactly
    (the underlying zero-sum construction guarantees this on every draw),
    so each cluster's MEAN also sums to exactly D -- every scenario shown
    is itself a valid, exactly-reconciled candidate history, not an
    approximation.

    Clustering is done on PER-YEAR STANDARDIZED draws (so a high-variance
    year doesn't dominate which draws get grouped together), but plotted
    lines/bands use the original, unstandardized scale. Clusters smaller
    than min_cluster_frac of all draws are dropped as noise rather than
    plotted as a spurious third "scenario".

    IMPORTANT CAVEAT, found the hard way and now surfaced directly in the
    title: k-means will always produce SOME n_clusters-way split, even
    when the underlying uncertainty isn't actually clustered. Checked
    E01035709 (9/10 years flagged multimodal) as a second example after
    the first plot for it was presented too confidently: it has literally
    zero P/E signal in 9 of 10 years, and clustering still found an
    apparently clean 72%/28% split. But checking which single year has
    the highest z in each posterior draw showed that "highest year" is
    spread almost uniformly across all 10 years (~0.10 each -- pure
    exchangeability), not concentrated on 2 candidates -- the "2
    scenarios" were an artifact of forcing k=2 on diffuse, near-
    exchangeable uncertainty, not 2 real stories. Cluster silhouette
    score does NOT reliably catch this (E01035709's was a middling 0.17,
    and separately, areas with ZERO active P/E cells were found with
    silhouette as high as 0.56 -- corr(silhouette, n_active_cells) across
    all 200 areas was 0.085, essentially nothing). What DOES catch it:
    the title's "top-n_clusters-year concentration" figure -- the
    fraction of draws whose single highest year falls among the
    n_clusters most-common highest-years. High (>~50%) means the
    decomposition reflects real concentrated structure; low means treat
    the scenario split as decoration on genuinely diffuse uncertainty,
    closer in spirit to what plot_z_area's single wide band already
    showed.

    Parameters mirror plot_z_area where they overlap; new ones:
    n_clusters       : max number of scenarios to look for (default 2 --
                       matches the dominant case found in the AZ3 scan;
                       raise for areas flagged with 3+ modes)
    min_cluster_frac : drop clusters smaller than this fraction of draws
    seed             : k-means initialisation seed, for reproducible plots

    Returns ax.
    """
    lo_pct = 100 * (1 - alpha_ci) / 2
    hi_pct = 100 - lo_pct

    z_area = z_post[:, :, idx, :]  # (chain, draw, year)
    flat = z_area.reshape(-1, z_area.shape[-1])  # (n_samples, n_years)

    std = flat.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    flat_norm = (flat - flat.mean(axis=0)) / std_safe

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(flat_norm, n_clusters, minit='++', seed=rng)

    weights = np.array([(labels == k).mean() for k in range(n_clusters)])
    order = np.argsort(-weights)

    for rank, k in enumerate(order):
        mask = labels == k
        if mask.mean() < min_cluster_frac:
            continue
        cluster_draws = flat[mask]  # raw (unstandardized) scale
        c_mean = cluster_draws.mean(axis=0)
        c_lo = np.percentile(cluster_draws, lo_pct, axis=0)
        c_hi = np.percentile(cluster_draws, hi_pct, axis=0)
        color = _SCENARIO_COLOURS[rank % len(_SCENARIO_COLOURS)]
        ax.plot(infer_years, c_mean, color=color, marker='o', linewidth=1.8,
                label=f'Scenario {rank + 1} ({weights[k]:.0%} of draws)')
        ax.fill_between(infer_years, c_lo, c_hi, alpha=0.15, color=color)

    if P_obs is not None:
        if resp_noise_P is not None:
            ax.plot(infer_years, P_obs[idx], color=COLOURS['planning'],
                    alpha=0.3, linewidth=1.0, zorder=2)
            ax.scatter(infer_years, P_obs[idx], c=resp_noise_P[idx],
                      cmap='RdYlGn_r', vmin=0, vmax=1, marker='s', s=45,
                      edgecolor='black', linewidth=0.5, zorder=3,
                      label='Planning (colour = P(noise))')
        else:
            ax.plot(infer_years, P_obs[idx], color=COLOURS['planning'],
                    marker='s', alpha=0.7, linewidth=1.0, label='Planning')

    if E_obs is not None:
        if resp_noise_E is not None:
            ax.plot(infer_years, E_obs[idx], color=COLOURS['ben'],
                    alpha=0.3, linewidth=1.0, zorder=2)
            ax.scatter(infer_years, E_obs[idx], c=resp_noise_E[idx],
                      cmap='RdYlGn_r', vmin=0, vmax=1, marker='^', s=45,
                      edgecolor='black', linewidth=0.5, zorder=3,
                      label='BEN (colour = P(noise))')
        else:
            ax.plot(infer_years, E_obs[idx], color=COLOURS['ben'],
                    marker='^', alpha=0.7, linewidth=1.0, label='BEN')

    if D is not None:
        ax.axhline(D[idx] / n_years, color=COLOURS['baseline'],
                   linewidth=0.8, linestyle='--', alpha=0.5,
                   label=f'D/n ({D[idx]/n_years:.1f})')

    # "Confidence" this decomposition is real structure, not a k-means
    # artifact of forcing n_clusters on diffuse, near-exchangeable
    # uncertainty -- found the hard way (see docs/az-family-work-plan.md
    # Phase 3): E01035709 clustered into an apparently clean 72%/28% split
    # (silhouette 0.17, looks plausible) that turned out to be meaningless
    # -- checking which year has the single highest z in each draw showed
    # it's spread almost uniformly across ALL 10 years (~0.10 each, pure
    # exchangeability), not concentrated on 2 candidates. The concentration
    # check below is what actually caught that; silhouette score did not.
    # A low value here means: don't trust the scenario split as 2 genuine
    # stories -- the underlying uncertainty is closer to diffuse/many-way.
    argmax_year = flat.argmax(axis=1)
    year_mass = np.bincount(argmax_year, minlength=flat.shape[1]) / len(flat)
    concentration = np.sort(year_mass)[::-1][:n_clusters].sum()
    confidence_note = (f'top-{n_clusters}-year concentration={concentration:.0%}'
                       + ('' if concentration >= 0.5 else ' -- LOW: scenarios may not be real'))

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    label = lsoa_codes[idx] if lsoa_codes is not None else f'LSOA {idx}'
    title = f'{label} -- {n_clusters}-scenario decomposition ({confidence_note})'
    if D is not None:
        title = f'{label}  (D={D[idx]:.0f})  -- {n_clusters}-scenario ({confidence_note})'
    ax.set_title(title, fontsize=9)
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
    z_post_da  = trace.posterior['z']
    lsoa_codes = (z_post_da.coords['area'].values.tolist()
                 if 'area' in z_post_da.coords else None)
    z_post = z_post_da.values
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
                    alpha_ci=alpha_ci, lsoa_codes=lsoa_codes)

    for ax in axes.ravel()[n_sample:]:
        ax.set_visible(False)

    plt.suptitle(f'{title} — posterior z vs planning and BEN')
    plt.tight_layout()

    return fig, axes


# LSOAs called out by name in chat/docs/az-family-work-plan.md during the
# AZ-family investigation -- kept here so every subsequent diagnostic plot
# includes them as a stable, directly-comparable reference set across
# models, rather than each plot's auto-selection landing on a different
# set of areas each time. Add an entry whenever a new area gets singled
# out by name in a report; there's no automatic sync with the doc.
REFERENCE_AREAS = {
    'E01033491': 'under-tracked: huge P spike (762), tiny D (126)',
    'E01001774': 'under-tracked: extreme mismatch (P_sum=460 vs D=18)',
    'E01033711': "AZ0a/AZ1a's worst-missed spike; AZ1b/AZ2's biggest win (D=634)",
    'E01002703': ('single-source P spike only, E does NOT agree '
                  '(2013: P=274, E=8; D=501) -- see work plan doc'),
    'E01002794': 'P/E disagree on spike year (2020 vs 2016)',
    'E01033700': 'high Pareto-k, mostly-quiet years (D=556)',
    'E01035656': 'under-tracked spike (498), D=412',
    'E01002702': "AZ1b mode-summary example: chains split on E's lag category",
}


def select_spike_tracking_areas(trace, data, n_examples=6, reference_areas=None):
    """
    Select areas illustrating specific spike-tracking failure/success
    modes, rather than areas spanning the D range (see
    select_sample_areas for that). Built from the AZ0a failure-mode
    investigation, where aggregate diagnostics (r-hat, elpd) looked fine
    while individual traces revealed the model silently smoothing away
    large P/E observations -- this selector surfaces exactly the cases
    that catch that.

    Categories, filled in priority order (a category contributes 1-2
    areas if it has qualifying candidates; remaining slots are filled
    with the next-worst under-tracked spikes):
      1. Worst under-tracked spikes -- large |P_obs|/|E_obs| where z's
         posterior mean at the spike year is far below the raw value.
      2. Worst LOO Pareto-k cell's area, if the trace has a log_likelihood
         group for 'P_like' (skipped otherwise -- not every model/trace
         has this available, or it may be unreliable, see AZ0b).
      3. Largest P/E spike-year disagreement -- both sources show a real
         spike, in different years (the case a lag/reallocation mechanism
         should help with).
      4. A well-tracked large spike, for contrast -- confirms the model
         *can* track spikes when magnitude and timing are unambiguous.

    After the above, every area in `reference_areas` (default
    REFERENCE_AREAS -- LSOAs called out by name in chat/docs during this
    investigation) is added as an EXTRA panel if not already selected and
    present in this trace, so plots stay directly comparable across
    models/reports rather than each one auto-selecting a different set.
    These do not count against n_examples -- they're added on top.

    Returns a list of (area_idx, reason_str) tuples -- length <= n_examples
    from the auto-selected categories, plus however many reference areas
    matched and weren't already included.
    """
    z_post_da = trace.posterior['z']  # (chain, draw, area, year)
    lsoa_codes = (z_post_da.coords['area'].values.tolist()
                 if 'area' in z_post_da.coords else None)
    z_post  = z_post_da.values
    n_areas = z_post.shape[2]
    z_mean  = z_post.mean(axis=(0, 1))     # (area, year)

    P_obs, E_obs = data['P_obs'], data['E_obs']
    area_range = np.arange(n_areas)

    t_p = np.argmax(np.abs(P_obs), axis=1)
    t_e = np.argmax(np.abs(E_obs), axis=1)
    p_spike = P_obs[area_range, t_p]
    e_spike = E_obs[area_range, t_e]
    z_at_p_spike = z_mean[area_range, t_p]

    with np.errstate(divide='ignore', invalid='ignore'):
        shrink_ratio = np.where(np.abs(p_spike) > 1e-9,
                                z_at_p_spike / p_spike, np.nan)

    selected, used = [], set()

    def add(idx, reason):
        idx = int(idx)
        if idx not in used and len(selected) < n_examples:
            selected.append((idx, reason))
            used.add(idx)

    big_spike = np.abs(p_spike) > 30

    under_tracked = np.where(big_spike & (shrink_ratio < 0.3))[0]
    under_tracked = under_tracked[np.argsort(-np.abs(p_spike[under_tracked]))]
    for idx in under_tracked[:2]:
        add(idx, f"under-tracked: P={p_spike[idx]:.0f} in year-idx {t_p[idx]}, "
                 f"z reaches only {z_at_p_spike[idx]:.0f}")

    try:
        loo_P = az.loo(trace, var_name='P_like', pointwise=True)
        k = loo_P.pareto_k.values.reshape(n_areas, -1)
        worst_area = int(np.argmax(k.max(axis=1)))
        add(worst_area, f"worst LOO Pareto-k = {k.max():.2f} (PSIS-LOO unreliable here)")
    except Exception:  # noqa: BLE001 — Pareto-k is a bonus category, not
        # every trace/model has a usable P_like log_likelihood (e.g. a
        # Potential-based model sampled before the pointwise-attachment
        # fix, or no log_likelihood group at all) -- degrade gracefully
        # rather than block the rest of the plot on it.
        pass

    disagree = np.where((t_p != t_e) &
                        (np.abs(p_spike) > 10) & (np.abs(e_spike) > 10))[0]
    disagree = disagree[np.argsort(
        -(np.abs(p_spike[disagree]) + np.abs(e_spike[disagree])))]
    for idx in disagree[:1]:
        add(idx, f"P/E disagree on spike year: P peaks at idx {t_p[idx]}, "
                 f"E peaks at idx {t_e[idx]}")

    well_tracked = np.where(big_spike & (shrink_ratio > 0.85) &
                            (shrink_ratio < 1.15))[0]
    well_tracked = well_tracked[np.argsort(-np.abs(p_spike[well_tracked]))]
    for idx in well_tracked[:1]:
        add(idx, f"well-tracked (for contrast): P={p_spike[idx]:.0f}, "
                 f"z={z_at_p_spike[idx]:.0f}")

    if len(selected) < n_examples:
        remaining = np.where(big_spike)[0]
        remaining = remaining[np.argsort(shrink_ratio[remaining])]
        for idx in remaining:
            add(idx, f"large spike, shrink ratio={shrink_ratio[idx]:.2f}")
            if len(selected) >= n_examples:
                break

    # Reference areas are ADDED on top, not counted against n_examples --
    # bypass add()'s n_examples cap directly.
    ref = reference_areas if reference_areas is not None else REFERENCE_AREAS
    if lsoa_codes is not None:
        code_to_idx = {code: i for i, code in enumerate(lsoa_codes)}
        for code, note in ref.items():
            idx = code_to_idx.get(code)
            if idx is not None and idx not in used:
                selected.append((idx, f"reference area: {note}"))
                used.add(idx)

    return selected


def plot_spike_tracking_examples(trace, data, n_examples=6, title='',
                                 infer_years=INFER_YEARS, alpha_ci=0.9,
                                 reference_areas=None):
    """
    Plot posterior z vs P_obs/E_obs for areas specifically illustrating
    spike-tracking failure/success modes (see select_spike_tracking_areas)
    -- NOT a representative D-quantile sample (see plot_sample_areas for
    that). Intended as a standing per-model check, run alongside
    diagnostics_summary()/az.compare(), since aggregate scalar diagnostics
    can look fine on a model that silently smooths away large P/E
    observations -- exactly what happened with AZ0a, only found by
    inspecting individual traces against raw data.

    Panel titles show the actual LSOA21CD code (e.g. "E01033711"), not a
    bare positional index. Every area in REFERENCE_AREAS (LSOAs called out
    by name in chat/docs during this investigation) is included as an
    extra panel on top of the n_examples auto-selected ones, so plots
    across different models stay directly comparable -- pass
    reference_areas={} to disable, or your own dict to override.

    For models with the noise-mixture likelihood (resp_noise_P/E in the
    posterior, e.g. AZ3 -- see _build_noise_mixture_likelihood), P_obs/
    E_obs markers are automatically colour-coded green (signal) to red
    (noise) by posterior mean P(this cell is noise), with a shared
    colourbar, so which specific spikes the model is discounting is
    visible directly on the plot rather than needing a separate lookup.
    Models without this mechanism are unaffected (plain marker colours).
    """
    z_post_da = trace.posterior['z']
    lsoa_codes = (z_post_da.coords['area'].values.tolist()
                 if 'area' in z_post_da.coords else None)
    z_post = z_post_da.values
    D, P_obs, E_obs = data['D'], data['P_obs'], data['E_obs']

    has_noise_mixture = ('resp_noise_P' in trace.posterior
                         and 'resp_noise_E' in trace.posterior)
    resp_noise_P = (trace.posterior['resp_noise_P'].mean(dim=('chain', 'draw')).values
                    if has_noise_mixture else None)
    resp_noise_E = (trace.posterior['resp_noise_E'].mean(dim=('chain', 'draw')).values
                    if has_noise_mixture else None)

    selected = select_spike_tracking_areas(trace, data, n_examples=n_examples,
                                           reference_areas=reference_areas)
    if not selected:
        raise ValueError("No qualifying spike-tracking examples found for this trace/data.")

    nrows = int(np.ceil(len(selected) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, (ax, (idx, reason)) in enumerate(zip(axes_flat, selected)):
        plot_z_area(ax, z_post, idx, infer_years=infer_years,
                    P_obs=P_obs, E_obs=E_obs, D=D, n_years=data['n_years'],
                    show_legend=(i == 0), alpha_ci=alpha_ci, lsoa_codes=lsoa_codes,
                    resp_noise_P=resp_noise_P, resp_noise_E=resp_noise_E)
        ax.set_title(ax.get_title() + f"\n{reason}", fontsize=8)

    for ax in axes_flat[len(selected):]:
        ax.set_visible(False)

    suptitle = f'{title} — spike-tracking diagnostic examples' if title \
        else 'Spike-tracking diagnostic examples'
    plt.suptitle(suptitle)

    if has_noise_mixture:
        mappable = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
        mappable.set_array([])
        # Reserve room on the right for the colourbar as well as the top
        # margin for the suptitle.
        plt.tight_layout(rect=[0, 0, 0.94, 0.97])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Posterior P(cell is noise)', fontsize=9)
    else:
        # Reserve top margin for the suptitle explicitly -- with
        # reference_areas now able to push the grid past 2 rows,
        # tight_layout()'s automatic spacing isn't reliable enough on its
        # own and the suptitle overlaps the top row's (two-line,
        # reason-appended) titles.
        plt.tight_layout(rect=[0, 0, 1, 0.97])

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
