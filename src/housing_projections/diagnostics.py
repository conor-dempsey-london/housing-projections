import arviz as az
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks

from housing_projections.config import (
    ALL_COLS_BEN,
    ALL_COLS_PLAN,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
    INFER_YEARS,
)
from housing_projections.spatial import build_weights_libpysal, compute_morans_i


def _check_rhat(trace, var_names=None, threshold=1.01, verbose=False):
    """
    Check r-hat convergence for all or specified variables.

    Returns
    -------
    dict with keys 'summary', 'problematic' (variables with r-hat > threshold)
    """
    summary = az.summary(trace, var_names=var_names)
    rhat_col    = pd.to_numeric(summary['r_hat'], errors='coerce').dropna()
    problematic = summary.loc[rhat_col[rhat_col > threshold].index]

    if verbose:
        print(f"\n── R-hat diagnostics (threshold={threshold}) ──────────────")
        print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])
        if len(problematic) > 0:
            print(f"\n*** WARNING: {len(problematic)} variables above threshold ***")
            print(problematic[['r_hat']])
        else:
            print(f"\nAll variables below r-hat threshold of {threshold}")

    return {'summary': summary, 'problematic': problematic}


def _check_divergences(trace, verbose=False):
    """
    Check number of divergences in trace.

    Returns
    -------
    int — number of divergences
    """
    n_divergences = int(trace.sample_stats.diverging.sum())

    if verbose:
        print("\n── Divergences ──────────────────────────────────────")
        if n_divergences > 0:
            print(f"*** WARNING: {n_divergences} divergences detected ***")
        else:
            print("No divergences detected")

    return n_divergences


def _check_calibration(trace, data, alpha=0.1, nu=4.0, verbose=False):
    """
    Check posterior predictive calibration coverage against planning and BEN
    observations.

    For each posterior draw of (z, sigma), samples from the StudentT likelihood
    to form the posterior predictive, then checks whether observations fall
    within the (1-alpha) credible interval.

    Falls back to z-only coverage if sigma variables are absent from the trace
    (e.g. models with fixed observation noise not stored in posterior).

    Returns
    -------
    dict with keys 'planning', 'ben'
    """
    z_post = trace.posterior['z'].values          # (chains, draws, n_areas, n_years)
    n_samples = z_post.shape[0] * z_post.shape[1]
    z_flat = z_post.reshape(n_samples, z_post.shape[2], z_post.shape[3])

    P_obs = data['P_obs']
    E_obs = data['E_obs']

    def _predictive_coverage(obs, sigma_key):
        if sigma_key in trace.posterior:
            sigma = trace.posterior[sigma_key].values.ravel()  # (n_samples,)
            rng   = np.random.default_rng(0)
            t_eps = stats.t.rvs(df=nu, size=z_flat.shape, random_state=rng)
            pred  = z_flat + sigma[:, None, None] * t_eps    # (n_samples, areas, years)
        else:
            pred = z_flat

        lo = np.percentile(pred, 100 * alpha / 2,       axis=0)
        hi = np.percentile(pred, 100 * (1 - alpha / 2), axis=0)
        return float(np.mean((obs >= lo) & (obs <= hi)))

    coverage = {
        'planning': _predictive_coverage(P_obs, 'sigma_plan'),
        'ben':      _predictive_coverage(E_obs, 'sigma_ben'),
    }

    if verbose:
        print(f"\n── Posterior predictive calibration ({int((1-alpha)*100)}% CI) ──")
        print(f"  Planning coverage: {coverage['planning']:.3f}  "
              f"(nominal {1-alpha:.2f})")
        print(f"  BEN coverage:      {coverage['ben']:.3f}  "
              f"(nominal {1-alpha:.2f})")

    return coverage


def _check_census_constraint(trace, data, verbose=False):
    """
    Check how well the census constraint is satisfied in the posterior.

    Returns
    -------
    dict with keys 'mean_violation', 'max_violation'
    """
    z_post    = trace.posterior['z'].values
    z_sums    = z_post.sum(axis=-1).reshape(-1, data['n_areas'])
    residuals = np.abs(z_sums - data['D'][None, :]).ravel()

    result = {
        'mean_violation': float(residuals.mean()),
        'max_violation':  float(residuals.max()),
    }

    if verbose:
        print("\n── Census constraint ────────────────────────────────")
        print(f"  Mean violation: {result['mean_violation']:.3f}")
        print(f"  Max violation:  {result['max_violation']:.3f}")

    return result


def _check_morans_i(trace, data, verbose=False):
    """
    Compute Moran's I on mean posterior residuals for planning and BEN.

    Returns
    -------
    dict with keys 'planning' and 'ben', each containing
    {'I': float, 'p_value': float, 'z_score': float}
    """
    z_post      = trace.posterior['z'].values
    z_mean_post = z_post.mean(axis=(0, 1))

    resid_plan  = (data['P_obs'] - z_mean_post).mean(axis=1)
    resid_ben   = (data['E_obs'] - z_mean_post).mean(axis=1)

    w = build_weights_libpysal(data['gdf'])

    result = {
        'planning': compute_morans_i(resid_plan, w),
        'ben':      compute_morans_i(resid_ben,  w),
    }

    if verbose:
        print("\n── Moran's I on residuals ───────────────────────────")
        for source, vals in result.items():
            print(f"  {source:10s}: I={vals['I']:.4f}  p={vals['p_value']:.4f}")

    return result


def _shift_source_series(obs_full, all_cols, infer_cols, mean_lag):
    """
    Shift a source's full observation series backward in calendar time by
    `mean_lag` years (rounded to the nearest integer) so that an event
    recorded with a delay is re-indexed to the year it's inferred to have
    actually occurred — e.g. if P_obs[t] on average reflects z from
    `mean_lag` years earlier, then P_aligned[s] = P_obs[s + mean_lag]
    pulls the observation recorded `mean_lag` years after year `s` back
    onto `s`. Out-of-range indices clip to the nearest available column
    (edge padding), consistent with `_build_pre_inference`'s head-padding.

    Returns array of shape (n_areas, len(infer_cols)).
    """
    infer_start = all_cols.index(infer_cols[0])
    n_years      = len(infer_cols)
    n_cols       = obs_full.shape[1]
    shift        = int(round(mean_lag))
    idx          = np.clip(infer_start + np.arange(n_years) + shift, 0, n_cols - 1)
    return obs_full[:, idx]


def _check_sigma_slab_vs_disagreement(trace, data, verbose=False):
    """
    M9-specific diagnostic: does the per-area hierarchical sigma_slab
    actually track genuine, timing-corrected disagreement between P and E
    (as intended), rather than just each area's raw scale?

    Method: take the posterior-mean lag distributions (lambda_weights_P/E),
    reduce each to its mean lag, and shift the full P/E observation series
    by that many years (`_shift_source_series`) to express both sources on
    a common "z-time" basis. De-mean each area's shifted P and E series and
    correlate their year-to-year deviations — this is the same statistic
    used pre-correction to diagnose the flat-z collapse (mean ~+0.01 on raw
    data). Correlate that per-area post-correction agreement against
    posterior-mean sigma_slab[a], both raw and after regressing out
    |D[a]|/n_years (area scale) — a sigma_slab[a] that just tracks area
    size rather than genuine disagreement would look superficially
    reasonable (real cross-area heterogeneity!) without doing what was
    asked. Expect a clear *negative* correlation: better-agreeing areas
    (after lag correction) should have tighter sigma_slab.

    Only meaningful for traces with 'lambda_weights_P', 'lambda_weights_E'
    and a per-area 'sigma_slab' (e.g. M9) — returns None otherwise.

    Returns
    -------
    dict or None, with keys 'mean_lag_P', 'mean_lag_E', 'n_areas_valid',
    'corr_agreement_vs_sigma_slab', 'corr_agreement_vs_sigma_slab_scale_controlled'
    """
    posterior = trace.posterior
    if 'lambda_weights_P' not in posterior or 'lambda_weights_E' not in posterior:
        return None
    sigma_slab_da = posterior['sigma_slab']
    if sigma_slab_da.ndim != 3:  # (chain, draw, area) — not per-area
        return None

    lambda_P = posterior['lambda_weights_P'].mean(dim=['chain', 'draw']).values
    lambda_E = posterior['lambda_weights_E'].mean(dim=['chain', 'draw']).values
    mean_lag_P = float(np.sum(np.arange(len(lambda_P)) * lambda_P))
    mean_lag_E = float(np.sum(np.arange(len(lambda_E)) * lambda_E))

    P_aligned = _shift_source_series(
        data['P_obs_full'], ALL_COLS_PLAN, INFER_COLS_PLAN, mean_lag_P)
    E_aligned = _shift_source_series(
        data['E_obs_full'], ALL_COLS_BEN, INFER_COLS_BEN, mean_lag_E)

    P_dm = P_aligned - P_aligned.mean(axis=1, keepdims=True)
    E_dm = E_aligned - E_aligned.mean(axis=1, keepdims=True)

    n_areas = P_aligned.shape[0]
    agreement = np.full(n_areas, np.nan)
    for a in range(n_areas):
        if P_dm[a].std() > 1e-9 and E_dm[a].std() > 1e-9:
            agreement[a] = np.corrcoef(P_dm[a], E_dm[a])[0, 1]

    sigma_slab = sigma_slab_da.mean(dim=['chain', 'draw']).values

    valid = np.isfinite(agreement)
    n_valid = int(valid.sum())

    corr_raw = (float(np.corrcoef(agreement[valid], sigma_slab[valid])[0, 1])
                if n_valid > 2 else float('nan'))

    corr_scale_controlled = float('nan')
    if n_valid > 2:
        scale = np.abs(data['D']) / data['n_years']
        design = np.column_stack([scale[valid], np.ones(n_valid)])
        coef, *_ = np.linalg.lstsq(design, sigma_slab[valid], rcond=None)
        resid   = sigma_slab[valid] - design @ coef
        if resid.std() > 1e-9:
            corr_scale_controlled = float(np.corrcoef(agreement[valid], resid)[0, 1])

    result = {
        'mean_lag_P':                                    mean_lag_P,
        'mean_lag_E':                                    mean_lag_E,
        'n_areas_valid':                                 n_valid,
        'corr_agreement_vs_sigma_slab':                  corr_raw,
        'corr_agreement_vs_sigma_slab_scale_controlled':  corr_scale_controlled,
    }

    if verbose:
        print("\n── sigma_slab vs lag-corrected P/E disagreement ─────")
        print(f"  Mean lag  — P: {mean_lag_P:.2f} yr   E: {mean_lag_E:.2f} yr")
        print(f"  Areas with valid agreement stat: {n_valid}/{n_areas}")
        print(f"  corr(agreement, sigma_slab):                 {corr_raw:.3f}")
        print(f"  corr(agreement, sigma_slab | area scale):    {corr_scale_controlled:.3f}")
        print("  (expect clearly negative: well-explained areas -> tight sigma_slab)")

    return result


def _check_kappa_vs_recording_rate(trace, data, verbose=False):
    """
    M10-specific diagnostic: does per-area capture-rate kappa track the
    systematic per-area log(P/E) recording-rate bias found in notebook
    4.0 section 10 (persistent per-area effect, distinct from any year
    effect), rather than leaving it unexplained in sigma_plan/sigma_ben
    the way M9 did?

    Method: posterior-mean kappa_P[a]/kappa_E[a] -> log_kappa_ratio[a].
    Independently, empirical_log_ratio[a] = mean(log(P_obs/E_obs)) over
    cells where both are non-zero for that area (a quick cross-check, not
    a re-run of section 10's full ANOVA). Correlate across areas with
    enough valid cells.

    Only meaningful for traces with 'kappa_P'/'kappa_E' (e.g. M10) —
    returns None otherwise.

    Returns
    -------
    dict or None, with keys 'n_areas_valid', 'corr_kappa_ratio_vs_empirical',
    'log_kappa_ratio', 'empirical_log_ratio' (per-area arrays, for plotting).
    """
    posterior = trace.posterior
    if 'kappa_P' not in posterior or 'kappa_E' not in posterior:
        return None

    kappa_P = posterior['kappa_P'].mean(dim=['chain', 'draw']).values
    kappa_E = posterior['kappa_E'].mean(dim=['chain', 'draw']).values
    log_kappa_ratio = np.log(kappa_P) - np.log(kappa_E)

    P_obs, E_obs = data['P_obs'], data['E_obs']
    mask_both = (P_obs > 0) & (E_obs > 0)
    n_areas   = P_obs.shape[0]
    min_obs   = 2

    empirical_log_ratio = np.full(n_areas, np.nan)
    for a in range(n_areas):
        sel = mask_both[a]
        if sel.sum() >= min_obs:
            empirical_log_ratio[a] = np.log(P_obs[a, sel] / E_obs[a, sel]).mean()

    valid   = np.isfinite(empirical_log_ratio)
    n_valid = int(valid.sum())
    corr = (float(np.corrcoef(log_kappa_ratio[valid], empirical_log_ratio[valid])[0, 1])
            if n_valid > 2 else float('nan'))

    result = {
        'n_areas_valid':                 n_valid,
        'corr_kappa_ratio_vs_empirical': corr,
        'log_kappa_ratio':               log_kappa_ratio,
        'empirical_log_ratio':           empirical_log_ratio,
    }

    if verbose:
        print("\n── kappa_P/kappa_E vs empirical log(P/E) recording rate ─────")
        print(f"  Areas with valid empirical ratio: {n_valid}/{n_areas}")
        print(f"  corr(log(kappa_P/kappa_E), empirical log(P/E)): {corr:.3f}")
        print("  (expect strongly positive)")

    return result


def _check_residuals(trace, data, verbose=False):
    """
    Compute residual statistics for planning and BEN.

    Returns
    -------
    dict with keys 'planning' and 'ben', each containing
    {'mean': float, 'std': float, 'mae': float}
    """
    z_post      = trace.posterior['z'].values
    z_mean_post = z_post.mean(axis=(0, 1))

    result = {}
    for obs, key in [(data['P_obs'], 'planning'), (data['E_obs'], 'ben')]:
        resid = obs - z_mean_post
        result[key] = {
            'mean': float(resid.mean()),
            'std':  float(resid.std()),
            'mae':  float(np.abs(resid).mean()),
        }

    if verbose:
        print("\n── Residuals ────────────────────────────────────────")
        for source, vals in result.items():
            print(f"  {source:10s}: mean={vals['mean']:6.2f}  "
                  f"std={vals['std']:6.2f}  mae={vals['mae']:6.2f}")

    return result


def check_chain_agreement(trace, scalar_vars=None, logp_gap_sigma=8.0, verbose=True):
    """
    Check whether all chains agree on scalar parameters and log-probability.

    R-hat flags *that* something is wrong but not *which* chain or *why*. A
    chain trapped in a distinct, lower-probability mode (e.g. a non-centred
    funnel neck) shows up here as a large gap between its mean `logp` and the
    other chains', well before you'd think to look at individual scalar
    posteriors.

    A chain is flagged only if its mean logp gap from the best chain is large
    relative to *within-chain* logp noise (pooled std across chains) — using
    the raw best-to-worst spread is too sensitive, since ordinary sampling
    noise always puts some chains below the mean.

    Parameters
    ----------
    trace           : az.InferenceData (must have sample_stats.logp)
    scalar_vars     : list of scalar variable names to report per-chain
                      means/stds for (e.g. ['sigma_slab', 'lambda_weights'])
    logp_gap_sigma  : a chain is flagged if its mean logp is more than this
                      many pooled within-chain logp standard deviations below
                      the best chain's mean logp (default 8 — a trapped-mode
                      chain typically shows gaps of 50-100+ sigma; ordinary
                      inter-chain noise is well under 1 sigma)
    verbose         : print a warning for flagged chains

    Returns
    -------
    dict with keys:
        'logp_by_chain'   : list of per-chain mean logp
        'flagged_chains'  : list of chain indices that appear trapped in a
                             distinct mode
        'scalar_by_chain' : {var: [{'mean': ..., 'std': ...}, ...]} per chain
    """
    # nutpie names this stat 'logp'; PyMC's own sampler (used whenever a
    # model has a discrete RV needing a compound Gibbs+NUTS step, e.g.
    # M14's profile_k, since nutpie can't compile discrete variables)
    # names the same quantity 'lp' instead.
    logp_name = 'logp' if 'logp' in trace.sample_stats else (
        'lp' if 'lp' in trace.sample_stats else None)
    if logp_name is None:
        # Some traces (e.g. small synthetic ones built without nutpie/PyMC's
        # full sample_stats) don't carry per-draw logp — nothing to check.
        if verbose:
            print("\n── Chain agreement ──────────────────────────────────")
            print("  No 'logp'/'lp' in sample_stats — skipping.")
        return {'logp_by_chain': [], 'flagged_chains': [], 'scalar_by_chain': {}}

    logp = trace.sample_stats[logp_name].values  # (chain, draw)
    logp_by_chain  = logp.mean(axis=1)
    within_std     = logp.std(axis=1).mean()  # pooled within-chain noise scale

    best = logp_by_chain.max()
    if within_std > 0:
        flagged = [
            int(c) for c, lp in enumerate(logp_by_chain)
            if (best - lp) / within_std > logp_gap_sigma
        ]
    else:
        flagged = []

    scalar_by_chain = {}
    for var in (scalar_vars or []):
        if var not in trace.posterior:
            continue
        vals = trace.posterior[var].values  # (chain, draw, ...)
        n_chains = vals.shape[0]
        flat = vals.reshape(n_chains, vals.shape[1], -1)
        scalar_by_chain[var] = [
            {'mean': flat[c].mean(axis=0).tolist(), 'std': flat[c].std(axis=0).tolist()}
            for c in range(n_chains)
        ]

    if verbose:
        print("\n── Chain agreement ──────────────────────────────────")
        print(f"  Mean logp by chain: {np.round(logp_by_chain, 1).tolist()}")
        if flagged:
            print(f"  *** WARNING: chain(s) {flagged} appear trapped in a "
                  f"distinct, lower-probability mode ***")
        else:
            print("  All chains agree on logp.")
        for var, chains in scalar_by_chain.items():
            means = [c['mean'] for c in chains]
            print(f"  {var} mean by chain: {means}")

    return {
        'logp_by_chain':   logp_by_chain.tolist(),
        'flagged_chains':  flagged,
        'scalar_by_chain': scalar_by_chain,
    }


def z_identifiability_summary(trace, rhat_threshold=1.01):
    """
    Per-area summary of how confidently the model identifies *which year*
    each area's change happened in.

    Some models (e.g. M2h) pin an area's total change tightly via the census
    constraint, but for areas with weak per-year signal (little informative
    planning/BEN data relative to the size of the change) there can be
    several roughly-equally-good ways to distribute that total across years
    — the sampler settles on different allocations across chains/draws even
    though the total is well identified. This shows up as elevated r-hat on
    individual (area, year) `z` cells without indicating a sampling bug.

    This is a reporting tool, not a fix: it flags which areas' *year-by-year*
    breakdown should be treated with caution, while the area's total change
    (and its uncertainty) remains reliable.

    Parameters
    ----------
    trace          : az.InferenceData (posterior must contain 'z')
    rhat_threshold : per-cell r-hat above this counts as low-confidence

    Returns
    -------
    pd.DataFrame with one row per area:
        area, n_low_confidence_years, max_rhat, confident (bool)
    """
    rhat_z = az.rhat(trace, var_names=['z'])['z']  # (area, year)
    rhat_vals = rhat_z.values

    area_codes = (rhat_z.coords['area'].values if 'area' in rhat_z.coords
                  else np.arange(rhat_vals.shape[0]))

    finite = np.where(np.isfinite(rhat_vals), rhat_vals, np.nan)
    n_low_confidence_years = (rhat_vals > rhat_threshold).sum(axis=1)

    return pd.DataFrame({
        'area':                    area_codes,
        'n_low_confidence_years':  n_low_confidence_years,
        'max_rhat':                np.nanmax(finite, axis=1),
        'confident':               n_low_confidence_years == 0,
    })


def hierarchical_mode_summary(trace, var_name, rhat_threshold=1.01,
                              purity_threshold=0.95):
    """
    Per-group characterization of genuine posterior multimodality in a
    hierarchically-pooled simplex/category variable (e.g. AZ1b's
    'lag_P_lambda_weights', shape (chain, draw, group, category)).

    Built specifically because AZ1b's elevated r-hat on per-area lag
    weights turned out NOT to be a sampling-efficiency problem: checked
    directly (not assumed) that individual chains spend effectively all
    their draws in one of several disconnected modes and never cross over
    -- with ~10 obs/area/source, some areas' data genuinely can't
    distinguish between two candidate lag years that each explain a spike
    about equally well, so the likelihood itself has separated peaks for
    those groups. Forcing r-hat -> 1 there would mean suppressing real
    epistemic ambiguity, not fixing a bug. This reports that ambiguity
    instead of hiding it: for each group with elevated r-hat, which
    category each chain's draws concentrate on, and (given enough chains
    to trust it -- see the module docstring reasoning in AZ1b's own
    docstring re: chain count) an estimated relative posterior mass per
    mode from the fraction of chains landing in each.

    A single chain's initial trajectory into one basin vs another is close
    to a coin flip, so the reported mode split is only as trustworthy as
    the number of chains it's estimated from -- more chains (e.g. 8 rather
    than the usual 4) meaningfully tighten this estimate; report the
    n_chains used alongside any split you quote.

    Parameters
    ----------
    trace            : az.InferenceData (posterior must contain var_name,
                       with dims (chain, draw, <group_dim>, <category_dim>))
    var_name         : name of the per-group simplex/category variable
    rhat_threshold   : per-group max r-hat above this triggers mode analysis
    purity_threshold : a chain counts as "pure" (cleanly in one mode) if
                       at least this fraction of its draws share the same
                       dominant category; chains below this are flagged as
                       'mixed' rather than assigned to a mode -- a signal
                       that the group's ambiguity may be more of a smooth
                       spread than genuinely disconnected modes.

    Returns
    -------
    pd.DataFrame with one row per group that has elevated r-hat:
        group, max_rhat, n_chains, n_modes, mode_categories (list),
        mode_chain_counts (list), mode_mass_estimate (list, fractions
        summing to <=1 -- less than 1 if any chains were 'mixed'),
        n_mixed_chains, stable (bool -- True iff every chain is pure)
    """
    da = trace.posterior[var_name]  # (chain, draw, group, category)
    group_dim = da.dims[2]
    group_labels = (da.coords[group_dim].values if group_dim in da.coords
                    else np.arange(da.sizes[group_dim]))

    rhat = az.rhat(trace, var_names=[var_name])[var_name].values  # (group, category)
    max_rhat_per_group = np.nanmax(rhat, axis=1)
    flagged = np.where(max_rhat_per_group > rhat_threshold)[0]

    values = da.values  # (chain, draw, group, category)
    n_chains = values.shape[0]
    dominant_category = values.argmax(axis=-1)  # (chain, draw, group)

    rows = []
    for g in flagged:
        chain_counts, mixed = {}, 0
        for c in range(n_chains):
            cats, counts = np.unique(dominant_category[c, :, g], return_counts=True)
            top_idx = counts.argmax()
            purity = counts[top_idx] / counts.sum()
            if purity < purity_threshold:
                mixed += 1
                continue
            cat = int(cats[top_idx])
            chain_counts[cat] = chain_counts.get(cat, 0) + 1

        mode_categories  = sorted(chain_counts)
        mode_chain_counts = [chain_counts[k] for k in mode_categories]
        mode_mass = [n / n_chains for n in mode_chain_counts]

        rows.append({
            'group':              group_labels[g],
            'max_rhat':           max_rhat_per_group[g],
            'n_chains':           n_chains,
            'n_modes':            len(mode_categories),
            'mode_categories':    mode_categories,
            'mode_chain_counts':  mode_chain_counts,
            'mode_mass_estimate': [round(m, 3) for m in mode_mass],
            'n_mixed_chains':     mixed,
            'stable':             mixed == 0,
        })

    return pd.DataFrame(rows)


def _detect_modes(samples, grid_points=256, prominence_frac=0.05):
    """
    KDE-based mode count for a 1-D array of posterior draws.

    Returns (mode_locations, mode_heights, n_modes). A cell with a single,
    tight point mass (e.g. sd effectively 0) is reported as exactly one
    mode at that value, rather than raising on a degenerate KDE bandwidth.

    prominence_frac: a candidate peak must rise at least this fraction of
    the tallest peak's height above its surrounding valley to count --
    filters out small KDE ripples from being mistaken for genuine modes.
    Validated against E01002702's known cells before use at scale (see
    docs/az-family-work-plan.md Phase 3): correctly finds 1 mode for its
    confident years (e.g. 2012, 2017) and 2 for its genuinely bimodal ones
    (2013, 2014).
    """
    lo, hi = samples.min(), samples.max()
    if hi - lo < 1e-6:
        return np.array([samples.mean()]), np.array([1.0]), 1
    kde = stats.gaussian_kde(samples)
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_points)
    density = kde(grid)
    peaks, _ = find_peaks(density, prominence=prominence_frac * density.max())
    if len(peaks) == 0:
        peaks = np.array([np.argmax(density)])
    return grid[peaks], density[peaks], len(peaks)


def detect_z_multimodality(trace, prominence_frac=0.05):
    """
    Per-(area, year) scan for genuine multimodality in z's marginal
    posterior -- built after finding that AZ3's posterior MEAN for
    E01002702's 2013/2014 cells (34.5, 40.2) sat in a low-density valley
    between a ~50%-of-draws spike at ~0 and a broad secondary hump
    (~30-150), misrepresenting both explanations rather than describing
    either. Point-and-CI summaries (the spike-tracking plots, this
    function's own sibling diagnostics) silently assume a roughly
    unimodal marginal; this checks that assumption directly instead of
    taking it for granted.

    Root cause (see docs/az-family-work-plan.md Phase 3): a noise-mixture
    likelihood with an unusually tight signal-branch scale creates a
    near-discrete "matches almost exactly, or call it noise" choice per
    cell, and the zero-sum constraint couples that choice across an
    area's whole year row -- so this is expected to be most relevant for
    models with that kind of mixture likelihood (e.g. AZ3), though the
    scan itself is generic and works on any model's z.

    Parameters
    ----------
    trace           : az.InferenceData (posterior must contain 'z')
    prominence_frac : passed to _detect_modes -- see its docstring

    Returns
    -------
    pd.DataFrame, one row per (area, year), columns:
        area, year, area_idx, year_idx, n_modes, mode_locations
    Sorted by n_modes descending. Use area-level groupby to find which
    AREAS have any multimodal year, e.g.:
        df.groupby('area')['n_modes'].apply(lambda s: (s >= 2).sum())
    """
    z_da = trace.posterior['z']
    lsoa_codes = (z_da.coords['area'].values.tolist()
                 if 'area' in z_da.coords else list(range(z_da.sizes['area'])))
    z = z_da.values  # (chain, draw, area, year)
    n_areas, n_years = z.shape[2], z.shape[3]
    years = list(INFER_YEARS[:n_years])

    rows = []
    for a in range(n_areas):
        for t in range(n_years):
            samples = z[:, :, a, t].flatten()
            locs, _, n_modes = _detect_modes(samples, prominence_frac=prominence_frac)
            rows.append({
                'area': lsoa_codes[a], 'year': years[t],
                'area_idx': a, 'year_idx': t,
                'n_modes': n_modes,
                'mode_locations': np.round(locs, 1).tolist(),
            })

    return pd.DataFrame(rows).sort_values('n_modes', ascending=False).reset_index(drop=True)


def z_flatness_summary(trace, data, active_threshold=3.0, flat_range_threshold=2.0):
    """
    Direct check for the core failure mode this entire model-iteration
    effort exists to fix: z collapsing to a (near-)constant line per area
    regardless of P_obs/E_obs activity, instead of tracking real signal
    when the sources agree. r-hat, divergences, and chain agreement are
    silent on this — a model can converge cleanly to a flat z and look
    healthy on every sampling-quality metric while completely failing the
    actual modelling goal. This is a model-behaviour check, not a
    sampling-quality check, and is not a substitute for the others —
    run it alongside them, not instead of them.

    For each area:
      range_z              : max(z_mean) - min(z_mean) across years
                              (posterior mean z, dwellings/year) — the
                              simplest possible "did z do anything at
                              all this decade" measure.
      has_active_year       : True if |P_obs| or |E_obs| exceeds
                              active_threshold in at least one year (same
                              convention as
                              _build_temporal_reallocation_likelihood).
      is_flat               : range_z < flat_range_threshold.
      flat_despite_active   : is_flat AND has_active_year — the actual
                              pathology. A flat z in a genuinely quiet
                              area (no active year) is not a problem;
                              flat_range_threshold=2.0 dwellings/year is
                              deliberately lenient (well below typical
                              active-year magnitudes, which are often
                              tens of dwellings) so this doesn't overstate
                              the failure by penalising a small-but-real
                              tracking response.

    Parameters
    ----------
    trace                : az.InferenceData (posterior must contain 'z')
    data                 : data dict with 'P_obs', 'E_obs' (n_areas, n_years)
    active_threshold     : dwellings/year defining an "active" cell
    flat_range_threshold : dwellings/year — z counts as flat below this

    Returns
    -------
    pd.DataFrame with one row per area (area, range_z, has_active_year,
    is_flat, flat_despite_active); df.attrs['summary'] holds the
    aggregate fractions (frac_flat, frac_active, frac_flat_despite_active)
    for quick reporting.
    """
    z_mean = trace.posterior['z'].mean(dim=('chain', 'draw')).values  # (area, year)
    area_codes = (trace.posterior['z'].coords['area'].values
                  if 'area' in trace.posterior['z'].coords
                  else np.arange(z_mean.shape[0]))

    range_z = z_mean.max(axis=1) - z_mean.min(axis=1)

    P_obs = data['P_obs']
    E_obs = data['E_obs']
    has_active_year = ((np.abs(P_obs) > active_threshold) |
                        (np.abs(E_obs) > active_threshold)).any(axis=1)

    is_flat             = range_z < flat_range_threshold
    flat_despite_active = is_flat & has_active_year

    df = pd.DataFrame({
        'area':                area_codes,
        'range_z':             range_z,
        'has_active_year':     has_active_year,
        'is_flat':             is_flat,
        'flat_despite_active': flat_despite_active,
    })
    df.attrs['summary'] = {
        'frac_flat':               float(is_flat.mean()),
        'frac_active':              float(has_active_year.mean()),
        'frac_flat_despite_active': float(flat_despite_active.mean()),
    }
    return df


def full_diagnostics(trace, data, model=None, verbose=True):
    """
    Run all diagnostics and return combined results.

    Parameters
    ----------
    trace   : az.InferenceData
    data    : dict
    model   : DwellingModel instance (optional, for var_names)
    verbose : bool

    Returns
    -------
    dict with all diagnostic results
    """
    var_names = model.var_names if model is not None else None

    flatness = z_flatness_summary(trace, data)
    if verbose:
        s = flatness.attrs['summary']
        print("\n── z flatness (does z actually track P/E activity?) ──────────")
        print(f"  Areas with an active P/E year: {s['frac_active']:.1%}")
        print(f"  Areas where z is flat (range < 2/yr): {s['frac_flat']:.1%}")
        print(f"  Areas flat DESPITE an active year (the real pathology): "
              f"{s['frac_flat_despite_active']:.1%}")

    result = {
        'rhat':            _check_rhat(trace,             var_names=var_names, verbose=verbose),
        'divergences':     _check_divergences(trace,                           verbose=verbose),
        'chain_agreement': check_chain_agreement(trace, scalar_vars=var_names, verbose=verbose),
        'calibration':     _check_calibration(trace, data,                     verbose=verbose),
        'census':          _check_census_constraint(trace, data,               verbose=verbose),
        'residuals':       _check_residuals(trace, data,                       verbose=verbose),
        'morans_i':        _check_morans_i(trace, data,                        verbose=verbose),
        'z_flatness':      flatness,
    }

    sigma_slab_check = _check_sigma_slab_vs_disagreement(trace, data, verbose=verbose)
    if sigma_slab_check is not None:
        result['sigma_slab_vs_disagreement'] = sigma_slab_check

    kappa_check = _check_kappa_vs_recording_rate(trace, data, verbose=verbose)
    if kappa_check is not None:
        result['kappa_vs_recording_rate'] = kappa_check

    return result


def diagnostics_summary(traces, data=None, rhat_threshold=1.01, var_names=None):
    """
    Build a per-model diagnostic summary table.

    Parameters
    ----------
    traces         : dict mapping model name (str) to az.InferenceData
    data           : data dict (optional) — if provided, adds 90% coverage columns
    rhat_threshold : variables above this are counted in n_bad_rhat (default 1.01)
    var_names      : dict mapping model name -> list of scalar parameter
                      names (optional, e.g. {name: _ALL_MODELS[name].var_names
                      for name in traces} in cli.py). Restricts r-hat/ESS to
                      those variables instead of every variable in the
                      posterior.

                      This matters a lot: without it, az.rhat/az.ess run
                      ELEMENTWISE over every posterior variable, including
                      large per-(area, year) Deterministics (z, delta,
                      resp_same_P/E, resp_prior_P/E, resp_noise_P/E,
                      P_like_pointwise, E_like_pointwise, ...) — measured
                      directly via cProfile at ~220s for just 2 small
                      (200-area) models, ~98% of which was inside
                      az.rhat/az.ess (~14 million individual numpy calls).
                      z's own convergence is already covered far more
                      cheaply and more informatively by z_flatness_summary/
                      z_identifiability_summary, so re-checking it elementwise
                      here via r-hat/ESS is redundant as well as slow.

    Returns
    -------
    pd.DataFrame with index = model name and columns:
        frac_flat_despite_active [if data given], max_rhat, mean_rhat,
        n_bad_rhat, divergences, min_ess_bulk [, plan_cov_90, ben_cov_90]

    frac_flat_despite_active (see z_flatness_summary) is a MODEL-BEHAVIOUR
    check, not a sampling-quality check — it answers "does z actually move
    in areas with real P/E activity," which r-hat/divergences/ESS cannot.
    Listed first deliberately: a model can converge cleanly (good r-hat,
    no divergences) while still producing a flat z that ignores real
    signal, and that failure is the more important one to catch.
    """
    rows = {}
    for name, trace in traces.items():
        names   = var_names.get(name) if var_names else None
        rhat_ds = az.rhat(trace, var_names=names)
        ess_ds  = az.ess(trace, var_names=names, method='bulk')

        rhat_vals = np.concatenate([v.values.ravel() for v in rhat_ds.data_vars.values()])
        ess_vals  = np.concatenate([v.values.ravel() for v in ess_ds.data_vars.values()])
        rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
        ess_vals  = ess_vals[np.isfinite(ess_vals)]

        max_rhat    = float(rhat_vals.max())   if len(rhat_vals) else float('nan')
        mean_rhat   = float(rhat_vals.mean())  if len(rhat_vals) else float('nan')
        n_bad_rhat  = int((rhat_vals > rhat_threshold).sum())
        divs        = int(trace.sample_stats.diverging.sum())
        min_ess     = int(ess_vals.min())      if len(ess_vals) else -1
        chain_check = check_chain_agreement(trace, verbose=False)

        row = {}
        if data is not None:
            flatness = z_flatness_summary(trace, data)
            row['frac_flat_despite_active'] = flatness.attrs['summary']['frac_flat_despite_active']

        row.update({
            'max_rhat':       max_rhat,
            'mean_rhat':      mean_rhat,
            'n_bad_rhat':     n_bad_rhat,
            'divergences':    divs,
            'min_ess':        min_ess,
            'flagged_chains': len(chain_check['flagged_chains']),
        })

        if data is not None:
            cov = _check_calibration(trace, data, alpha=0.10, verbose=False)
            row['plan_cov_90'] = cov['planning']
            row['ben_cov_90']  = cov['ben']

        rows[name] = row

    df = pd.DataFrame(rows).T
    df['n_bad_rhat']     = df['n_bad_rhat'].astype(int)
    df['divergences']    = df['divergences'].astype(int)
    df['min_ess']        = df['min_ess'].astype(int)
    df['flagged_chains'] = df['flagged_chains'].astype(int)
    return df


def observation_summary(data, burst_threshold=30):
    """
    Summarise the empirical distribution of planning (P_obs) and BEN (E_obs)
    observations in the data dict.

    Useful for calibrating priors on z — the prior should broadly cover the
    range of observations without being wildly wider.

    Parameters
    ----------
    data            : data dict from make_data_dict
    burst_threshold : values above this are counted as 'burst' observations

    Returns
    -------
    pd.DataFrame with one row per source (planning, ben) and columns:
        p05, p25, p50, p75, p95, p99, mean, std, pct_negative, pct_burst, n_obs
    """
    rows = {}
    for label, obs in [('planning', data['P_obs']), ('ben', data['E_obs'])]:
        flat = obs.ravel()
        rows[label] = {
            'p05':          float(np.percentile(flat, 5)),
            'p25':          float(np.percentile(flat, 25)),
            'p50':          float(np.percentile(flat, 50)),
            'p75':          float(np.percentile(flat, 75)),
            'p95':          float(np.percentile(flat, 95)),
            'p99':          float(np.percentile(flat, 99)),
            'mean':         float(flat.mean()),
            'std':          float(flat.std()),
            'pct_negative': float(100 * np.mean(flat < 0)),
            'pct_burst':    float(100 * np.mean(flat > burst_threshold)),
            'n_obs':        int(len(flat)),
        }
    return pd.DataFrame(rows).T


def prior_predictive_summary(models, draws=500, burst_threshold=30, neg_threshold=0):
    """
    Run prior predictive simulation for one or more models and return a
    numerical summary of the implied z distribution.

    Parameters
    ----------
    models          : dict mapping name (str) to DwellingModel instance
                      (already built, or will be built here)
    draws           : number of prior predictive draws
    burst_threshold : z above this counts as a 'burst year' (default 30)
    neg_threshold   : z below this counts as implausible negative (default 0)

    Returns
    -------
    pd.DataFrame with index = model name and columns:
        z_p05, z_p25, z_p50, z_p75, z_p95, z_p99,
        z_mean, z_std, pct_negative, pct_burst
    """
    rows = {}
    for name, model in models.items():
        if model.model is None:
            model.build()
        prior = model.prior_predictive(draws=draws)
        z     = prior.prior['z'].values.ravel()

        rows[name] = {
            'z_p05':      float(np.percentile(z, 5)),
            'z_p25':      float(np.percentile(z, 25)),
            'z_p50':      float(np.percentile(z, 50)),
            'z_p75':      float(np.percentile(z, 75)),
            'z_p95':      float(np.percentile(z, 95)),
            'z_p99':      float(np.percentile(z, 99)),
            'z_mean':     float(z.mean()),
            'z_std':      float(z.std()),
            'pct_negative': float(100 * np.mean(z < neg_threshold)),
            'pct_burst':    float(100 * np.mean(z > burst_threshold)),
        }

    return pd.DataFrame(rows).T


