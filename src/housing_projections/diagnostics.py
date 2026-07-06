import arviz as az
import numpy as np
import pandas as pd
import scipy.stats as stats

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

    return {
        'rhat':        _check_rhat(trace,             var_names=var_names, verbose=verbose),
        'divergences': _check_divergences(trace,                           verbose=verbose),
        'calibration': _check_calibration(trace, data,                     verbose=verbose),
        'census':      _check_census_constraint(trace, data,               verbose=verbose),
        'residuals':   _check_residuals(trace, data,                       verbose=verbose),
        'morans_i':    _check_morans_i(trace, data,                        verbose=verbose),
    }


def diagnostics_summary(traces, data=None, rhat_threshold=1.01):
    """
    Build a per-model diagnostic summary table.

    Parameters
    ----------
    traces         : dict mapping model name (str) to az.InferenceData
    data           : data dict (optional) — if provided, adds 90% coverage columns
    rhat_threshold : variables above this are counted in n_bad_rhat (default 1.01)

    Returns
    -------
    pd.DataFrame with index = model name and columns:
        max_rhat, mean_rhat, n_bad_rhat, divergences, min_ess_bulk
        [, plan_cov_90, ben_cov_90]
    """
    rows = {}
    for name, trace in traces.items():
        rhat_ds = az.rhat(trace)
        ess_ds  = az.ess(trace, method='bulk')

        rhat_vals = np.concatenate([v.values.ravel() for v in rhat_ds.data_vars.values()])
        ess_vals  = np.concatenate([v.values.ravel() for v in ess_ds.data_vars.values()])
        rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
        ess_vals  = ess_vals[np.isfinite(ess_vals)]

        max_rhat    = float(rhat_vals.max())   if len(rhat_vals) else float('nan')
        mean_rhat   = float(rhat_vals.mean())  if len(rhat_vals) else float('nan')
        n_bad_rhat  = int((rhat_vals > rhat_threshold).sum())
        divs        = int(trace.sample_stats.diverging.sum())
        min_ess     = int(ess_vals.min())      if len(ess_vals) else -1

        row = {
            'max_rhat':    max_rhat,
            'mean_rhat':   mean_rhat,
            'n_bad_rhat':  n_bad_rhat,
            'divergences': divs,
            'min_ess':     min_ess,
        }

        if data is not None:
            cov = _check_calibration(trace, data, alpha=0.10, verbose=False)
            row['plan_cov_90'] = cov['planning']
            row['ben_cov_90']  = cov['ben']

        rows[name] = row

    df = pd.DataFrame(rows).T
    df['n_bad_rhat']  = df['n_bad_rhat'].astype(int)
    df['divergences'] = df['divergences'].astype(int)
    df['min_ess']     = df['min_ess'].astype(int)
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


