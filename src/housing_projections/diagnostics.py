import arviz as az
import numpy as np
import pandas as pd

from housing_projections.spatial import (
    build_spatial_weights,
    build_weights_libpysal,
    compute_morans_i,
)


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


def _check_calibration(trace, data, alpha=0.1, verbose=False):
    """
    Check calibration coverage of posterior credible intervals
    against planning and BEN observations.

    Returns
    -------
    dict with keys 'planning', 'ben'
    """
    z_post   = trace.posterior['z'].values         # (chains, draws, n_areas, n_years)
    z_lo     = np.percentile(z_post, 100 * alpha / 2,       axis=(0, 1))
    z_hi     = np.percentile(z_post, 100 * (1 - alpha / 2), axis=(0, 1))

    P_obs    = data['P_obs']
    E_obs    = data['E_obs']

    coverage = {
        'planning': float(np.mean((P_obs >= z_lo) & (P_obs <= z_hi))),
        'ben':      float(np.mean((E_obs >= z_lo) & (E_obs <= z_hi))),
    }

    if verbose:
        print(f"\n── Calibration ({int((1-alpha)*100)}% CI) ────────────────────")
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
        summary   = az.summary(trace)
        rhat_vals = pd.to_numeric(summary['r_hat'], errors='coerce').dropna()
        ess_vals  = pd.to_numeric(summary.get('ess_bulk', pd.Series(dtype=float)),
                                  errors='coerce').dropna()

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


def compute_model_comparison(traces, verbose=True):
    """
    Compare models using Leave-One-Out cross-validation (LOO-CV).

    Uses LOO-CV (PSIS-LOO). Requires traces sampled with
    ``idata_kwargs={'log_likelihood': True}`` (the default sampling config
    already sets this).

    Parameters
    ----------
    traces  : dict mapping model name (str) to az.InferenceData
    verbose : bool — if True, print the comparison table

    Returns
    -------
    pd.DataFrame — ArviZ LOO comparison table, models ranked best-to-worst.
        Key columns: ``elpd``, ``se``, ``p``, ``elpd_diff``, ``weight``.
    """
    # Use P_like (planning likelihood) for LOO comparison. All models share this
    # variable with consistent shape (n_areas × n_years). Using var_name avoids
    # az.compare failing when multiple log_likelihood variables with different
    # shapes (P_like, E_like, census_obs) are present in the trace.
    comparison = az.compare(traces, var_name='P_like')

    if verbose:
        print("\n── LOO model comparison ─────────────────────────────────────")
        display_cols = [c for c in ('elpd', 'se', 'p', 'elpd_diff', 'weight')
                        if c in comparison.columns]
        print(comparison[display_cols].to_string())
        best = comparison.index[0]
        print(f"\n  Best model: {best}")

    return comparison


# ── M3 specific ───────────────────────────────────────────────────────────────

def compute_lag_weights(trace, verbose=False):
    """
    Extract posterior lag weight statistics from M3 trace.

    Returns
    -------
    dict with keys 'means', 'lo', 'hi', 'implied_mean_lag'
    """
    lambda_post = trace.posterior['lambda_weights'].values
    lambda_flat = lambda_post.reshape(-1, lambda_post.shape[-1])
    n_lags      = lambda_flat.shape[-1]
    lags        = list(range(n_lags))

    means = lambda_flat.mean(axis=0)
    lo    = np.percentile(lambda_flat, 5,  axis=0)
    hi    = np.percentile(lambda_flat, 95, axis=0)
    implied_mean_lag = sum(k * means[k] for k in lags)

    result = {
        'means':            means,
        'lo':               lo,
        'hi':               hi,
        'implied_mean_lag': implied_mean_lag,
        'n_lags':           n_lags,
        'lambda_flat':      lambda_flat,
    }

    if verbose:
        print("\n── M3 lag weights ───────────────────────────────────────────")
        for k, (m, lo_k, h) in enumerate(zip(means, lo, hi)):
            print(f"  lag {k}: mean={m:.3f}  90% CI=[{lo_k:.3f}, {h:.3f}]")
        print(f"\n  Implied mean lag: {implied_mean_lag:.2f} years")

    return result


def compute_lag_residuals(trace, data, lambda_weights=None):
    """
    Compute planning residuals with and without lag correction.

    Parameters
    ----------
    trace          : az.InferenceData
    data           : dict
    lambda_weights : array-like, optional
        Fixed lag weights to use when lambda_weights is not a sampled
        variable in the posterior (e.g. M6 where they are a model constant).
        Ignored if lambda_weights is present in trace.posterior.

    Returns
    -------
    dict with keys 'no_lag', 'with_lag'
    each of shape (n_areas, n_years)
    """
    z_post = trace.posterior['z'].values
    z_mean = z_post.mean(axis=(0, 1))
    plain  = data['P_obs'] - z_mean

    if 'lambda_weights' in trace.posterior:
        # Sampled weights — average over chains and draws
        lambda_mean = trace.posterior['lambda_weights'].values.mean(axis=(0, 1))
    elif lambda_weights is not None:
        # Fixed weights passed in explicitly by the caller
        lambda_mean = np.asarray(lambda_weights)
    else:
        # Model has no lag component at all
        return {'no_lag': plain, 'with_lag': plain}

    n_lags  = len(lambda_mean)
    n_years     = data['n_years']

    P_mean_post = np.zeros_like(z_mean)
    for t in range(n_years):
        for k in range(n_lags):
            t_src               = max(t - k, 0)
            P_mean_post[:, t]  += lambda_mean[k] * z_mean[:, t_src]

    return {
        'no_lag':   plain,
        'with_lag': data['P_obs'] - P_mean_post,
    }


def compute_spatial_misallocation_stats(trace, data):
    """
    Compute spatial misallocation diagnostics from M6 trace.
    Returns dict with alpha_spatial posterior stats and
    spatial lag correlation of z.
    """
    alpha_post = trace.posterior['alpha_spatial'].values.ravel()
    z_mean     = trace.posterior['z'].values.mean(axis=(0, 1))

    W     = build_spatial_weights(data['gdf'])
    z_lag = (W @ z_mean).ravel()
    z_flat = z_mean.ravel()

    return {
        'alpha_mean':  alpha_post.mean(),
        'alpha_std':   alpha_post.std(),
        'alpha_lo':    np.percentile(alpha_post, 5),
        'alpha_hi':    np.percentile(alpha_post, 95),
        'alpha_post':  alpha_post,
        'z_flat':      z_flat,
        'z_lag':       z_lag,
    }
