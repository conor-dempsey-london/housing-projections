import arviz as az
import gla_data._ons
import numpy as np
import pandas as pd

from housing_projections.spatial import build_spatial_weights

__all__ = [
    "compute_model_comparison",
    "compute_lag_weights",
    "compute_lag_residuals",
    "compute_spatial_misallocation_stats",
    "uncertainty_by_geography",
    "variance_components",
]


def _joint_pe_loglik(idata, name):
    """
    Return (idata, var_name) where var_name is the pointwise log-likelihood
    variable to hand to az.compare, scoring BOTH observation sources jointly
    rather than P_like alone.

    - If a joint variable ('PE_like') already exists, used as-is.
    - If both 'P_like' and 'E_like' exist, sums them elementwise into a new
      'PE_like' variable on a copy of idata -- valid because every P/E
      likelihood in this codebase is conditionally independent given z, so
      the joint pointwise log-density is just the elementwise sum. Summed via
      raw .values, not xarray's own '+', and wrapped back into P_like's own
      dims/coords: some older traces (e.g. AZ0a) attached P_like/E_like with
      each variable's own auto-generated dim names (P_like_dim_0 vs
      E_like_dim_0) rather than shared ('area', 'year') dims, and xarray's
      dimension-name alignment treats differently-named dims of the same
      length as genuinely different axes -- '+' silently broadcasts a full
      outer product (200x10 vs 200x10 becomes 200x10x200x10) instead of
      raising, which is a multi-hundred-GB memory blowup waiting to happen.
    - If only one source exists (e.g. a single-source test fixture), falls
      back to that one alone.
    """
    ll = idata['log_likelihood']
    if 'PE_like' in ll.data_vars:
        return idata, 'PE_like'
    has_p, has_e = 'P_like' in ll.data_vars, 'E_like' in ll.data_vars
    if has_p and has_e:
        p_like, e_like = ll['P_like'], ll['E_like']
        if p_like.shape != e_like.shape:
            raise ValueError(f"{name}: P_like {p_like.shape} and E_like "
                             f"{e_like.shape} have different shapes -- "
                             f"cannot be summed into a joint likelihood")
        idata = idata.copy()
        idata['log_likelihood']['PE_like'] = p_like.copy(
            data=p_like.values + e_like.values)
        return idata, 'PE_like'
    if has_p:
        return idata, 'P_like'
    if has_e:
        return idata, 'E_like'
    raise ValueError(f"{name}: log_likelihood group has none of 'PE_like', "
                     f"'P_like', 'E_like' -- found {list(ll.data_vars)}")


def compute_model_comparison(traces, verbose=True):
    """
    Compare models using Leave-One-Out cross-validation (LOO-CV).

    Uses LOO-CV (PSIS-LOO). Requires traces sampled with
    ``idata_kwargs={'log_likelihood': True}`` (the default sampling config
    already sets this). Scores the JOINT P+E pointwise log-likelihood
    (elementwise sum of 'P_like'/'E_like', see _joint_pe_loglik) when both
    are present, not 'P_like' alone -- a model that fits P well but E badly
    (or vice versa) should not look artificially good here.

    Parameters
    ----------
    traces  : dict mapping model name (str) to az.InferenceData
    verbose : bool — if True, print the comparison table

    Returns
    -------
    pd.DataFrame — ArviZ LOO comparison table, models ranked best-to-worst.
        Key columns: ``elpd``, ``se``, ``p``, ``elpd_diff``, ``weight``.
    """
    joint_traces, var_names_used = {}, set()
    for name, idata in traces.items():
        joint_idata, var_name = _joint_pe_loglik(idata, name)
        joint_traces[name] = joint_idata
        var_names_used.add(var_name)

    if len(var_names_used) > 1:
        raise ValueError(
            f"Models resolve to different log-likelihood variables after "
            f"joining ({sorted(var_names_used)}) -- az.compare needs the "
            f"same var_name across every model being compared, so these "
            f"models aren't directly comparable as given.")

    comparison = az.compare(joint_traces, var_name=var_names_used.pop())

    if verbose:
        print("\n── LOO model comparison ─────────────────────────────────────")
        display_cols = [c for c in ('elpd', 'se', 'p', 'elpd_diff', 'weight')
                        if c in comparison.columns]
        print(comparison[display_cols].to_string())
        best = comparison.index[0]
        print(f"\n  Best model: {best}")

    return comparison


def compute_lag_weights(trace, verbose=False):
    """
    Extract posterior lag weight statistics from a trace with lambda_weights.

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
        print("\n── Lag weights ──────────────────────────────────────────────")
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
        variable in the posterior.

    Returns
    -------
    dict with keys 'no_lag', 'with_lag', each of shape (n_areas, n_years)
    """
    z_post = trace.posterior['z'].values
    z_mean = z_post.mean(axis=(0, 1))
    plain  = data['P_obs'] - z_mean

    if 'lambda_weights' in trace.posterior:
        lambda_mean = trace.posterior['lambda_weights'].values.mean(axis=(0, 1))
    elif lambda_weights is not None:
        lambda_mean = np.asarray(lambda_weights)
    else:
        return {'no_lag': plain, 'with_lag': plain}

    n_lags  = len(lambda_mean)
    n_years = data['n_years']

    P_mean_post = np.zeros_like(z_mean)
    for t in range(n_years):
        for k in range(n_lags):
            t_src              = max(t - k, 0)
            P_mean_post[:, t] += lambda_mean[k] * z_mean[:, t_src]

    return {
        'no_lag':   plain,
        'with_lag': data['P_obs'] - P_mean_post,
    }


def compute_spatial_misallocation_stats(trace, data):
    """
    Compute spatial misallocation diagnostics from a trace with alpha_spatial.

    Returns
    -------
    dict with alpha_spatial posterior stats and spatial lag correlation of z.
    """
    alpha_post = trace.posterior['alpha_spatial'].values.ravel()
    z_mean     = trace.posterior['z'].values.mean(axis=(0, 1))

    W      = build_spatial_weights(data['gdf'])
    z_lag  = (W @ z_mean).ravel()
    z_flat = z_mean.ravel()

    return {
        'alpha_mean': alpha_post.mean(),
        'alpha_std':  alpha_post.std(),
        'alpha_lo':   np.percentile(alpha_post, 5),
        'alpha_hi':   np.percentile(alpha_post, 95),
        'alpha_post': alpha_post,
        'z_flat':     z_flat,
        'z_lag':      z_lag,
    }


def uncertainty_by_geography(trace, lsoa_codes=None):
    """
    Aggregate z posterior to MSOA and Borough level and compute uncertainty
    statistics at each geographic level.

    Demonstrates that posterior uncertainty is naturally consistent across
    geographies — higher-level estimates are just sums of the same posterior
    draws, so credible intervals are coherent by construction.

    Parameters
    ----------
    trace      : az.InferenceData — must contain posterior z with area coord
    lsoa_codes : list of str, optional — if None, read from trace coordinates

    Returns
    -------
    dict with keys 'lsoa', 'msoa', 'borough', each a pd.DataFrame with columns:
        n_lsoas, post_mean, post_sd, cv, ci90_lo, ci90_hi
    and a 'summary' DataFrame with one row per level showing median CV.
    """
    z_post = trace.posterior['z'].values           # (chains, draws, areas, years)
    C, S, A, T = z_post.shape
    z_flat = z_post.reshape(C * S, A, T)           # (draws, areas, years)

    if lsoa_codes is None:
        lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()

    lookup = gla_data._ons.fetch_geography_lookup(2021, 'lsoa')
    lookup = lookup[lookup['LSOA21CD'].isin(lsoa_codes)].copy()
    lookup = lookup.set_index('LSOA21CD').loc[lsoa_codes].reset_index()

    lsoa_to_msoa    = lookup.set_index('LSOA21CD')['MSOA21CD']
    lsoa_to_borough = lookup.set_index('LSOA21CD')['LAD22NM']

    def _stats_for_group(group_series, level_col):
        rows = []
        for grp in group_series.unique():
            idx   = np.where(group_series.values == grp)[0]
            z_grp = z_flat[:, idx, :].sum(axis=(1, 2))
            mean  = z_grp.mean()
            rows.append({
                level_col:   grp,
                'n_lsoas':   len(idx),
                'post_mean': mean,
                'post_sd':   z_grp.std(),
                'cv':        z_grp.std() / abs(mean) if mean != 0 else np.nan,
                'ci90_lo':   np.percentile(z_grp,  5),
                'ci90_hi':   np.percentile(z_grp, 95),
            })
        return pd.DataFrame(rows).sort_values('post_mean', ascending=False)

    z_lsoa_total = z_flat.sum(axis=2)              # (draws, areas)
    lsoa_means   = z_lsoa_total.mean(axis=0)
    df_lsoa = pd.DataFrame({
        'lsoa':      lsoa_codes,
        'n_lsoas':   1,
        'post_mean': lsoa_means,
        'post_sd':   z_lsoa_total.std(axis=0),
        'cv':        z_lsoa_total.std(axis=0) / np.abs(lsoa_means),
        'ci90_lo':   np.percentile(z_lsoa_total,  5, axis=0),
        'ci90_hi':   np.percentile(z_lsoa_total, 95, axis=0),
    })

    df_msoa    = _stats_for_group(lsoa_to_msoa,    'msoa')
    df_borough = _stats_for_group(lsoa_to_borough, 'borough')

    summary_rows = []
    for label, df in [('LSOA', df_lsoa), ('MSOA', df_msoa), ('Borough', df_borough)]:
        summary_rows.append({
            'level':      label,
            'n':          len(df),
            'median_cv':  df['cv'].median(),
            'cv_p90':     df['cv'].quantile(0.9),
            'median_sd':  df['post_sd'].median(),
        })

    return {
        'lsoa':    df_lsoa,
        'msoa':    df_msoa,
        'borough': df_borough,
        'summary': pd.DataFrame(summary_rows).set_index('level'),
    }


def variance_components(trace):
    """
    Extract variance component posteriors from a hierarchical trace (M0h).

    Returns posterior summaries of sigma_mu (between-area) and sigma_slab
    (within-area temporal), and their ratio. Returns an empty dict if neither
    variable is present in the trace.

    Parameters
    ----------
    trace : az.InferenceData

    Returns
    -------
    dict with keys 'sigma_mu', 'sigma_slab', 'ratio' (each a dict with
    'mean' and 'sd'), or empty dict if variables not found.
    """
    posterior = trace.posterior
    result    = {}

    for key in ('sigma_mu', 'sigma_slab'):
        if key in posterior:
            vals = posterior[key].values.ravel()
            result[key] = {'mean': float(vals.mean()), 'sd': float(vals.std())}

    if 'sigma_mu' in result and 'sigma_slab' in result:
        ratio = result['sigma_mu']['mean'] / result['sigma_slab']['mean']
        result['ratio'] = ratio

    return result
