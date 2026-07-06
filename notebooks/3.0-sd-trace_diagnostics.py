# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing_projections.analysis import uncertainty_by_geography, variance_components
from housing_projections.config import DATA_PATH, INFER_YEARS, TRACES_DIR
from housing_projections.diagnostics import diagnostics_summary, observation_summary, prior_predictive_summary
from housing_projections.models import M0, M0h, M1
import housing_projections.data as data_utils
import housing_projections.outliers as outliers

# %% Configuration — edit these
MODELS_TO_DIAGNOSE = ['M0', 'M0h', 'M1']   # or None to load all found
RHAT_THRESHOLD     = 1.01
N_SAMPLE_AREAS     = 6   # areas to show in trace plots
N_SAMPLE_YEARS     = 3   # years to show per area (evenly spaced)
BURST_THRESHOLD    = 30  # dwellings/year above which a z value counts as a burst year

# %% Load traces
traces_dir = TRACES_DIR

def load_traces(names=None):
    import os
    if names is None:
        names = [p.stem for p in sorted(traces_dir.glob('*.nc'))]
    out = {}
    for name in names:
        path = traces_dir / f'{name}.nc'
        if path.exists():
            print(f'  Loading {name}')
            out[name] = az.from_netcdf(str(path))
        else:
            print(f'  [skip] {name}: not found at {path}')
    return out

traces = load_traces(MODELS_TO_DIAGNOSE)
print(f'\nLoaded: {list(traces)}')

# %% Load data (for coverage diagnostics)
gdf_raw         = data_utils.load_data(DATA_PATH)
gdf_clean, _    = outliers.apply_outlier_exclusion(gdf_raw)

first_trace = next(iter(traces.values()))
if 'area' in first_trace.posterior['z'].coords:
    lsoa_codes = first_trace.posterior['z'].coords['area'].values.tolist()
    gdf        = gdf_clean[gdf_clean['LSOA21CD'].isin(lsoa_codes)].copy()
    gdf        = gdf.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
    data       = data_utils.make_data_dict(gdf)
else:
    n = first_trace.posterior['z'].shape[2]
    data = data_utils.make_data_dict(gdf_clean, n_areas=n)

print(f'Data: {data["n_areas"]} areas, {data["n_years"]} years')

# %% Summary table
diag = diagnostics_summary(traces, data=data, rhat_threshold=RHAT_THRESHOLD)
print('\n── Diagnostics summary ──────────────────────────────────────')
print(diag.to_string(
    formatters={
        'max_rhat':    '{:.4f}'.format,
        'mean_rhat':   '{:.4f}'.format,
        'n_bad_rhat':  '{:d}'.format,
        'divergences': '{:d}'.format,
        'min_ess':     '{:d}'.format,
        'plan_cov_90': '{:.3f}'.format,
        'ben_cov_90':  '{:.3f}'.format,
    }
))

# %% R-hat distribution per model
fig, axes = plt.subplots(1, len(traces), figsize=(4 * len(traces), 3), sharey=False)
if len(traces) == 1:
    axes = [axes]

for ax, (name, trace) in zip(axes, traces.items()):
    rhat_ds   = az.rhat(trace)
    rhat_vals = np.concatenate([v.values.ravel() for v in rhat_ds.data_vars.values()])
    rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
    ax.hist(rhat_vals, bins=50, color='steelblue', edgecolor='none')
    ax.axvline(RHAT_THRESHOLD, color='red', linestyle='--', linewidth=1, label=f'{RHAT_THRESHOLD}')
    ax.set_title(name)
    ax.set_xlabel('R-hat')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

fig.suptitle('R-hat distribution by model', fontsize=12)
plt.tight_layout()
plt.show()

# %% Per-model: inspect the worst variables
for name, trace in traces.items():
    rhat_ds   = az.rhat(trace)
    rows = []
    for var, da in rhat_ds.data_vars.items():
        vals = da.values.ravel()
        for i, v in enumerate(vals):
            if np.isfinite(v) and v > RHAT_THRESHOLD:
                rows.append({'var': var, 'index': i, 'rhat': v})
    if not rows:
        print(f'\n{name}: all r-hats ≤ {RHAT_THRESHOLD}')
        continue
    df_bad = pd.DataFrame(rows).sort_values('rhat', ascending=False)
    print(f'\n{name}: {len(df_bad)} variables above {RHAT_THRESHOLD} '
          f'(worst: {df_bad.iloc[0]["var"]} = {df_bad.iloc[0]["rhat"]:.4f})')
    print(df_bad.groupby('var')['rhat'].agg(['count', 'max', 'mean']).sort_values('max', ascending=False).to_string())

# %% Trace plots — sample a few z variables
for name, trace in traces.items():
    z_post  = trace.posterior['z']
    n_areas = z_post.shape[2]
    idx     = np.linspace(0, n_areas - 1, N_SAMPLE_AREAS, dtype=int)

    if 'area' in z_post.coords:
        area_coords = z_post.coords['area'].values[idx].tolist()
    else:
        area_coords = list(idx)

    year_dim  = [d for d in z_post.dims if d not in ('chain', 'draw', 'area')][0]
    n_years   = z_post.sizes[year_dim]
    year_idx  = np.linspace(0, n_years - 1, min(N_SAMPLE_YEARS, n_years), dtype=int)
    year_coords = (z_post.coords[year_dim].values[year_idx].tolist()
                   if year_dim in z_post.coords else list(year_idx))

    n_plots = N_SAMPLE_AREAS * N_SAMPLE_YEARS
    az.plot_trace(
        trace,
        var_names=['z'],
        coords={'area': area_coords, year_dim: year_coords},
        figure_kwargs={'figsize': (12, 2 * n_plots)},
    )
    plt.suptitle(f'{name} — z trace ({N_SAMPLE_AREAS} areas × {N_SAMPLE_YEARS} years)', fontsize=11)
    plt.tight_layout()
    plt.show()

# %% Energy plot — detects geometry / funnel issues
for name, trace in traces.items():
    az.plot_energy(trace, figure_kwargs={'figsize': (6, 3)})
    plt.suptitle(f'{name} — energy plot')
    plt.tight_layout()
    plt.show()

# %% Pair plot for scalar parameters (helpful for spotting funnels)
SCALAR_VARS = {
    'M0':  ['sigma_plan', 'sigma_ben'],
    'M0h': ['mu_global', 'sigma_mu', 'sigma_slab', 'sigma_plan', 'sigma_ben'],
    'M1':  ['sigma_plan', 'sigma_ben', 'lambda_weights'],
}

for name, trace in traces.items():
    vars_to_plot = [v for v in SCALAR_VARS.get(name, [])
                    if v in trace.posterior]
    if len(vars_to_plot) < 2:
        continue
    n = len(vars_to_plot)
    az.plot_pair(
        trace,
        var_names=vars_to_plot,
        visuals={'divergence': True},
        figure_kwargs={'figsize': (3 * n, 3 * n)},
    )
    plt.suptitle(f'{name} — scalar parameter pairs (red = divergences)', fontsize=11)
    plt.tight_layout()
    plt.show()

# %% Posterior of sigma_plan / sigma_ben across models
sig_models = {
    name: trace for name, trace in traces.items()
    if 'sigma_plan' in trace.posterior
}
if sig_models:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for name, trace in sig_models.items():
        sp = trace.posterior['sigma_plan'].values.ravel()
        sb = trace.posterior['sigma_ben'].values.ravel()
        axes[0].hist(sp, bins=60, alpha=0.5, label=name, density=True)
        axes[1].hist(sb, bins=60, alpha=0.5, label=name, density=True)
    axes[0].set_title('sigma_plan posterior')
    axes[1].set_title('sigma_ben posterior')
    for ax in axes:
        ax.set_xlabel('value')
        ax.legend()
    plt.tight_layout()
    plt.show()

# %% Observation summary — empirical distribution of P_obs and E_obs
obs_summary = observation_summary(data, burst_threshold=BURST_THRESHOLD)
print(f'\n── Observed data summary (burst threshold = {BURST_THRESHOLD}) ──')
print(obs_summary.to_string(float_format='{:.2f}'.format))
print('\nUse these to calibrate the z prior:')
print('  pct_negative should be in the same ballpark as the prior')
print('  pct_burst should be roughly matched by prior pct_burst')
print('  p99 gives the scale of genuine large observations')

# %% Prior predictive summary — run before/after model changes to compare z prior
# Edit this dict to include the models you want to check.
# A 'burst year' is z > BURST_THRESHOLD dwellings in a single year for one area.
prior_models = {
    'M0':  M0(data),
    'M0h': M0h(data),
}
pp_summary = prior_predictive_summary(
    prior_models, draws=500, burst_threshold=BURST_THRESHOLD
)
print(f'\n── Prior predictive z summary (burst threshold = {BURST_THRESHOLD}) ──')
print(pp_summary.to_string(float_format='{:.2f}'.format))
print('\nInterpretation:')
print('  pct_negative : fraction of prior z draws < 0 (implausible)')
print(f'  pct_burst    : fraction of prior z draws > {BURST_THRESHOLD} (burst years)')
print('  z_p99        : 99th percentile — how fat are the tails?')

# %% Uncertainty decomposition across geographic levels (M0h)
# Requires M0h to be loaded in `traces`.

if 'M0h' in traces:
    geo = uncertainty_by_geography(traces['M0h'])

    print(f'\n── Uncertainty by geographic level (M0h, {INFER_YEARS[0]}–{INFER_YEARS[-1]}) ──')
    fmt = {'median_cv': '{:.3f}'.format, 'cv_p90': '{:.3f}'.format, 'median_sd': '{:.1f}'.format}
    print(geo['summary'].to_string(formatters=fmt))

    print('\n── Borough-level posterior (sorted by posterior mean) ────────────')
    fmt_b = {
        'n_lsoas':   '{:d}'.format,
        'post_mean': '{:.0f}'.format,
        'post_sd':   '{:.0f}'.format,
        'cv':        '{:.3f}'.format,
        'ci90_lo':   '{:.0f}'.format,
        'ci90_hi':   '{:.0f}'.format,
    }
    print(geo['borough'].to_string(index=False, formatters=fmt_b))

    vc = variance_components(traces['M0h'])
    if vc:
        print('\n── M0h variance components ───────────────────────────────────────')
        print(f'  sigma_mu   (between-area):  mean={vc["sigma_mu"]["mean"]:.2f}  '
              f'SD={vc["sigma_mu"]["sd"]:.2f}')
        print(f'  sigma_slab (within-area):   mean={vc["sigma_slab"]["mean"]:.2f}  '
              f'SD={vc["sigma_slab"]["sd"]:.2f}')
        print(f'  Ratio sigma_mu/sigma_slab:  {vc["ratio"]:.2f}')
        print('  (>1 means between-area variation dominates; '
              'aggregation to borough does not cancel much uncertainty)')
else:
    print('M0h not found in traces — set MODELS_TO_DIAGNOSE to include M0h')


# %% Census constraint check — how well does z sum match D?
for name, trace in traces.items():
    z_post    = trace.posterior['z'].values          # (chains, draws, areas, years)
    z_sums    = z_post.sum(axis=-1).reshape(-1, data['n_areas'])
    residuals = z_sums - data['D'][None, :]
    print(f'\n{name} census residuals (z.sum - D):')
    print(f'  mean abs: {np.abs(residuals).mean():.2f}  '
          f'max abs: {np.abs(residuals).max():.2f}  '
          f'std: {residuals.std():.2f}')
