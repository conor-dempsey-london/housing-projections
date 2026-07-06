# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing_projections.config import DATA_PATH, TRACES_DIR
from housing_projections.diagnostics import diagnostics_summary
import housing_projections.data as data_utils
import housing_projections.outliers as outliers

# %% Configuration — edit these
MODELS_TO_DIAGNOSE = ['M0', 'M0h', 'M1']   # or None to load all found
RHAT_THRESHOLD     = 1.01
N_SAMPLE_AREAS     = 3   # areas to show in trace plots
N_SAMPLE_YEARS     = 2   # years per area to show in trace plots

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

    n_years = z_post.shape[3]
    year_idx = np.linspace(0, n_years - 1, min(N_SAMPLE_YEARS, n_years), dtype=int)
    year_dim = [d for d in z_post.dims if d not in ('chain', 'draw', 'area')][0]
    if year_dim in z_post.coords:
        year_coords = z_post.coords[year_dim].values[year_idx].tolist()
    else:
        year_coords = list(year_idx)

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
        divergences=True,
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

# %% Census constraint check — how well does z sum match D?
for name, trace in traces.items():
    z_post    = trace.posterior['z'].values          # (chains, draws, areas, years)
    z_sums    = z_post.sum(axis=-1).reshape(-1, data['n_areas'])
    residuals = z_sums - data['D'][None, :]
    print(f'\n{name} census residuals (z.sum - D):')
    print(f'  mean abs: {np.abs(residuals).mean():.2f}  '
          f'max abs: {np.abs(residuals).max():.2f}  '
          f'std: {residuals.std():.2f}')
