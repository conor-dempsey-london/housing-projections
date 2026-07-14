# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import housing_projections.data as data_utils
import housing_projections.models as models_pkg
import housing_projections.outliers as outliers
from housing_projections.config import DATA_PATH, INFER_YEARS, TRACES_DIR

# %% Configuration — edit these, then re-run the cells below
MODEL_NAME     = 'M11'   # any model name with a saved trace in results/traces/
RHAT_THRESHOLD = 1.01
N_SAMPLE_AREAS = 6       # areas to show in the z-timeseries plot

# %% Load data (once)
gdf_raw      = data_utils.load_data(DATA_PATH)
gdf_clean, _ = outliers.apply_outlier_exclusion(gdf_raw)

# %% Load the trace for MODEL_NAME and rebuild the matching data dict
trace_path = TRACES_DIR / f'{MODEL_NAME}.nc'
trace      = az.from_netcdf(str(trace_path))

z_post     = trace.posterior['z']
lsoa_codes = (z_post.coords['area'].values.tolist()
              if 'area' in z_post.coords else None)

if lsoa_codes is not None:
    gdf  = gdf_clean[gdf_clean['LSOA21CD'].isin(lsoa_codes)].copy()
    gdf  = gdf.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
    data = data_utils.make_data_dict(gdf)
else:
    data = data_utils.make_data_dict(gdf_clean, n_areas=z_post.shape[2])

model = getattr(models_pkg, MODEL_NAME)(data)

print(f'{MODEL_NAME}: {data["n_areas"]} areas, {data["n_years"]} years, '
      f'{trace.posterior.sizes["chain"]} chains x {trace.posterior.sizes["draw"]} draws')

# %% Sample z-timeseries vs P_obs/E_obs
def select_diverse_areas(P_obs, n):
    """n area indices spanning low->high P_obs temporal variability (CV)."""
    means = np.abs(P_obs).mean(axis=1)
    stds  = P_obs.std(axis=1)
    cv    = np.where(means > 0.5, stds / means, 0.0)
    thresholds = np.percentile(cv, np.linspace(0, 100, n + 2)[1:-1])
    return [int(np.argmin(np.abs(cv - t))) for t in thresholds]

z_vals     = z_post.values   # (chain, draw, area, year)
C, S, A, T = z_vals.shape
z_flat     = z_vals.reshape(C * S, A, T)
z_mean     = z_flat.mean(axis=0)
z_lo       = np.percentile(z_flat,  5, axis=0)
z_hi       = np.percentile(z_flat, 95, axis=0)

area_idx = select_diverse_areas(data['P_obs'], N_SAMPLE_AREAS)
years    = np.array(INFER_YEARS)
ncols    = 3
nrows    = int(np.ceil(N_SAMPLE_AREAS / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
axes = np.array(axes).ravel()

for ax, area_i in zip(axes, area_idx):
    ax.fill_between(years, z_lo[area_i], z_hi[area_i],
                    alpha=0.25, color='steelblue', label='z 90% CI')
    ax.plot(years, z_mean[area_i], color='steelblue', linewidth=1.5, label='z mean')
    ax.plot(years, data['P_obs'][area_i], 'x', color='darkorange',
            markersize=5, label='P_obs')
    ax.plot(years, data['E_obs'][area_i], 'o', color='forestgreen',
            markersize=4, fillstyle='none', label='E_obs')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    code = lsoa_codes[area_i] if lsoa_codes is not None else area_i
    ax.set_title(f'{str(code)[:12]}  D={data["D"][area_i]:.0f}  '
                 f'Σz={z_mean[area_i].sum():.0f}', fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[len(area_idx):]:
    ax.set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=8, ncol=2)
fig.suptitle(f'{MODEL_NAME} — z posterior vs observations', fontsize=11)
plt.tight_layout()
plt.show()

# %% Sample traces — scalar hyperparameters
scalar_vars = [v for v in model.var_names if v in trace.posterior]
az.plot_trace(trace, var_names=scalar_vars,
              figure_kwargs={'figsize': (12, 2 * len(scalar_vars))})
plt.suptitle(f'{MODEL_NAME} — scalar parameter traces', fontsize=11)
plt.tight_layout()
plt.show()

# %% R-hat summary — which variables have poor convergence
rows = []
for var, da in az.rhat(trace).data_vars.items():
    vals = da.values.ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        continue
    rows.append({
        'variable':  var,
        'n_params':  len(vals),
        'max_rhat':  vals.max(),
        'mean_rhat': vals.mean(),
        'frac_bad':  (vals > RHAT_THRESHOLD).mean(),
    })
rhat_summary  = pd.DataFrame(rows).sort_values('max_rhat', ascending=False)
n_divergences = int(trace.sample_stats.diverging.sum())

print(f'\n── {MODEL_NAME} r-hat summary (threshold={RHAT_THRESHOLD}) '
      f'────────────────────')
print(f'Divergences: {n_divergences}')
print(rhat_summary.to_string(index=False, formatters={
    'max_rhat':  '{:.3f}'.format,
    'mean_rhat': '{:.3f}'.format,
    'frac_bad':  '{:.2%}'.format,
}))
