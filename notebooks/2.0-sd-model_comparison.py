# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import time
import arviz as az
import pymc as pm
import numpy as np
from pathlib import Path

import housing_projections.data as data_utils
import housing_projections.outliers as outliers
import housing_projections.reporting as reporting
from housing_projections.models import M0, M0h, M1, M2, M3, M4, M5
from housing_projections.config import DATA_PATH, DEFAULT_SAMPLE_KWARGS, TRACES_DIR

# %% Configuration
N_AREAS = 200
RESULTS_DIR = TRACES_DIR

# Set True to resample, False to load from disk
RESAMPLE = {
    'M0': False,
    'M0h': False, 
    'M1': False,
    'M2': False,
    'M3': False,
    'M4': False,
    'M5': True
}
MODELS_TO_RUN = ['M3', 'M5']

SAMPLE_KWARGS  = {
    **DEFAULT_SAMPLE_KWARGS,
    'draws':  1500,
    'tune':   500,
    'chains': 8,
    'target_accept': 0.95
}

# %% Load data
gdf_raw                 = data_utils.load_data(DATA_PATH)
gdf_clean, _            = outliers.apply_outlier_exclusion(gdf_raw)
gdf_sample              = data_utils.select_spatial_sample(
    gdf_clean,
    n_areas      = N_AREAS,
)

data = data_utils.make_data_dict(gdf_sample)
print(f"n_areas: {data['n_areas']}, n_years: {data['n_years']}")

print(f"Areas:   {data['n_areas']}")
print(f"Years:   {data['n_years']}")
print(f"D mean:  {data['D'].mean():.2f}")
print(f"D range: {data['D'].min():.0f} to {data['D'].max():.0f}")

# %% Instantiate models
model_registry = {
    'M0': M0,
    'M0h': M0h,
    'M1': M1,
    'M2': M2,
    'M3': M3,
    'M4': M4,
    'M5': M5
}

models = {
    name: model_registry[name](data)
    for name in MODELS_TO_RUN
}

# M5 specific — fix lambda weights at M3 posterior means
# to resolve identifiability with alpha_spatial
models['M5'].lambda_weights_fixed = np.array([0.04822521, 0.92909531, 0.00763223, 0.01504725])

for name, model in models.items():
    print(model)

# %% Sample or load
for name, model in models.items():
    trace_path = Path(RESULTS_DIR) / f'{name}.nc'
    if RESAMPLE.get(name, False) or not trace_path.exists():
        print(f"\nSampling {name}: {model.description}")
        model.run(results_dir=RESULTS_DIR, **SAMPLE_KWARGS)
        
    else:
        print(f"\nLoading {name} from {trace_path}")
        model.load(results_dir=RESULTS_DIR)

traces = {name: model.trace for name, model in models.items()}

# %% Full diagnostics, plots and model comparison
post_preds = {}

REPORT_SEED = int(time.time())

for name, model in models.items():
    print(f"\n{'═'*60}")
    print(f" {name} — {model.description}")
    print(f"{'═'*60}")

    prior            = model.prior_predictive(draws=200)
    post_preds[name] = model.posterior_predictive()

    reporting.full_report(
        model.trace, data, post_preds[name],
        prior=prior, model=model, title=name, random_state=REPORT_SEED)

    print(f"\nComputing log likelihood for {name}...")
    with model.model:
        model.trace = pm.compute_log_likelihood(model.trace)

# Update traces after log likelihood computation
traces = {name: model.trace for name, model in models.items()}

# %%
reporting.run_comparison_reports(models, traces, data, post_preds)

# %%
# Model comparison
for name, trace in traces.items():
    p_ll = trace['log_likelihood']['P_like']
    e_ll = trace['log_likelihood']['E_like'].rename({
        'E_like_dim_0': 'P_like_dim_0',
        'E_like_dim_1': 'P_like_dim_1',
    })
    traces[name]['log_likelihood']['combined'] = (p_ll + e_ll).sum(
        dim='P_like_dim_1')

comparison = az.compare(
    {name: trace for name, trace in traces.items()},
    var_name='combined',
)
print("\nModel comparison (LOO-CV):")
print(comparison)

az.plot_compare(comparison)
plt.title('Model comparison — LOO-CV')
plt.tight_layout()
plt.show()

# %%