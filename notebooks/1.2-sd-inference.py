# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np

from housing_projections.config import DATA_PATH, DEFAULT_SAMPLE_KWARGS, TRACES_DIR
from housing_projections.data import load_data, make_data_dict
from housing_projections.outliers import apply_outlier_exclusion, plot_outlier_areas, plot_outlier_map
from housing_projections.models import M0
from housing_projections.diagnostics import full_diagnostics
from housing_projections.plots import (
    plot_sample_areas,
    plot_posterior_predictive,
    plot_prior_predictive,
    plot_residual_analysis,
    plot_parameter_trace,
)

# %% Configuration
N_AREAS = 100

SAMPLE_KWARGS = {
    **DEFAULT_SAMPLE_KWARGS,
    'draws': 1500,
    'tune':  500,
    'cores': 8,
    'chains': 8,
}

# %% Load data
gdf  = load_data(DATA_PATH)

# %% Load data

# Outlier exclusion
gdf_clean, outlier_df = apply_outlier_exclusion(gdf)

# Inspect flagged areas if needed
plot_outlier_map(gdf, outlier_df)
plot_outlier_areas(gdf, outlier_df, severity='hard')

data = make_data_dict(gdf, n_areas=N_AREAS)

print(f"Areas:   {data['n_areas']}")
print(f"Years:   {data['n_years']}")
print(f"D mean:  {data['D'].mean():.2f}")
print(f"D range: {data['D'].min():.0f} to {data['D'].max():.0f}")

# %% Instantiate model
m0 = M0(data)
print(m0)
m0.graph()

# %% Prior predictive
prior = m0.prior_predictive(draws=200)

z_prior = prior.prior['z'].values
print(f"Prior z mean:  {z_prior.mean():.3f}")
print(f"Prior z sd:    {z_prior.std():.3f}")
print(f"Prior z 99th:  {np.percentile(z_prior, 99):.3f}")
print(f"Prior z 1st:   {np.percentile(z_prior,  1):.3f}")

plot_prior_predictive(prior, data, title='M0')

# %% Sample
m0.run(results_dir=TRACES_DIR, **SAMPLE_KWARGS)

# %% Traces
plot_parameter_trace(m0.trace, m0.var_names, title='M0')

# %% Diagnostics
diags = full_diagnostics(m0.trace, data, model=m0, verbose=True)

# %% Posterior predictive
post_pred = m0.posterior_predictive()
plot_posterior_predictive(post_pred, data, title='M0')

# %% Sample area plots
plot_sample_areas(m0.trace, data, title='M0')

# %% Residual analysis
plot_residual_analysis(m0.trace, data, title='M0')
# %%
