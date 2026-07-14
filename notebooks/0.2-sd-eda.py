# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import housing_projections.data as data_utils
import housing_projections.outliers as outliers
import housing_projections.eda.comparison as comparison
import housing_projections.eda.timeseries as timeseries
import housing_projections.eda.agreement as agreement
import housing_projections.eda.spatial_plots as spatial
from housing_projections.config import DATA_PATH, INFER_COLS_BEN, INFER_COLS_PLAN

# %% Load data
gdf = data_utils.load_data(DATA_PATH)

# %% Outlier exclusion
gdf_clean, outlier_df = outliers.apply_outlier_exclusion(gdf)
outliers.plot_outlier_map(gdf, outlier_df)
outliers.plot_outlier_areas(gdf, outlier_df, severity='hard')
outliers.plot_outlier_areas(gdf, outlier_df, severity='soft')

# %% Census stock overview
comparison.plot_census_stocks(gdf_clean)
comparison.plot_stock_scatter(gdf_clean)

# %% Census stock spatial analysis
spatial.plot_census_stock_maps(gdf_clean)
spatial.plot_intercensal_change_histogram_map(gdf_clean)
spatial.plot_change_hotspots(gdf_clean)
spatial.plot_spatial_autocorrelation_change(gdf_clean)

# %% Annual P vs E
comparison.plot_annual_p_vs_e(gdf_clean)

# %% Cumulative flow vs intercensal change
comparison.plot_cumulative_vs_intercensal(
    gdf_clean,
    cols=['total_change_2011_to_2021_ben', 'intercensal_completions'],
    labels=['BEN', 'Completions'],
)

# %% Overall P vs E correlation
comparison.compute_overall_correlation(gdf_clean, verbose=True)
comparison.plot_per_area_correlation(gdf_clean)

# %% Time series distributions
timeseries.plot_distributions_by_year(gdf_clean)
timeseries.plot_mean_trends(gdf_clean)
timeseries.plot_year_correlation(gdf_clean)

# %% Autocorrelation
ac_results = timeseries.compute_autocorrelations(
    gdf_clean, INFER_COLS_PLAN, INFER_COLS_BEN)
timeseries.plot_autocorrelations(ac_results, labels=('Planning', 'BEN'))

# %% Cross-correlation (raw)
xc_raw = timeseries.compute_crosscorrelations(
    gdf_clean, INFER_COLS_PLAN, INFER_COLS_BEN)
timeseries.plot_crosscorrelations(xc_raw, labels=('Planning', 'BEN'))

# %% Cross-correlation (prewhitened)
xc_prewhitened = timeseries.compute_crosscorrelations_prewhitened(
    gdf_clean, INFER_COLS_PLAN, INFER_COLS_BEN, method='difference')
timeseries.plot_crosscorrelations(
    xc_prewhitened,
    labels=('Planning (differenced)', 'BEN (differenced)'),
)

# %% Agreement analysis
results = agreement.full_agreement_analysis(gdf_clean)

# %% Spatial analysis
spatial.plot_intercensal_change_map(gdf_clean)
spatial.plot_mean_change_maps(gdf_clean)
spatial.plot_source_disagreement_map(gdf_clean)
spatial.plot_morans_i_by_year(gdf_clean)


# %%
