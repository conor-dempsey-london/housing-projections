from housing_projections.data import load_data, make_data_dict, select_spatial_sample
from housing_projections.diagnostics import full_diagnostics
from housing_projections.outliers import (
    apply_outlier_exclusion,
    plot_outlier_areas,
    plot_hard_outlier_areas,
    plot_soft_outlier_areas,
    plot_outlier_map,
)
from housing_projections.plots import (
    plot_sample_areas,
    plot_posterior_predictive,
    plot_prior_predictive,
    plot_residual_analysis,
    plot_parameter_trace,
    plot_spatial_diagnostics,
)
from housing_projections.reporting import full_report, run_comparison_reports
from housing_projections.spatial import (
    build_weights_libpysal,
    build_spatial_weights,
    compute_morans_i,
    compute_morans_i_by_year,
    add_spatial_lag_features,
    SpatialLagTransformer,
)

__all__ = [
    # data
    "load_data",
    "make_data_dict",
    "select_spatial_sample",
    # diagnostics
    "full_diagnostics",
    # outliers
    "apply_outlier_exclusion",
    "plot_outlier_areas",
    "plot_hard_outlier_areas",
    "plot_soft_outlier_areas",
    "plot_outlier_map",
    # plots
    "plot_sample_areas",
    "plot_posterior_predictive",
    "plot_prior_predictive",
    "plot_residual_analysis",
    "plot_parameter_trace",
    "plot_spatial_diagnostics",
    # reporting
    "full_report",
    "run_comparison_reports",
    # spatial
    "build_weights_libpysal",
    "build_spatial_weights",
    "compute_morans_i",
    "compute_morans_i_by_year",
    "add_spatial_lag_features",
    "SpatialLagTransformer",
]
