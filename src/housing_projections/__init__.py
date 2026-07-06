from housing_projections.data import (
    load_data,
    make_data_dict,
    select_spatial_sample,
    validate_data_path,
)
from housing_projections.diagnostics import compute_model_comparison, full_diagnostics
from housing_projections.outliers import (
    apply_outlier_exclusion,
    plot_hard_outlier_areas,
    plot_outlier_areas,
    plot_outlier_map,
    plot_soft_outlier_areas,
)
from housing_projections.plots import (
    plot_parameter_trace,
    plot_posterior_predictive,
    plot_prior_predictive,
    plot_residual_analysis,
    plot_sample_areas,
    plot_spatial_diagnostics,
)
from housing_projections.reporting import full_report, run_comparison_reports
from housing_projections.sensitivity import (
    compute_model_agreement_matrix,
    compute_z_ensemble,
    compute_z_model_sensitivity,
    plot_model_agreement_matrix,
    plot_sensitivity_vs_disagreement,
    plot_z_range_distribution,
    plot_z_sensitivity_map,
)
from housing_projections.spatial import (
    SpatialLagTransformer,
    add_spatial_lag_features,
    build_spatial_weights,
    build_weights_libpysal,
    compute_morans_i,
    compute_morans_i_by_year,
)

__all__ = [
    # data
    "load_data",
    "make_data_dict",
    "select_spatial_sample",
    "validate_data_path",
    # diagnostics
    "full_diagnostics",
    "compute_model_comparison",
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
    # sensitivity
    "compute_z_model_sensitivity",
    "compute_model_agreement_matrix",
    "compute_z_ensemble",
    "plot_z_sensitivity_map",
    "plot_model_agreement_matrix",
    "plot_z_range_distribution",
    "plot_sensitivity_vs_disagreement",
    # spatial
    "build_weights_libpysal",
    "build_spatial_weights",
    "compute_morans_i",
    "compute_morans_i_by_year",
    "add_spatial_lag_features",
    "SpatialLagTransformer",
]
