from .agreement import (
    classify_lsoas,
    compute_agreement_stats,
    full_agreement_analysis,
    plot_category_breakdown,
    plot_category_examples,
    plot_lag_candidates,
    plot_sign_disagreements,
    plot_total_agreement,
)
from .comparison import (
    compute_overall_correlation,
    plot_annual_p_vs_e,
    plot_census_stocks,
    plot_cumulative_vs_intercensal,
    plot_per_area_correlation,
    plot_stock_scatter,
)
from .spatial_plots import (
    plot_census_stock_maps,
    plot_change_hotspots,
    plot_intercensal_change_histogram_map,
    plot_intercensal_change_map,
    plot_mean_change_maps,
    plot_morans_i_by_year,
    plot_source_disagreement_map,
    plot_spatial_autocorrelation_change,
    plot_spatial_distribution,
)
from .timeseries import (
    compute_autocorrelations,
    compute_crosscorrelations,
    compute_crosscorrelations_prewhitened,
    plot_autocorrelations,
    plot_crosscorrelations,
    plot_distributions_by_year,
    plot_mean_trends,
    plot_year_correlation,
)

__all__ = [
    # comparison
    "plot_census_stocks",
    "plot_stock_scatter",
    "plot_cumulative_vs_intercensal",
    "compute_overall_correlation",
    "plot_per_area_correlation",
    "plot_annual_p_vs_e",
    # timeseries
    "plot_distributions_by_year",
    "plot_mean_trends",
    "plot_year_correlation",
    "compute_autocorrelations",
    "compute_crosscorrelations",
    "compute_crosscorrelations_prewhitened",
    "plot_autocorrelations",
    "plot_crosscorrelations",
    # agreement
    "compute_agreement_stats",
    "classify_lsoas",
    "full_agreement_analysis",
    "plot_total_agreement",
    "plot_category_breakdown",
    "plot_category_examples",
    "plot_lag_candidates",
    "plot_sign_disagreements",
    # spatial
    "plot_morans_i_by_year",
    "plot_spatial_distribution",
    "plot_mean_change_maps",
    "plot_source_disagreement_map",
    "plot_intercensal_change_map",
    "plot_census_stock_maps",
    "plot_intercensal_change_histogram_map",
    "plot_change_hotspots",
    "plot_spatial_autocorrelation_change",
]
