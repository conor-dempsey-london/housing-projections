"""
One-off comparison of AZ3's full-scale domain-fitness metrics between the
production PLD cut and the corrected re-extraction, with outlier exclusion run
INDEPENDENTLY per input (fixing the gap recorded in DATA_MANIFEST.md's
2026-08-03 full-scale entry, where both runs were forced onto the production
trace's frozen 4987-LSOA list instead of each cut's own outlier screening).

Reports metrics on:
  - each input's own independently-screened 4987-area universe
  - the ~4977-area INTERSECTION of both (excludes the ~10 LSOAs that are a
    hard outlier under one cut but not the other), for a strict apples-to-
    apples comparison unaffected by which side's outlier screening differs.

Usage
-----
    pixi run python scripts/compare_pld_fullscale.py
"""
import sys
from pathlib import Path

import arviz as az
import numpy as np


class _PosteriorOnly:
    """Minimal trace-like wrapper: the diagnostics functions used here only
    ever touch `.posterior`, so a `.sel(area=...)`'d posterior Dataset is
    enough to reuse them on an area subset without reconstructing a full
    InferenceData object."""
    def __init__(self, posterior):
        self.posterior = posterior

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from housing_projections.data import load_data, make_data_dict  # noqa: E402
from housing_projections.diagnostics import (  # noqa: E402
    _check_calibration,
    _check_morans_i,
    z_flatness_summary,
)
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402


def load_trace_no_warmup(trace_path):
    return az.from_netcdf(trace_path)


def data_matching_codes(gdf, lsoa_codes):
    subset = gdf[gdf['LSOA21CD'].isin(lsoa_codes)].copy()
    subset = subset.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
    return make_data_dict(subset)


def report_metrics(label, trace, data):
    flat = z_flatness_summary(trace, data)
    cov = _check_calibration(trace, data)
    morans = _check_morans_i(trace, data)

    z_post = trace.posterior['z'].values
    z_mean = z_post.mean(axis=(0, 1))
    resid_plan = data['P_obs'] - z_mean
    resid_uprn = data['E_obs'] - z_mean

    print(f'\n-- {label} (n_areas={data["n_areas"]}) --')
    print(f'  frac_flat_despite_active: {flat.attrs["summary"]["frac_flat_despite_active"]:.4f}')
    print(f'  plan coverage (90%):      {cov["planning"]:.4f}')
    print(f'  uprn coverage (90%):      {cov["uprn"]:.4f}')
    print(f'  planning residual std/MAE: {resid_plan.std():.3f} / {np.abs(resid_plan).mean():.3f}')
    print(f'  uprn residual std/MAE:     {resid_uprn.std():.3f} / {np.abs(resid_uprn).mean():.3f}')
    print(f"  Moran's I (planning resid): I={morans['planning']['I']:.4f} p={morans['planning']['p_value']:.4f}")
    print(f"  Moran's I (uprn resid):     I={morans['uprn']['I']:.4f} p={morans['uprn']['p_value']:.4f}")

    return {
        'frac_flat_despite_active': flat.attrs['summary']['frac_flat_despite_active'],
        'plan_cov_90': cov['planning'],
        'uprn_cov_90': cov['uprn'],
        'plan_resid_std': float(resid_plan.std()),
        'plan_resid_mae': float(np.abs(resid_plan).mean()),
        'uprn_resid_std': float(resid_uprn.std()),
        'uprn_resid_mae': float(np.abs(resid_uprn).mean()),
        'morans_i_plan': morans['planning']['I'],
        'morans_i_plan_p': morans['planning']['p_value'],
        'morans_i_uprn': morans['uprn']['I'],
        'morans_i_uprn_p': morans['uprn']['p_value'],
    }


def main():
    print('-- Loading traces (lazy, z/coords only touched) --')
    prod_trace = load_trace_no_warmup('results/traces_full/AZ3.nc')
    corr_trace = load_trace_no_warmup('results/traces_full_corrected_v2/AZ3.nc')

    prod_codes = set(prod_trace.posterior['z'].coords['area'].values.tolist())
    corr_codes = set(corr_trace.posterior['z'].coords['area'].values.tolist())
    common_codes = prod_codes & corr_codes
    print(f'  production universe: {len(prod_codes)}  corrected universe: {len(corr_codes)}  '
          f'common: {len(common_codes)}  (differ: {len(prod_codes ^ corr_codes)})')

    print('\n-- Loading gdf per input, matched to each trace\'s own universe --')
    gdf_prod = load_data('data', pld_filename='lsoa_completions_time_series_pivot.csv')
    gdf_corr = load_data('data', pld_filename='lsoa_completions_time_series_pivot_unit_level.csv')

    # own-universe metrics
    data_prod_own = data_matching_codes(gdf_prod, prod_trace.posterior['z'].coords['area'].values.tolist())
    data_corr_own = data_matching_codes(gdf_corr, corr_trace.posterior['z'].coords['area'].values.tolist())
    m_prod_own = report_metrics('PRODUCTION, own 4987-area universe', prod_trace, data_prod_own)
    m_corr_own = report_metrics('CORRECTED, own 4987-area universe', corr_trace, data_corr_own)

    # common-intersection metrics (strict apples-to-apples)
    common_order = [c for c in prod_trace.posterior['z'].coords['area'].values.tolist() if c in common_codes]
    data_prod_common = data_matching_codes(gdf_prod, common_order)
    data_corr_common = data_matching_codes(gdf_corr, common_order)
    # subset each trace's z posterior to the common area order too
    prod_trace_common = _PosteriorOnly(prod_trace.posterior.sel(area=common_order))
    corr_trace_common = _PosteriorOnly(corr_trace.posterior.sel(area=common_order))
    m_prod_common = report_metrics(f'PRODUCTION, common {len(common_order)}-area intersection',
                                    prod_trace_common, data_prod_common)
    m_corr_common = report_metrics(f'CORRECTED, common {len(common_order)}-area intersection',
                                    corr_trace_common, data_corr_common)

    print('\n-- Summary (common intersection, production -> corrected) --')
    for k in m_prod_common:
        print(f'  {k}: {m_prod_common[k]:.4f} -> {m_corr_common[k]:.4f}')


if __name__ == '__main__':
    main()
