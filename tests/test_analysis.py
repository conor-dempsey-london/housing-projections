"""
Smoke tests for housing_projections.analysis — public API only.
Uses synthetic traces and data from conftest; no network/file I/O.
"""
import arviz as az
import numpy as np
import pandas as pd
import pytest

from housing_projections.analysis import (
    compute_lag_residuals,
    compute_lag_weights,
    compute_model_comparison,
    compute_spatial_misallocation_stats,
    uncertainty_by_geography,
    variance_components,
)
from tests.conftest import N_CHAINS, N_DRAWS


class TestComputeModelComparison:
    def test_returns_comparison_dataframe(self, mock_traces_with_ll):
        result = compute_model_comparison(mock_traces_with_ll, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert set(result.index) == {'MA', 'MB'}

    def test_scores_joint_pe_when_both_present(self, mock_traces_with_pe_ll):
        result = compute_model_comparison(mock_traces_with_pe_ll, verbose=False)
        # MB's much worse E_like should make it rank behind MA despite
        # identical P_like — confirms E is actually scored, not ignored.
        assert result.index[0] == 'MA'


class TestComputeLagWeights:
    def test_returns_expected_keys(self, mock_trace):
        result = compute_lag_weights(mock_trace)
        assert {'means', 'lo', 'hi', 'implied_mean_lag', 'n_lags', 'lambda_flat'} <= set(result)

    def test_means_sum_to_one(self, mock_trace):
        # lambda_weights is a Dirichlet-style simplex in the fixture
        result = compute_lag_weights(mock_trace)
        assert result['means'].sum() == pytest.approx(1.0, abs=1e-6)


class TestComputeLagResiduals:
    def test_returns_no_lag_and_with_lag(self, mock_trace, data_dict):
        result = compute_lag_residuals(mock_trace, data_dict)
        assert set(result) == {'no_lag', 'with_lag'}
        assert result['no_lag'].shape == (data_dict['n_areas'], data_dict['n_years'])

    def test_falls_back_to_plain_without_lambda_weights(self, mock_trace_with_divergences, data_dict):
        # mock_trace_with_divergences has no lambda_weights and no override passed
        result = compute_lag_residuals(mock_trace_with_divergences, data_dict)
        np.testing.assert_array_equal(result['no_lag'], result['with_lag'])


class TestComputeSpatialMisallocationStats:
    def test_returns_expected_keys(self, mock_trace, data_dict):
        result = compute_spatial_misallocation_stats(mock_trace, data_dict)
        assert {'alpha_mean', 'alpha_std', 'alpha_lo', 'alpha_hi',
                'alpha_post', 'z_flat', 'z_lag'} <= set(result)
        assert 0 <= result['alpha_mean'] <= 1


class TestUncertaintyByGeography:
    @pytest.fixture
    def lsoa_codes(self, data_dict):
        return data_dict['gdf']['LSOA21CD'].tolist()

    @pytest.fixture(autouse=True)
    def _patch_geography_lookup(self, monkeypatch, lsoa_codes):
        # 9 synthetic LSOAs -> 3 MSOAs (3 each) -> 2 boroughs, avoiding the
        # real gla_data._ons network/file lookup.
        lookup = pd.DataFrame({
            'LSOA21CD': lsoa_codes,
            'MSOA21CD': [f'MSOA{i // 3}' for i in range(len(lsoa_codes))],
            'LAD22NM':  ['BoroughA' if i < 6 else 'BoroughB' for i in range(len(lsoa_codes))],
        })
        monkeypatch.setattr(
            'gla_data._ons.fetch_geography_lookup', lambda *a, **k: lookup)

    def test_returns_expected_levels(self, mock_trace, lsoa_codes):
        result = uncertainty_by_geography(mock_trace, lsoa_codes=lsoa_codes)
        assert set(result) == {'lsoa', 'msoa', 'borough', 'summary'}
        assert len(result['lsoa']) == len(lsoa_codes)
        assert len(result['msoa']) == 3
        assert len(result['borough']) == 2

    def test_summary_has_one_row_per_level(self, mock_trace, lsoa_codes):
        result = uncertainty_by_geography(mock_trace, lsoa_codes=lsoa_codes)
        assert list(result['summary'].index) == ['LSOA', 'MSOA', 'Borough']


class TestVarianceComponents:
    def test_empty_ratio_when_sigma_mu_absent(self, mock_trace_with_divergences):
        # fixture has sigma_slab but no sigma_mu
        result = variance_components(mock_trace_with_divergences)
        assert 'sigma_slab' in result
        assert 'ratio' not in result

    def test_computes_ratio_when_both_present(self, rng):
        trace = az.from_dict({'posterior': {
            'sigma_mu':   np.abs(rng.normal(2.0, 0.2, size=(N_CHAINS, N_DRAWS))),
            'sigma_slab': np.abs(rng.normal(5.0, 0.2, size=(N_CHAINS, N_DRAWS))),
        }})
        result = variance_components(trace)
        assert set(result) == {'sigma_mu', 'sigma_slab', 'ratio'}
        assert result['ratio'] == pytest.approx(
            result['sigma_mu']['mean'] / result['sigma_slab']['mean'])
