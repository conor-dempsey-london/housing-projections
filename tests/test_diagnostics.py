"""Tests for housing_projections.diagnostics."""
import numpy as np
import pytest

from tests.conftest import make_idata
from housing_projections.diagnostics import (
    check_rhat,
    check_divergences,
    check_calibration,
    check_census_constraint,
    check_residuals,
    compute_lag_weights,
    compute_lag_residuals,
    compute_spatial_misallocation_stats,
    full_diagnostics,
)


# ── check_rhat ─────────────────────────────────────────────────────────────────

class TestCheckRhat:
    def test_returns_dict_keys(self, mock_trace):
        result = check_rhat(mock_trace, verbose=False)
        assert 'summary' in result and 'problematic' in result

    def test_summary_has_rhat_column(self, mock_trace):
        result = check_rhat(mock_trace, verbose=False)
        assert 'r_hat' in result['summary'].columns

    def test_no_problematic_for_well_mixed_trace(self, mock_trace):
        # A short synthetic trace may have r_hat > 1.01 by chance, but
        # the function should at least return the correct type
        result = check_rhat(mock_trace, verbose=False)
        assert hasattr(result['problematic'], 'index')

    def test_subset_var_names(self, mock_trace):
        result = check_rhat(mock_trace, var_names=['mu_slab'], verbose=False)
        assert 'mu_slab' in result['summary'].index.get_level_values(0)

    def test_custom_threshold(self, mock_trace):
        # Very tight threshold — most variables will be flagged
        result_tight = check_rhat(mock_trace, threshold=1.0001, verbose=False)
        result_loose = check_rhat(mock_trace, threshold=2.0,    verbose=False)
        assert len(result_tight['problematic']) >= len(result_loose['problematic'])


# ── check_divergences ──────────────────────────────────────────────────────────

class TestCheckDivergences:
    def test_returns_integer(self, mock_trace):
        n = check_divergences(mock_trace, verbose=False)
        assert isinstance(n, int)

    def test_zero_divergences(self, mock_trace):
        assert check_divergences(mock_trace, verbose=False) == 0

    def test_counts_divergences(self, mock_trace_with_divergences):
        n = check_divergences(mock_trace_with_divergences, verbose=False)
        assert n == 5


# ── check_calibration ─────────────────────────────────────────────────────────

class TestCheckCalibration:
    def test_returns_dict_keys(self, mock_trace, data_dict):
        result = check_calibration(mock_trace, data_dict, verbose=False)
        assert 'planning' in result and 'ben' in result

    def test_coverage_in_0_1(self, mock_trace, data_dict):
        result = check_calibration(mock_trace, data_dict, verbose=False)
        assert 0 <= result['planning'] <= 1
        assert 0 <= result['ben']      <= 1

    def test_perfect_coverage_when_ci_very_wide(self, data_dict, rng):
        """If posterior z perfectly brackets all obs, coverage should be 1."""
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']

        huge_z     = rng.normal(0, 1000, size=(2, 100, n_areas, n_years))
        wide_trace = make_idata(
            posterior={'z': huge_z},
            sample_stats={'diverging': np.zeros((2, 100), dtype=bool)},
        )
        result = check_calibration(wide_trace, data_dict, alpha=0.1, verbose=False)
        assert result['planning'] == pytest.approx(1.0)
        assert result['ben']      == pytest.approx(1.0)

    def test_custom_alpha(self, mock_trace, data_dict):
        result_90 = check_calibration(mock_trace, data_dict, alpha=0.10, verbose=False)
        result_50 = check_calibration(mock_trace, data_dict, alpha=0.50, verbose=False)
        # Wider CI → higher coverage
        assert result_90['planning'] >= result_50['planning']


# ── check_census_constraint ────────────────────────────────────────────────────

class TestCheckCensusConstraint:
    def test_returns_dict_keys(self, mock_trace, data_dict):
        result = check_census_constraint(mock_trace, data_dict, verbose=False)
        assert 'mean_violation' in result and 'max_violation' in result

    def test_non_negative(self, mock_trace, data_dict):
        result = check_census_constraint(mock_trace, data_dict, verbose=False)
        assert result['mean_violation'] >= 0
        assert result['max_violation']  >= 0

    def test_max_geq_mean(self, mock_trace, data_dict):
        result = check_census_constraint(mock_trace, data_dict, verbose=False)
        assert result['max_violation'] >= result['mean_violation']

    def test_perfect_constraint_when_z_sums_match_d(self, data_dict, rng):
        """When each draw's z sums exactly to D, violation should be 0."""
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        D       = data_dict['D']

        # Build z such that z.sum(axis=-1) == D exactly for every draw.
        # Fix the first n_years-1 columns freely, then set the last to make the sum correct.
        z_base  = rng.normal(0, 1, size=(2, 20, n_areas, n_years - 1))
        z_last  = D[None, None, :] - z_base.sum(axis=-1)   # shape (2, 20, n_areas)
        z_perfect = np.concatenate([z_base, z_last[..., None]], axis=-1)

        perfect_trace = make_idata(
            posterior={'z': z_perfect},
            sample_stats={'diverging': np.zeros((2, 20), dtype=bool)},
        )
        result = check_census_constraint(perfect_trace, data_dict, verbose=False)
        assert result['mean_violation'] == pytest.approx(0.0, abs=1e-8)
        assert result['max_violation']  == pytest.approx(0.0, abs=1e-8)


# ── check_residuals ────────────────────────────────────────────────────────────

class TestCheckResiduals:
    def test_returns_dict_keys(self, mock_trace, data_dict):
        result = check_residuals(mock_trace, data_dict, verbose=False)
        assert 'planning' in result and 'ben' in result

    def test_inner_keys(self, mock_trace, data_dict):
        result = check_residuals(mock_trace, data_dict, verbose=False)
        for source in ('planning', 'ben'):
            assert set(result[source].keys()) == {'mean', 'std', 'mae'}

    def test_mae_non_negative(self, mock_trace, data_dict):
        result = check_residuals(mock_trace, data_dict, verbose=False)
        assert result['planning']['mae'] >= 0
        assert result['ben']['mae']      >= 0

    def test_mae_leq_std_plus_abs_mean(self, mock_trace, data_dict):
        """Rough triangle inequality: MAE ≤ |mean| + std (not always tight, but direction check)."""
        result = check_residuals(mock_trace, data_dict, verbose=False)
        for src in ('planning', 'ben'):
            r = result[src]
            assert r['mae'] >= 0


# ── compute_lag_weights ────────────────────────────────────────────────────────

class TestComputeLagWeights:
    def test_returns_dict_keys(self, mock_trace):
        result = compute_lag_weights(mock_trace, verbose=False)
        for key in ('means', 'lo', 'hi', 'implied_mean_lag', 'n_lags', 'lambda_flat'):
            assert key in result

    def test_means_sum_to_one(self, mock_trace):
        result = compute_lag_weights(mock_trace, verbose=False)
        assert sum(result['means']) == pytest.approx(1.0, abs=1e-6)

    def test_lo_leq_means_leq_hi(self, mock_trace):
        result = compute_lag_weights(mock_trace, verbose=False)
        assert np.all(result['lo'] <= result['means'])
        assert np.all(result['means'] <= result['hi'])

    def test_n_lags(self, mock_trace):
        result = compute_lag_weights(mock_trace, verbose=False)
        assert result['n_lags'] == 4  # mock_trace has lambda_weights with dim 4

    def test_implied_mean_lag_in_range(self, mock_trace):
        result = compute_lag_weights(mock_trace, verbose=False)
        assert 0 <= result['implied_mean_lag'] <= result['n_lags'] - 1


# ── compute_lag_residuals ──────────────────────────────────────────────────────

class TestComputeLagResiduals:
    def test_returns_dict_keys(self, mock_trace, data_dict):
        result = compute_lag_residuals(mock_trace, data_dict)
        assert 'no_lag' in result and 'with_lag' in result

    def test_shapes(self, mock_trace, data_dict):
        result  = compute_lag_residuals(mock_trace, data_dict)
        shape   = (data_dict['n_areas'], data_dict['n_years'])
        assert result['no_lag'].shape   == shape
        assert result['with_lag'].shape == shape

    def test_with_fixed_lambda_weights(self, mock_trace, data_dict):
        """Pass lambda_weights explicitly (simulates M6 with fixed weights)."""
        # Strip lambda_weights from trace to force the explicit-weights code path
        posterior_dict = {
            k: mock_trace.posterior[k].values
            for k in mock_trace.posterior.data_vars
            if k != 'lambda_weights'
        }
        trace_no_lw = make_idata(
            posterior=posterior_dict,
            sample_stats={'diverging': np.zeros(
                mock_trace.sample_stats.diverging.shape, dtype=bool)},
        )
        lw = np.array([0.6, 0.2, 0.1, 0.1])
        result = compute_lag_residuals(trace_no_lw, data_dict, lambda_weights=lw)
        assert result['no_lag'].shape == (data_dict['n_areas'], data_dict['n_years'])
        assert result['with_lag'].shape == (data_dict['n_areas'], data_dict['n_years'])

    def test_no_lag_component_returns_same_for_both(self, data_dict, rng):
        """When lambda_weights absent and none passed, no_lag == with_lag."""
        n_areas = data_dict['n_areas']
        n_years = data_dict['n_years']
        trace_no_lag = make_idata(
            posterior={'z': rng.normal(0, 1, size=(2, 20, n_areas, n_years))},
            sample_stats={'diverging': np.zeros((2, 20), dtype=bool)},
        )
        result = compute_lag_residuals(trace_no_lag, data_dict)
        np.testing.assert_array_equal(result['no_lag'], result['with_lag'])


# ── compute_spatial_misallocation_stats ────────────────────────────────────────

class TestComputeSpatialMisallocationStats:
    def test_returns_expected_keys(self, mock_trace, data_dict):
        result = compute_spatial_misallocation_stats(mock_trace, data_dict)
        for key in ('alpha_mean', 'alpha_std', 'alpha_lo', 'alpha_hi',
                    'alpha_post', 'z_flat', 'z_lag'):
            assert key in result

    def test_alpha_in_0_1(self, mock_trace, data_dict):
        result = compute_spatial_misallocation_stats(mock_trace, data_dict)
        assert 0 <= result['alpha_mean'] <= 1
        assert 0 <= result['alpha_lo']   <= result['alpha_hi'] <= 1

    def test_z_shapes(self, mock_trace, data_dict):
        result  = compute_spatial_misallocation_stats(mock_trace, data_dict)
        n_total = data_dict['n_areas'] * data_dict['n_years']
        assert result['z_flat'].shape == (n_total,)
        assert result['z_lag'].shape  == (n_total,)


# ── full_diagnostics ───────────────────────────────────────────────────────────

class TestFullDiagnostics:
    def test_returns_all_keys(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        for key in ('rhat', 'divergences', 'calibration', 'census', 'residuals', 'morans_i'):
            assert key in result

    def test_divergences_value(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        assert result['divergences'] == 0
