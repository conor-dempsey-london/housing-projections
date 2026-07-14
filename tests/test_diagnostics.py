"""Tests for housing_projections.diagnostics."""
import arviz as az
import numpy as np
import pandas as pd

from housing_projections.analysis import compute_model_comparison
from housing_projections.diagnostics import (
    _check_sigma_slab_vs_disagreement,
    full_diagnostics,
    z_flatness_summary,
)


class TestFullDiagnostics:
    def test_returns_all_keys(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        for key in ('rhat', 'divergences', 'calibration', 'census', 'residuals', 'morans_i'):
            assert key in result

    def test_divergences_value(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        assert result['divergences'] == 0

    def test_omits_sigma_slab_check_for_non_m9_trace(self, mock_trace, data_dict):
        # mock_trace has scalar sigma_slab and no lambda_weights_P/E (M0h/M1h
        # shape) — the M9-specific check should be skipped, not error.
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        assert 'sigma_slab_vs_disagreement' not in result

    def test_includes_sigma_slab_check_for_m9_trace(self, mock_trace_m9, data_dict):
        result = full_diagnostics(mock_trace_m9, data_dict, verbose=False)
        assert 'sigma_slab_vs_disagreement' in result


class TestZFlatnessSummary:
    def _make_case(self):
        # 2 areas, 3 years, 1 chain, 4 draws.
        z = np.zeros((1, 4, 2, 3))
        z[:, :, 0, :] = 5.0                 # area 0: perfectly flat
        z[:, :, 1, :] = [0.0, 10.0, 20.0]    # area 1: clearly varies

        trace = az.from_dict(
            {'posterior': {'z': z}},
            coords={'area': ['A0', 'A1'], 'year': [2020, 2021, 2022]},
            dims={'z': ['area', 'year']})

        data = {
            'P_obs': np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]),  # area 0 active in year 1
            'E_obs': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
        return trace, data

    def test_flags_flat_despite_active(self):
        trace, data = self._make_case()
        df = z_flatness_summary(trace, data, active_threshold=3.0, flat_range_threshold=2.0)
        row0 = df[df['area'] == 'A0'].iloc[0]
        assert row0['has_active_year']
        assert row0['is_flat']
        assert row0['flat_despite_active']

    def test_does_not_flag_area_that_moves(self):
        trace, data = self._make_case()
        df = z_flatness_summary(trace, data, active_threshold=3.0, flat_range_threshold=2.0)
        row1 = df[df['area'] == 'A1'].iloc[0]
        assert not row1['is_flat']
        assert not row1['flat_despite_active']

    def test_summary_attrs_present(self):
        trace, data = self._make_case()
        df = z_flatness_summary(trace, data)
        for key in ('frac_flat', 'frac_active', 'frac_flat_despite_active'):
            assert key in df.attrs['summary']
        assert df.attrs['summary']['frac_flat_despite_active'] == 0.5


class TestCheckSigmaSlabVsDisagreement:
    def test_returns_none_without_lambda_weights_p_e(self, mock_trace, data_dict):
        assert _check_sigma_slab_vs_disagreement(mock_trace, data_dict) is None

    def test_returns_dict_with_expected_keys(self, mock_trace_m9, data_dict):
        result = _check_sigma_slab_vs_disagreement(mock_trace_m9, data_dict)
        assert result is not None
        for key in ('mean_lag_P', 'mean_lag_E', 'n_areas_valid',
                    'corr_agreement_vs_sigma_slab',
                    'corr_agreement_vs_sigma_slab_scale_controlled'):
            assert key in result

    def test_mean_lag_within_window(self, mock_trace_m9, data_dict):
        result = _check_sigma_slab_vs_disagreement(mock_trace_m9, data_dict)
        assert 0 <= result['mean_lag_P'] <= 3
        assert 0 <= result['mean_lag_E'] <= 3


class TestComputeModelComparison:
    def test_returns_dataframe(self, mock_traces_with_ll):
        result = compute_model_comparison(mock_traces_with_ll, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_model_names(self, mock_traces_with_ll):
        result = compute_model_comparison(mock_traces_with_ll, verbose=False)
        assert set(result.index) == set(mock_traces_with_ll.keys())

    def test_has_expected_columns(self, mock_traces_with_ll):
        result = compute_model_comparison(mock_traces_with_ll, verbose=False)
        for col in ('elpd', 'p', 'weight'):
            assert col in result.columns

    def test_scores_e_like_jointly_with_p_like(self, mock_traces_with_pe_ll):
        # Regression test for the hardcoded var_name='P_like' bug (see
        # model-evaluation-methods.md): MA/MB share identical P_like but MB's
        # E_like is far worse -- if E were being ignored (the old bug), MA
        # and MB would score identically; scoring the joint P+E likelihood
        # must rank MA strictly ahead of MB.
        result = compute_model_comparison(mock_traces_with_pe_ll, verbose=False)
        assert result.loc['MA', 'elpd'] > result.loc['MB', 'elpd']
