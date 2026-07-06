"""Tests for housing_projections.diagnostics."""
import pandas as pd

from housing_projections.diagnostics import compute_model_comparison, full_diagnostics


class TestFullDiagnostics:
    def test_returns_all_keys(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        for key in ('rhat', 'divergences', 'calibration', 'census', 'residuals', 'morans_i'):
            assert key in result

    def test_divergences_value(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        assert result['divergences'] == 0


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
