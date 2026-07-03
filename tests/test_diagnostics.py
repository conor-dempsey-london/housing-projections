"""Tests for housing_projections.diagnostics."""
from housing_projections.diagnostics import full_diagnostics


class TestFullDiagnostics:
    def test_returns_all_keys(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        for key in ('rhat', 'divergences', 'calibration', 'census', 'residuals', 'morans_i'):
            assert key in result

    def test_divergences_value(self, mock_trace, data_dict):
        result = full_diagnostics(mock_trace, data_dict, verbose=False)
        assert result['divergences'] == 0
