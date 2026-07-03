"""
Smoke tests for housing_projections.reporting — public API only.
Uses synthetic trace and data from conftest; no matplotlib display.
"""
import numpy as np
import arviz as az
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so plt.show() is a no-op

from housing_projections.reporting import full_report, run_comparison_reports
from tests.conftest import N_CHAINS, N_DRAWS


@pytest.fixture(scope='module')
def mock_post_pred(data_dict):
    """Minimal post_pred InferenceData with P_like and E_like in posterior_predictive."""
    n_areas = data_dict['n_areas']
    n_years = data_dict['n_years']
    rng = np.random.default_rng(0)
    return az.from_dict({
        'posterior_predictive': {
            'P_like': rng.normal(1.0, 2.0, size=(N_CHAINS, N_DRAWS, n_areas, n_years)),
            'E_like': rng.normal(1.0, 2.0, size=(N_CHAINS, N_DRAWS, n_areas, n_years)),
        }
    })


class TestFullReport:
    """Smoke test: full_report should run without raising for each model tier."""

    def test_runs_without_model(self, mock_trace, data_dict, mock_post_pred):
        """No model argument — skips model-specific diagnostics and trace plot."""
        full_report(
            trace=mock_trace,
            data=data_dict,
            post_pred=mock_post_pred,
            model=None,
        )

    def test_runs_with_model_stub(self, mock_trace, data_dict, mock_post_pred):
        """With a minimal model stub that provides .name and .var_names."""
        class _ModelStub:
            name = 'M0'
            var_names = ['mu_slab', 'sigma_slab']

        full_report(
            trace=mock_trace,
            data=data_dict,
            post_pred=mock_post_pred,
            model=_ModelStub(),
        )

    def test_runs_m3_registry(self, mock_trace, data_dict, mock_post_pred):
        """M3 model dispatches to _plot_lag_diagnostics via MODEL_DIAGNOSTICS."""
        class _M3Stub:
            name = 'M3'
            var_names = ['mu_slab', 'sigma_slab']
            lambda_weights = None

        full_report(
            trace=mock_trace,
            data=data_dict,
            post_pred=mock_post_pred,
            model=_M3Stub(),
        )


class TestRunComparisonReports:
    def test_no_common_models_runs_silently(self, mock_trace, data_dict):
        """When no model pair from the registry is present, nothing runs — no error."""
        run_comparison_reports(
            models={'M0': None},
            traces={'M0': mock_trace},
            data=data_dict,
            post_preds={'M0': mock_trace},
        )
