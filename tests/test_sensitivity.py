"""Unit tests for housing_projections.sensitivity — no real traces required."""
import numpy as np

from housing_projections.sensitivity import (
    compute_model_agreement_matrix,
    compute_z_ensemble,
    compute_z_model_sensitivity,
)

# ── Synthetic trace fixture ────────────────────────────────────────────────────

class _FakePosterior:
    """Minimal stand-in for az.InferenceData.posterior."""
    def __init__(self, z_array):
        # z_array: (n_areas, n_years)
        self._z = z_array

    def __contains__(self, item):
        return item == 'z'

    def __getitem__(self, item):
        if item != 'z':
            raise KeyError(item)
        # Wrap in a 4D array: (1 chain, 1 draw, n_areas, n_years)
        return type('ZArray', (), {'values': self._z[None, None, :, :]})()


class _FakeTrace:
    def __init__(self, z_array):
        self.posterior = _FakePosterior(z_array)


def _make_traces(n_areas=20, n_years=10, n_models=3, seed=0):
    rng = np.random.default_rng(seed)
    traces = {}
    for k in range(n_models):
        z = rng.normal(loc=k * 0.5, scale=2.0, size=(n_areas, n_years))
        traces[f'M{k}'] = _FakeTrace(z)
    return traces


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestComputeZModelSensitivity:
    def test_returns_two_outputs(self):
        traces = _make_traces()
        result = compute_z_model_sensitivity(traces)
        assert isinstance(result, tuple) and len(result) == 2

    def test_summary_has_expected_columns(self):
        traces = _make_traces(n_models=3)
        summary, _ = compute_z_model_sensitivity(traces)
        assert 'z_std_across_models' in summary.columns
        assert 'z_range_across_models' in summary.columns
        assert 'z_mean_across_models' in summary.columns

    def test_summary_length_matches_n_areas(self):
        n_areas = 15
        traces = _make_traces(n_areas=n_areas)
        summary, _ = compute_z_model_sensitivity(traces)
        assert len(summary) == n_areas

    def test_long_form_has_expected_columns(self):
        traces = _make_traces()
        _, long_form = compute_z_model_sensitivity(traces)
        for col in ('model', 'lsoa_idx', 'year', 'z_mean'):
            assert col in long_form.columns

    def test_long_form_length(self):
        n_areas, n_years, n_models = 10, 5, 2
        traces = _make_traces(n_areas=n_areas, n_years=n_years, n_models=n_models)
        _, long_form = compute_z_model_sensitivity(traces)
        assert len(long_form) == n_areas * n_years * n_models

    def test_z_std_non_negative(self):
        traces = _make_traces()
        summary, _ = compute_z_model_sensitivity(traces)
        assert (summary['z_std_across_models'] >= 0).all()

    def test_z_range_non_negative(self):
        traces = _make_traces()
        summary, _ = compute_z_model_sensitivity(traces)
        assert (summary['z_range_across_models'] >= 0).all()

    def test_single_model_std_zero(self):
        traces = _make_traces(n_models=1)
        summary, _ = compute_z_model_sensitivity(traces)
        assert (summary['z_std_across_models'] == 0).all()

    def test_each_model_column_present(self):
        traces = _make_traces(n_models=3)
        summary, _ = compute_z_model_sensitivity(traces)
        for name in traces:
            assert f'z_mean_{name}' in summary.columns


class TestComputeModelAgreementMatrix:
    def test_returns_square_dataframe(self):
        traces = _make_traces(n_models=4)
        corr = compute_model_agreement_matrix(traces)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self):
        traces = _make_traces(n_models=3)
        corr = compute_model_agreement_matrix(traces)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self):
        traces = _make_traces(n_models=3)
        corr = compute_model_agreement_matrix(traces)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_values_in_range(self):
        traces = _make_traces(n_models=3)
        corr = compute_model_agreement_matrix(traces)
        assert (corr.values >= -1.0).all() and (corr.values <= 1.0 + 1e-9).all()

    def test_identical_traces_give_high_correlation(self):
        # Use non-constant z so correlation is well-defined
        rng = np.random.default_rng(42)
        z = rng.normal(size=(10, 5))
        traces = {'A': _FakeTrace(z), 'B': _FakeTrace(z)}
        corr = compute_model_agreement_matrix(traces)
        np.testing.assert_allclose(corr.loc['A', 'B'], 1.0, atol=1e-8)


class TestComputeZEnsemble:
    def test_shape_matches_z(self):
        n_areas, n_years = 12, 8
        traces = _make_traces(n_areas=n_areas, n_years=n_years, n_models=3)
        ensemble = compute_z_ensemble(traces)
        assert ensemble.shape == (n_areas, n_years)

    def test_equal_weights_give_mean(self):
        n_areas, n_years = 5, 4
        traces = _make_traces(n_areas=n_areas, n_years=n_years, n_models=2)
        ensemble = compute_z_ensemble(traces)
        z0 = traces['M0'].posterior['z'].values[0, 0]
        z1 = traces['M1'].posterior['z'].values[0, 0]
        expected = (z0 + z1) / 2
        np.testing.assert_allclose(ensemble, expected, atol=1e-10)

    def test_single_model_weight_one(self):
        traces = _make_traces(n_models=1)
        ensemble = compute_z_ensemble(traces)
        z = traces['M0'].posterior['z'].values[0, 0]
        np.testing.assert_allclose(ensemble, z, atol=1e-10)

    def test_with_comparison_df(self):
        import pandas as pd
        traces = _make_traces(n_models=2)
        comp = pd.DataFrame({'weight': [0.8, 0.2]}, index=['M0', 'M1'])
        ensemble = compute_z_ensemble(traces, comparison_df=comp)
        z0 = traces['M0'].posterior['z'].values[0, 0]
        z1 = traces['M1'].posterior['z'].values[0, 0]
        expected = 0.8 * z0 + 0.2 * z1
        np.testing.assert_allclose(ensemble, expected, atol=1e-10)
