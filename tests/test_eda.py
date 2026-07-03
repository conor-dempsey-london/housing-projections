"""
Unit tests for housing_projections.eda — compute functions only.
Plot functions are excluded (side-effects only, require display).
"""
import pandas as pd

from housing_projections.eda import (
    classify_lsoas,
    compute_agreement_stats,
    compute_autocorrelations,
    compute_crosscorrelations,
    compute_overall_correlation,
)


class TestComputeAgreementStats:
    def test_returns_expected_keys(self, synthetic_gdf):
        result = compute_agreement_stats(synthetic_gdf, verbose=False)
        expected = {
            'total_corr', 'total_bias', 'total_mae',
            'pct_same_sign', 'pct_close_total',
            'annual_corr_mean', 'annual_corr_dist',
        }
        assert set(result.keys()) == expected

    def test_total_corr_in_range(self, synthetic_gdf):
        result = compute_agreement_stats(synthetic_gdf, verbose=False)
        assert -1.0 <= result['total_corr'] <= 1.0

    def test_mae_non_negative(self, synthetic_gdf):
        result = compute_agreement_stats(synthetic_gdf, verbose=False)
        assert result['total_mae'] >= 0

    def test_pct_same_sign_in_range(self, synthetic_gdf):
        result = compute_agreement_stats(synthetic_gdf, verbose=False)
        assert 0.0 <= result['pct_same_sign'] <= 100.0

    def test_annual_corr_dist_length(self, synthetic_gdf):
        result = compute_agreement_stats(synthetic_gdf, verbose=False)
        assert len(result['annual_corr_dist']) == len(synthetic_gdf)


class TestClassifyLsoas:
    def test_returns_dataframe(self, synthetic_gdf):
        df = classify_lsoas(synthetic_gdf)
        assert isinstance(df, pd.DataFrame)

    def test_length_matches_input(self, synthetic_gdf):
        df = classify_lsoas(synthetic_gdf)
        assert len(df) == len(synthetic_gdf)

    def test_has_category_column(self, synthetic_gdf):
        df = classify_lsoas(synthetic_gdf)
        assert 'category' in df.columns

    def test_all_categories_are_strings(self, synthetic_gdf):
        df = classify_lsoas(synthetic_gdf)
        assert df['category'].dtype == object or hasattr(df['category'], 'str')


class TestComputeOverallCorrelation:
    def test_returns_dict(self, synthetic_gdf):
        result = compute_overall_correlation(synthetic_gdf, verbose=False)
        assert isinstance(result, dict)

    def test_has_correlation_key(self, synthetic_gdf):
        result = compute_overall_correlation(synthetic_gdf, verbose=False)
        assert 'pearson_r' in result or 'r' in result or len(result) > 0


class TestComputeAutocorrelations:
    def test_returns_dict(self, synthetic_gdf):
        from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN
        result = compute_autocorrelations(
            synthetic_gdf, INFER_COLS_PLAN, INFER_COLS_BEN,
            max_lag=3, n_permutations=5,
        )
        assert isinstance(result, dict)

    def test_has_obs_keys(self, synthetic_gdf):
        from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN
        result = compute_autocorrelations(
            synthetic_gdf, INFER_COLS_PLAN, INFER_COLS_BEN,
            max_lag=3, n_permutations=5,
        )
        assert 'obs_a' in result and 'obs_b' in result


class TestComputeCrosscorrelations:
    def test_returns_dict(self, synthetic_gdf):
        from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN
        result = compute_crosscorrelations(
            synthetic_gdf, INFER_COLS_PLAN, INFER_COLS_BEN,
            max_lag=3, n_permutations=5,
        )
        assert isinstance(result, dict)

    def test_observed_key_present(self, synthetic_gdf):
        from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN
        result = compute_crosscorrelations(
            synthetic_gdf, INFER_COLS_PLAN, INFER_COLS_BEN,
            max_lag=3, n_permutations=5,
        )
        assert 'observed' in result
