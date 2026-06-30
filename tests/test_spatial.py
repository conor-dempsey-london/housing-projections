"""Tests for housing_projections.spatial."""
import numpy as np
import pytest

from housing_projections.spatial import (
    build_weights_libpysal,
    build_spatial_weights,
    weights_to_dense,
    add_spatial_lag_features,
    morans_i,
    morans_i_by_year,
    SpatialLagTransformer,
)
from housing_projections.config import INFER_COLS_PLAN


class TestBuildWeightsLibpysal:
    def test_returns_weights_object(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        import libpysal
        assert isinstance(w, libpysal.weights.weights.W)

    def test_transform_is_row_normalised(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        assert w.transform.lower() == 'r'

    def test_n_equals_n_areas(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        assert w.n == len(synthetic_gdf)

    def test_centre_has_more_neighbours_than_corner(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        # In a 3×3 grid, the centre cell (index 4) is adjacent to all 8 others;
        # a corner cell (index 0) is adjacent to at most 3.
        assert len(w.neighbors[4]) > len(w.neighbors[0])


class TestBuildSpatialWeights:
    def test_returns_numpy_array(self, synthetic_gdf):
        W = build_spatial_weights(synthetic_gdf)
        assert isinstance(W, np.ndarray)

    def test_shape(self, synthetic_gdf):
        n = len(synthetic_gdf)
        W = build_spatial_weights(synthetic_gdf)
        assert W.shape == (n, n)

    def test_diagonal_is_zero(self, synthetic_gdf):
        W = build_spatial_weights(synthetic_gdf)
        np.testing.assert_array_equal(np.diag(W), 0)

    def test_rows_sum_to_one_or_zero(self, synthetic_gdf):
        W = build_spatial_weights(synthetic_gdf)
        row_sums = W.sum(axis=1)
        # Each area has at least one neighbour in a connected grid
        assert np.all((np.abs(row_sums - 1.0) < 1e-10) | (row_sums == 0))

    def test_non_negative(self, synthetic_gdf):
        W = build_spatial_weights(synthetic_gdf)
        assert np.all(W >= 0)


class TestWeightsToDense:
    def test_matches_build_spatial_weights(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        W_dense_direct = build_spatial_weights(synthetic_gdf)
        W_via_convert  = weights_to_dense(w)
        np.testing.assert_allclose(W_via_convert, W_dense_direct, atol=1e-12)

    def test_shape(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        W = weights_to_dense(w)
        assert W.shape == (len(synthetic_gdf), len(synthetic_gdf))


class TestAddSpatialLagFeatures:
    def test_adds_lag_columns(self, synthetic_gdf):
        col = INFER_COLS_PLAN[0]
        out = add_spatial_lag_features(synthetic_gdf, [col])
        assert f'lag_{col}' in out.columns

    def test_original_columns_preserved(self, synthetic_gdf):
        col = INFER_COLS_PLAN[0]
        out = add_spatial_lag_features(synthetic_gdf, [col])
        assert col in out.columns

    def test_lag_values_are_weighted_averages(self, synthetic_gdf):
        col = INFER_COLS_PLAN[0]
        out = add_spatial_lag_features(synthetic_gdf, [col])
        W   = build_spatial_weights(synthetic_gdf)
        expected = W @ synthetic_gdf[col].values
        np.testing.assert_allclose(out[f'lag_{col}'].values, expected, atol=1e-10)

    def test_multiple_columns(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:3]
        out  = add_spatial_lag_features(synthetic_gdf, cols)
        for col in cols:
            assert f'lag_{col}' in out.columns

    def test_does_not_mutate_input(self, synthetic_gdf):
        col = INFER_COLS_PLAN[0]
        before = synthetic_gdf.columns.tolist()
        add_spatial_lag_features(synthetic_gdf, [col])
        assert synthetic_gdf.columns.tolist() == before


class TestMoransI:
    def test_returns_dict_keys(self, synthetic_gdf, rng):
        w      = build_weights_libpysal(synthetic_gdf)
        values = rng.normal(0, 1, size=len(synthetic_gdf))
        result = morans_i(values, w, permutations=99)
        assert set(result.keys()) == {'I', 'p_value', 'z_score'}

    def test_i_in_valid_range(self, synthetic_gdf, rng):
        w      = build_weights_libpysal(synthetic_gdf)
        values = rng.normal(0, 1, size=len(synthetic_gdf))
        result = morans_i(values, w, permutations=99)
        assert -1 <= result['I'] <= 1

    def test_p_value_in_valid_range(self, synthetic_gdf, rng):
        w      = build_weights_libpysal(synthetic_gdf)
        values = rng.normal(0, 1, size=len(synthetic_gdf))
        result = morans_i(values, w, permutations=99)
        assert 0 <= result['p_value'] <= 1

    def test_positive_autocorrelation_detected(self, synthetic_gdf):
        """Smooth spatial gradient should yield positive I."""
        w = build_weights_libpysal(synthetic_gdf)
        # Values increase left→right (column index of the 3×3 grid)
        values = np.array([j for _ in range(3) for j in range(3)], dtype=float)
        result = morans_i(values, w, permutations=99)
        assert result['I'] > 0


class TestMoransIByYear:
    def test_returns_dataframe(self, synthetic_gdf, rng):
        w      = build_weights_libpysal(synthetic_gdf)
        values = rng.normal(0, 1, size=(len(synthetic_gdf), 5))
        result = morans_i_by_year(values, w, permutations=99)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, synthetic_gdf, rng):
        n_years = 5
        w       = build_weights_libpysal(synthetic_gdf)
        values  = rng.normal(0, 1, size=(len(synthetic_gdf), n_years))
        result  = morans_i_by_year(values, w, permutations=99)
        assert result.shape == (n_years, 3)

    def test_columns(self, synthetic_gdf, rng):
        w      = build_weights_libpysal(synthetic_gdf)
        values = rng.normal(0, 1, size=(len(synthetic_gdf), 3))
        result = morans_i_by_year(values, w, permutations=99)
        assert set(result.columns) == {'I', 'p_value', 'z_score'}


class TestSpatialLagTransformer:
    def test_fit_returns_self(self, synthetic_gdf):
        transformer = SpatialLagTransformer(feature_cols=INFER_COLS_PLAN[:2])
        out = transformer.fit(synthetic_gdf)
        assert out is transformer

    def test_transform_output_shape(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:3]
        transformer = SpatialLagTransformer(feature_cols=cols)
        transformer.fit(synthetic_gdf)
        out = transformer.transform(synthetic_gdf)
        # n_areas × (n_features + n_lag_features)
        assert out.shape == (len(synthetic_gdf), len(cols) * 2)

    def test_transform_with_separate_lag_cols(self, synthetic_gdf):
        feature_cols = INFER_COLS_PLAN[:3]
        lag_cols     = INFER_COLS_PLAN[:1]
        transformer  = SpatialLagTransformer(feature_cols=feature_cols, lag_cols=lag_cols)
        transformer.fit(synthetic_gdf)
        out = transformer.transform(synthetic_gdf)
        # 3 features + 1 lag feature
        assert out.shape == (len(synthetic_gdf), len(feature_cols) + len(lag_cols))

    def test_default_lag_cols_set_on_fit(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:2]
        transformer = SpatialLagTransformer(feature_cols=cols)
        transformer.fit(synthetic_gdf)
        assert transformer.lag_cols_ == cols

    def test_explicit_lag_cols_preserved(self, synthetic_gdf):
        feature_cols = INFER_COLS_PLAN[:3]
        lag_cols     = INFER_COLS_PLAN[:1]
        transformer  = SpatialLagTransformer(feature_cols=feature_cols, lag_cols=lag_cols)
        transformer.fit(synthetic_gdf)
        assert transformer.lag_cols_ == lag_cols
