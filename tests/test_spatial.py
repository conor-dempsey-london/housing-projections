"""
Tests for housing_projections.spatial — public API smoke tests.
All tests use the synthetic 3×3 GDF from conftest and run offline.
"""
import numpy as np

from housing_projections.config import INFER_COLS_PLAN
from housing_projections.spatial import (
    SpatialLagTransformer,
    add_spatial_lag_features,
    build_spatial_weights,
    build_weights_libpysal,
    compute_morans_i,
    compute_morans_i_by_year,
)


class TestBuildWeights:
    def test_libpysal_weights_shape(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        assert len(w.neighbors) == len(synthetic_gdf)

    def test_libpysal_weights_row_normalised(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        for i, weights in w.weights.items():
            if weights:
                assert abs(sum(weights) - 1.0) < 1e-9

    def test_dense_weights_shape(self, synthetic_gdf):
        n = len(synthetic_gdf)
        W = build_spatial_weights(synthetic_gdf)
        assert W.shape == (n, n)

    def test_dense_weights_row_normalised(self, synthetic_gdf):
        W = build_spatial_weights(synthetic_gdf)
        row_sums = W.sum(axis=1)
        # Corner cells have fewer neighbours — every non-isolated row sums to 1
        for s in row_sums:
            assert abs(s - 1.0) < 1e-9 or s == 0.0


class TestMoransI:
    def test_returns_expected_keys(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        values = synthetic_gdf[INFER_COLS_PLAN[0]].values.astype(float)
        result = compute_morans_i(values, w, permutations=9)
        assert set(result.keys()) == {'I', 'p_value', 'z_score'}

    def test_i_in_valid_range(self, synthetic_gdf):
        w = build_weights_libpysal(synthetic_gdf)
        values = synthetic_gdf[INFER_COLS_PLAN[0]].values.astype(float)
        result = compute_morans_i(values, w, permutations=9)
        assert -1.0 <= result['I'] <= 1.0

    def test_by_year_returns_dataframe(self, synthetic_gdf):
        import pandas as pd
        w = build_weights_libpysal(synthetic_gdf)
        values_by_year = synthetic_gdf[INFER_COLS_PLAN].values.astype(float)
        df = compute_morans_i_by_year(values_by_year, w, permutations=9)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['I', 'p_value', 'z_score']
        assert len(df) == values_by_year.shape[1]


class TestSpatialLagFeatures:
    def test_adds_lag_columns(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:2]
        result = add_spatial_lag_features(synthetic_gdf, cols)
        for col in cols:
            assert f'lag_{col}' in result.columns

    def test_original_columns_unchanged(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:2]
        result = add_spatial_lag_features(synthetic_gdf, cols)
        for col in cols:
            np.testing.assert_array_equal(result[col].values, synthetic_gdf[col].values)


class TestSpatialLagTransformer:
    def test_fit_transform_shape(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:3]
        transformer = SpatialLagTransformer(feature_cols=cols)
        X = transformer.fit_transform(synthetic_gdf)
        assert X.shape == (len(synthetic_gdf), len(cols) * 2)

    def test_fit_transform_separate(self, synthetic_gdf):
        cols = INFER_COLS_PLAN[:2]
        transformer = SpatialLagTransformer(feature_cols=cols)
        transformer.fit(synthetic_gdf)
        X = transformer.transform(synthetic_gdf)
        assert X.shape == (len(synthetic_gdf), len(cols) * 2)

    def test_custom_lag_cols(self, synthetic_gdf):
        feature_cols = INFER_COLS_PLAN[:3]
        lag_cols = INFER_COLS_PLAN[:1]
        transformer = SpatialLagTransformer(feature_cols=feature_cols, lag_cols=lag_cols)
        X = transformer.fit_transform(synthetic_gdf)
        assert X.shape == (len(synthetic_gdf), len(feature_cols) + len(lag_cols))
