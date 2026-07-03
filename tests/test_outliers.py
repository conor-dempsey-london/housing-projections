"""Tests for housing_projections.outliers."""
import geopandas as gpd
import pandas as pd
import pytest

from housing_projections.outliers import apply_outlier_exclusion, get_hard_outlier_lsoa_indices


class TestApplyOutlierExclusion:
    def test_returns_tuple(self, outlier_gdf):
        result = apply_outlier_exclusion(outlier_gdf, verbose=False)
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_geodataframe(self, outlier_gdf):
        gdf_clean, _ = apply_outlier_exclusion(outlier_gdf, verbose=False)
        assert isinstance(gdf_clean, gpd.GeoDataFrame)

    def test_second_element_is_dataframe(self, outlier_gdf):
        _, outlier_df = apply_outlier_exclusion(outlier_gdf, verbose=False)
        assert isinstance(outlier_df, pd.DataFrame)

    def test_outlier_df_has_expected_columns(self, outlier_gdf):
        _, outlier_df = apply_outlier_exclusion(outlier_gdf, verbose=False)
        for col in ('lsoa_idx', 'severity', 'source', 'year'):
            assert col in outlier_df.columns

    def test_hard_outliers_removed(self, outlier_gdf):
        gdf_clean, outlier_df = apply_outlier_exclusion(outlier_gdf, verbose=False)
        hard_idx = get_hard_outlier_lsoa_indices(outlier_df)
        assert len(gdf_clean) == len(outlier_gdf) - len(hard_idx)
