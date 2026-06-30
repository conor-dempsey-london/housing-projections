"""Tests for housing_projections.outliers."""
import numpy as np
import pandas as pd
import pytest

from housing_projections.outliers import (
    find_outliers,
    get_hard_outlier_lsoa_indices,
    get_soft_outlier_lsoa_indices,
    analyse_outliers,
    exclude_hard_outlier_lsoas,
    apply_outlier_exclusion,
)
from housing_projections.config import INFER_COLS_PLAN, INFER_COLS_BEN


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_outlier_df():
    """Empty outlier DataFrame (no anomalies detected)."""
    return pd.DataFrame(columns=['lsoa_idx', 'lsoa_id', 'year',
                                  'source', 'value', 'reason', 'severity'])


# ── find_outliers ──────────────────────────────────────────────────────────────

class TestFindOutliers:
    def test_clean_data_returns_empty(self, synthetic_gdf):
        df = find_outliers(synthetic_gdf)
        assert len(df) == 0

    def test_hard_outlier_above_max(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        hard = df[df['severity'] == 'hard']
        assert len(hard) > 0
        assert any(hard['source'] == 'planning')

    def test_hard_outlier_below_min(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        hard = df[df['severity'] == 'hard']
        assert any(hard['source'] == 'ben')

    def test_soft_outlier_detected(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        soft = df[df['severity'] == 'soft']
        assert len(soft) > 0
        assert all(soft['source'] == 'both')

    def test_output_columns(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        expected = {'lsoa_idx', 'lsoa_id', 'year', 'source', 'value', 'reason', 'severity'}
        assert expected.issubset(df.columns)

    def test_severity_values(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        assert set(df['severity'].unique()).issubset({'hard', 'soft'})

    def test_hard_outlier_lsoa_idx(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        hard_indices = get_hard_outlier_lsoa_indices(df)
        assert 0 in hard_indices  # area 0: planning hard outlier
        assert 1 in hard_indices  # area 1: BEN hard outlier

    def test_soft_outlier_lsoa_idx(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        soft_indices = get_soft_outlier_lsoa_indices(df)
        assert 2 in soft_indices  # area 2: soft outlier

    def test_custom_thresholds(self, synthetic_gdf):
        """Setting max_plausible=0 makes every positive value a hard outlier."""
        df = find_outliers(synthetic_gdf, max_plausible=0, min_plausible=-9999)
        hard = df[df['severity'] == 'hard']
        # The synthetic data has many positive values, so at least one hard outlier
        assert len(hard) > 0

    def test_hard_reason_contains_threshold(self, outlier_gdf):
        df   = find_outliers(outlier_gdf)
        hard = df[df['severity'] == 'hard']
        reasons = hard['reason'].str.lower()
        assert any('max_plausible' in r or 'min_plausible' in r for r in reasons)

    def test_soft_reason_mentions_discrepancy(self, outlier_gdf):
        df   = find_outliers(outlier_gdf)
        soft = df[df['severity'] == 'soft']
        assert all('discrepancy' in r.lower() for r in soft['reason'])


# ── get_hard_outlier_lsoa_indices ──────────────────────────────────────────────

class TestGetHardOutlierLsoaIndices:
    def test_empty_df_returns_empty(self, clean_outlier_df):
        assert get_hard_outlier_lsoa_indices(clean_outlier_df) == []

    def test_returns_sorted_list(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        indices = get_hard_outlier_lsoa_indices(df)
        assert indices == sorted(indices)

    def test_only_hard_included(self):
        df = pd.DataFrame([
            {'lsoa_idx': 0, 'severity': 'hard'},
            {'lsoa_idx': 1, 'severity': 'soft'},
        ])
        assert get_hard_outlier_lsoa_indices(df) == [0]


# ── get_soft_outlier_lsoa_indices ──────────────────────────────────────────────

class TestGetSoftOutlierLsoaIndices:
    def test_empty_df_returns_empty(self, clean_outlier_df):
        assert get_soft_outlier_lsoa_indices(clean_outlier_df) == []

    def test_hard_lsoas_excluded(self, outlier_gdf):
        df   = find_outliers(outlier_gdf)
        hard = set(get_hard_outlier_lsoa_indices(df))
        soft = set(get_soft_outlier_lsoa_indices(df))
        assert hard.isdisjoint(soft), "Hard-outlier LSOAs must not appear in soft list"

    def test_area_2_in_soft_not_hard(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        assert 2 in get_soft_outlier_lsoa_indices(df)
        assert 2 not in get_hard_outlier_lsoa_indices(df)

    def test_returns_sorted_list(self, outlier_gdf):
        df = find_outliers(outlier_gdf)
        indices = get_soft_outlier_lsoa_indices(df)
        assert indices == sorted(indices)


# ── analyse_outliers ───────────────────────────────────────────────────────────

class TestAnalyseOutliers:
    def test_returns_dict_structure(self, outlier_gdf):
        df     = find_outliers(outlier_gdf)
        result = analyse_outliers(outlier_gdf, df, verbose=False)
        assert 'hard' in result and 'soft' in result

    def test_hard_keys(self, outlier_gdf):
        df     = find_outliers(outlier_gdf)
        result = analyse_outliers(outlier_gdf, df, verbose=False)
        for key in ('n_lsoas', 'n_lsoa_years', 'by_source', 'by_year', 'details'):
            assert key in result['hard']

    def test_soft_keys(self, outlier_gdf):
        df     = find_outliers(outlier_gdf)
        result = analyse_outliers(outlier_gdf, df, verbose=False)
        for key in ('n_lsoas', 'n_lsoa_years', 'by_source', 'by_year', 'details'):
            assert key in result['soft']

    def test_clean_data_zero_counts(self, synthetic_gdf, clean_outlier_df):
        result = analyse_outliers(synthetic_gdf, clean_outlier_df, verbose=False)
        assert result['hard']['n_lsoas'] == 0
        assert result['soft']['n_lsoas'] == 0

    def test_hard_count_matches_find_outliers(self, outlier_gdf):
        df     = find_outliers(outlier_gdf)
        result = analyse_outliers(outlier_gdf, df, verbose=False)
        expected = df[df['severity'] == 'hard']['lsoa_idx'].nunique()
        assert result['hard']['n_lsoas'] == expected


# ── exclude_hard_outlier_lsoas ─────────────────────────────────────────────────

class TestExcludeHardOutlierLsoas:
    def test_removes_hard_outlier_rows(self, outlier_gdf):
        df        = find_outliers(outlier_gdf)
        clean     = exclude_hard_outlier_lsoas(outlier_gdf, df, verbose=False)
        hard_idx  = get_hard_outlier_lsoa_indices(df)
        assert len(clean) == len(outlier_gdf) - len(hard_idx)

    def test_index_is_reset(self, outlier_gdf):
        df    = find_outliers(outlier_gdf)
        clean = exclude_hard_outlier_lsoas(outlier_gdf, df, verbose=False)
        assert list(clean.index) == list(range(len(clean)))

    def test_soft_lsoas_retained(self, outlier_gdf):
        df        = find_outliers(outlier_gdf)
        clean     = exclude_hard_outlier_lsoas(outlier_gdf, df, verbose=False)
        soft_ids  = set(outlier_gdf.iloc[get_soft_outlier_lsoa_indices(df)]['LSOA21CD'])
        clean_ids = set(clean['LSOA21CD'])
        assert soft_ids.issubset(clean_ids)

    def test_clean_data_unchanged_length(self, synthetic_gdf, clean_outlier_df):
        result = exclude_hard_outlier_lsoas(synthetic_gdf, clean_outlier_df, verbose=False)
        assert len(result) == len(synthetic_gdf)

    def test_returns_geodataframe(self, outlier_gdf):
        import geopandas as gpd
        df    = find_outliers(outlier_gdf)
        clean = exclude_hard_outlier_lsoas(outlier_gdf, df, verbose=False)
        assert isinstance(clean, gpd.GeoDataFrame)


# ── apply_outlier_exclusion ────────────────────────────────────────────────────

class TestApplyOutlierExclusion:
    def test_returns_tuple(self, outlier_gdf):
        result = apply_outlier_exclusion(outlier_gdf, verbose=False)
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_geodataframe(self, outlier_gdf):
        import geopandas as gpd
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
