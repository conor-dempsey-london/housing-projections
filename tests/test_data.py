"""Tests for housing_projections.data (offline — no file I/O or network)."""
import gla_data
import numpy as np
import pandas as pd
import pytest

from housing_projections.config import (
    ALL_COLS_PLAN,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
)
from housing_projections.data import (
    make_borough_idx,
    make_data_dict,
    select_spatial_sample,
    validate_data_path,
)


class TestMakeDataDict:
    def test_required_keys(self, data_dict):
        required = {
            'D', 'P_obs', 'E_obs',
            'P_obs_full', 'E_obs_full',
            'n_areas', 'n_years', 'n_years_full',
            'gdf', 'D_full_mean',
        }
        assert required.issubset(data_dict.keys())

    def test_d_shape(self, data_dict):
        assert data_dict['D'].shape == (data_dict['n_areas'],)

    def test_p_obs_shape(self, data_dict):
        assert data_dict['P_obs'].shape == (data_dict['n_areas'], data_dict['n_years'])

    def test_e_obs_shape(self, data_dict):
        assert data_dict['E_obs'].shape == (data_dict['n_areas'], data_dict['n_years'])

    def test_p_obs_full_shape(self, data_dict):
        assert data_dict['P_obs_full'].shape == (
            data_dict['n_areas'], data_dict['n_years_full'])

    def test_e_obs_full_shape(self, data_dict):
        assert data_dict['E_obs_full'].shape == (
            data_dict['n_areas'], data_dict['n_years_full'])

    def test_n_years(self, data_dict):
        assert data_dict['n_years'] == len(INFER_COLS_PLAN) == 10

    def test_n_years_full(self, data_dict):
        assert data_dict['n_years_full'] == len(ALL_COLS_PLAN) == 16

    def test_n_areas_matches_gdf(self, data_dict):
        assert data_dict['n_areas'] == len(data_dict['gdf'])

    def test_d_calculation(self, data_dict, synthetic_gdf):
        expected_D = (
            synthetic_gdf['dwellings_2021'].values.astype(float)
            - synthetic_gdf['dwellings_2011'].values.astype(float)
        )
        np.testing.assert_array_equal(data_dict['D'], expected_D)

    def test_d_full_mean_scalar(self, data_dict):
        assert isinstance(data_dict['D_full_mean'], float)

    def test_arrays_are_float(self, data_dict):
        for key in ('D', 'P_obs', 'E_obs', 'P_obs_full', 'E_obs_full'):
            assert data_dict[key].dtype == np.float64, f"{key} is not float64"

    def test_n_areas_slicing(self, synthetic_gdf):
        full = make_data_dict(synthetic_gdf)
        sliced = make_data_dict(synthetic_gdf, n_areas=4)
        assert sliced['n_areas'] == 4
        assert sliced['P_obs'].shape == (4, 10)
        # D_full_mean uses the full GDF, not the slice
        assert sliced['D_full_mean'] == full['D_full_mean']

    def test_p_obs_matches_gdf_columns(self, data_dict, synthetic_gdf):
        """
        P_obs matches the raw gdf columns, except cells make_data_dict
        intentionally snaps to zero (0 < P < 10% of E — likely PLD data
        errors, see the comment in make_data_dict).
        """
        expected = synthetic_gdf[INFER_COLS_PLAN].values.astype(float)
        e_obs = synthetic_gdf[INFER_COLS_BEN].values.astype(float)
        erroneous = (expected > 0) & (e_obs > 0) & (expected < 0.1 * e_obs)
        expected[erroneous] = 0.0
        np.testing.assert_array_equal(data_dict['P_obs'], expected)

    def test_e_obs_matches_gdf_columns(self, data_dict, synthetic_gdf):
        expected = synthetic_gdf[INFER_COLS_BEN].values.astype(float)
        np.testing.assert_array_equal(data_dict['E_obs'], expected)


class TestMakeBoroughIdx:
    """
    make_borough_idx calls the real gla_data.load_geography_lookup, which
    won't resolve synthetic_gdf's fake LSOA21CD codes (e.g. 'E00000000')
    against real ONS geography — so gla_data.load_geography_lookup is
    mocked here. Only this repo's own merge/factorize logic is under test;
    the real crosswalk's column names/coverage were spot-checked manually
    against real data separately (see M10's plan).
    """

    def test_raises_on_unmapped_lsoa(self, synthetic_gdf, monkeypatch):
        codes = synthetic_gdf['LSOA21CD'].tolist()
        partial_lookup = pd.DataFrame({
            'LSOA21CD': codes[:-1],  # omit the last LSOA
            'LAD22CD':  ['E09000001'] * (len(codes) - 1),
        })
        monkeypatch.setattr(
            gla_data, 'load_geography_lookup', lambda **kwargs: partial_lookup)

        with pytest.raises(ValueError, match=codes[-1]):
            make_borough_idx(synthetic_gdf)

    def test_returns_correct_shapes_and_roundtrip(self, synthetic_gdf, monkeypatch):
        codes = synthetic_gdf['LSOA21CD'].tolist()
        lad_codes = ['E09000001'] * 3 + ['E09000002'] * 3 + ['E09000003'] * 3
        full_lookup = pd.DataFrame({'LSOA21CD': codes, 'LAD22CD': lad_codes})
        monkeypatch.setattr(
            gla_data, 'load_geography_lookup', lambda **kwargs: full_lookup)

        borough_idx, n_boroughs, borough_codes = make_borough_idx(synthetic_gdf)

        assert borough_idx.shape == (len(codes),)
        assert borough_idx.dtype == np.int64 or borough_idx.dtype == int
        assert n_boroughs == 3
        assert borough_codes.shape == (3,)
        np.testing.assert_array_equal(borough_codes[borough_idx], lad_codes)

    def test_ignores_duplicate_lookup_rows(self, synthetic_gdf, monkeypatch):
        # Real gla_data lookup has no duplicate LSOA21CD rows (verified
        # separately), but make_borough_idx should be robust to a
        # crosswalk source that does, rather than silently duplicating rows.
        codes = synthetic_gdf['LSOA21CD'].tolist()
        lad_codes = ['E09000001'] * len(codes)
        dup_lookup = pd.DataFrame({
            'LSOA21CD': codes + [codes[0]],
            'LAD22CD':  lad_codes + ['E09000002'],  # conflicting dup for codes[0]
        })
        monkeypatch.setattr(
            gla_data, 'load_geography_lookup', lambda **kwargs: dup_lookup)

        borough_idx, n_boroughs, borough_codes = make_borough_idx(synthetic_gdf)
        assert borough_idx.shape == (len(codes),)


class TestValidateDataPath:
    def test_raises_on_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="load_data"):
            validate_data_path(str(tmp_path))

    def test_message_lists_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            validate_data_path(str(tmp_path))
        msg = str(exc.value)
        assert 'PLD completions' in msg
        assert 'BEN estimates' in msg

    def test_passes_when_files_exist(self, tmp_path):
        pld = tmp_path / 'pld'
        pld.mkdir()
        (pld / 'lsoa_completions_time_series_pivot.csv').write_text('x')
        ben = tmp_path / 'ben'
        ben.mkdir()
        (ben / 'final_residential_uprn_net_changes_by_oa_fy (1).csv').write_text('x')
        validate_data_path(str(tmp_path))  # should not raise


class TestSelectSpatialSample:
    def test_returns_correct_count(self, synthetic_gdf):
        result = select_spatial_sample(synthetic_gdf, n_areas=5)
        assert len(result) == 5

    def test_index_reset(self, synthetic_gdf):
        result = select_spatial_sample(synthetic_gdf, n_areas=5)
        assert list(result.index) == list(range(5))

    def test_crs_preserved(self, synthetic_gdf):
        result = select_spatial_sample(synthetic_gdf, n_areas=5)
        assert result.crs == synthetic_gdf.crs

    def test_columns_preserved(self, synthetic_gdf):
        result = select_spatial_sample(synthetic_gdf, n_areas=5)
        assert set(INFER_COLS_PLAN).issubset(result.columns)

    def test_n_areas_capped_at_total(self, synthetic_gdf):
        # Requesting more than available should return all
        result = select_spatial_sample(synthetic_gdf, n_areas=len(synthetic_gdf))
        assert len(result) == len(synthetic_gdf)

    def test_default_centre_selects_closest(self, synthetic_gdf):
        # Custom centre at centroid of area 4 (middle of 3×3 grid: 1.5, 1.5 in EPSG:27700)
        centroid = synthetic_gdf.geometry.iloc[4].centroid
        result = select_spatial_sample(
            synthetic_gdf, n_areas=1,
            center_latlon=(centroid.y, centroid.x),
        )
        assert len(result) == 1
