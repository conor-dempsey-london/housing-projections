"""Tests for housing_projections.config."""
import pytest
from housing_projections.config import (
    INFER_YEARS, N_YEARS,
    INFER_COLS_PLAN, INFER_COLS_BEN,
    ALL_COLS_PLAN, ALL_COLS_BEN,
    CENSUS_COLS,
    CENSUS_REL_ERROR, CENSUS_ABS_FLOOR,
    DEFAULT_SAMPLE_KWARGS,
    COLOURS,
    LONDON_LAS,
)


class TestInferenceYears:
    def test_infer_years_range(self):
        assert INFER_YEARS == list(range(2012, 2022))

    def test_n_years_matches(self):
        assert N_YEARS == len(INFER_YEARS) == 10


class TestColumnNaming:
    def test_infer_cols_plan_count(self):
        assert len(INFER_COLS_PLAN) == 10

    def test_infer_cols_plan_format(self):
        # e.g. '2011/12', '2012/13'
        assert INFER_COLS_PLAN[0] == '2011/12'
        assert INFER_COLS_PLAN[-1] == '2020/21'

    def test_infer_cols_ben_count(self):
        assert len(INFER_COLS_BEN) == 10

    def test_infer_cols_ben_format(self):
        assert INFER_COLS_BEN[0] == '2011_ben'
        assert INFER_COLS_BEN[-1] == '2020_ben'

    def test_all_cols_plan_count(self):
        assert len(ALL_COLS_PLAN) == 16  # 2009-2024

    def test_all_cols_ben_count(self):
        assert len(ALL_COLS_BEN) == 16

    def test_infer_cols_subset_of_all(self):
        assert set(INFER_COLS_PLAN).issubset(set(ALL_COLS_PLAN))
        assert set(INFER_COLS_BEN).issubset(set(ALL_COLS_BEN))

    def test_census_cols(self):
        assert 'dwellings_2011' in CENSUS_COLS
        assert 'dwellings_2021' in CENSUS_COLS


class TestCensusConstraintParams:
    def test_rel_error_is_small_positive(self):
        assert 0 < CENSUS_REL_ERROR < 0.1

    def test_abs_floor_is_positive(self):
        assert CENSUS_ABS_FLOOR > 0


class TestSamplingDefaults:
    def test_required_keys_present(self):
        for key in ('draws', 'tune', 'chains', 'random_seed'):
            assert key in DEFAULT_SAMPLE_KWARGS

    def test_sensible_draws(self):
        assert DEFAULT_SAMPLE_KWARGS['draws'] >= 500

    def test_random_seed_set(self):
        # Reproducibility: seed must be an integer
        assert isinstance(DEFAULT_SAMPLE_KWARGS['random_seed'], int)


class TestColours:
    def test_required_keys(self):
        for key in ('z', 'planning', 'ben', 'baseline', 'posterior'):
            assert key in COLOURS

    def test_values_are_strings(self):
        assert all(isinstance(v, str) for v in COLOURS.values())


class TestLondonLAs:
    def test_count(self):
        assert len(LONDON_LAS) == 33

    def test_known_boroughs_present(self):
        for name in ('Islington', 'Camden', 'Tower Hamlets', 'Westminster'):
            assert name in LONDON_LAS

    def test_no_duplicates(self):
        assert len(LONDON_LAS) == len(set(LONDON_LAS))
