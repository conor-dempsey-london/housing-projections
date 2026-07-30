"""Tests for housing_projections.cli — mostly argument parsing (no I/O), plus a
small tmp_path-based regression test for the comparison cache's model-set check."""
import os

import pandas as pd
import pytest

from housing_projections.cli import (
    _build_parser,
    _load_comparison_cache,
    _parse_model_list,
    _save_comparison_cache,
)


class TestParseModelList:
    def test_single(self):
        assert _parse_model_list('M0') == ['M0']

    def test_multiple(self):
        assert _parse_model_list('M0,M1,M3') == ['M0', 'M1', 'M3']

    def test_strips_whitespace(self):
        assert _parse_model_list('M0, M1 , M3') == ['M0', 'M1', 'M3']

    def test_empty_string_gives_empty_list(self):
        assert _parse_model_list('') == []


class TestParser:
    def _parse(self, args):
        return _build_parser().parse_args(args)

    def test_run_models_defaults(self):
        args = self._parse(['run-models'])
        assert args.data_path == 'data'
        assert args.models is None
        assert args.n_areas is None
        assert args.traces_dir == 'results/traces'
        assert args.no_nutpie is False

    def test_run_models_custom_args(self):
        args = self._parse(['run-models', '--data-path', '/d', '--models', 'M0,M1',
                            '--n-areas', '100', '--traces-dir', '/traces', '--no-nutpie'])
        assert args.models == 'M0,M1'
        assert args.n_areas == 100
        assert args.traces_dir == '/traces'
        assert args.no_nutpie is True

    def test_compare_defaults(self):
        args = self._parse(['compare'])
        assert args.traces_dir == 'results/traces'
        assert args.models is None

    def test_compare_custom(self):
        args = self._parse(['compare', '--traces-dir', '/t', '--models', 'M0,M3'])
        assert args.traces_dir == '/t'
        assert args.models == 'M0,M3'

    def test_report_defaults(self):
        args = self._parse(['report'])
        assert args.data_path == 'data'
        assert args.output == 'results/report.html'
        assert args.title == 'Housing Projections: Model Analysis Report'
        assert args.traces_dir == 'results/traces'

    def test_report_custom_output(self):
        args = self._parse(['report', '--data-path', '/d', '--output', '/out/r.html',
                            '--title', 'My Report'])
        assert args.output == '/out/r.html'
        assert args.title == 'My Report'

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            self._parse([])

    def test_check_multimodality_defaults(self):
        args = self._parse(['check-multimodality'])
        assert args.traces_dir == 'results/traces'
        assert args.models is None
        # None means "every *_lambda_weights var found in each trace" -- see
        # cmd_check_multimodality; there is no single-variable default any more since
        # that was exactly the source of the check-multimodality/diagnose mismatch this
        # arg replaced.
        assert args.lag_var is None
        assert args.rhat_threshold == 1.01
        assert args.resolve is False
        assert args.resolve_chains == 16
        assert args.data_path == 'data'

    def test_check_multimodality_custom_args(self):
        args = self._parse(['check-multimodality', '--traces-dir', '/t', '--models', 'AZ1d',
                            '--lag-var', 'lag_E_lambda_weights',
                            '--rhat-threshold', '1.05', '--resolve', '--resolve-chains', '8',
                            '--data-path', '/d'])
        assert args.traces_dir == '/t'
        assert args.models == 'AZ1d'
        assert args.lag_var == 'lag_E_lambda_weights'
        assert args.rhat_threshold == 1.05
        assert args.resolve is True
        assert args.resolve_chains == 8
        assert args.data_path == '/d'

    def test_check_multimodality_lag_var_accepts_comma_list(self):
        args = self._parse(['check-multimodality',
                            '--lag-var', 'lag_P_lambda_weights,lag_E_lambda_weights'])
        assert args.lag_var == 'lag_P_lambda_weights,lag_E_lambda_weights'


class TestComparisonCache:
    """Regression coverage for a real bug: `compare --models A,B,C` after an
    earlier `compare --models A,B` run silently returned the stale 2-model
    result, because the cache validity check only looked at mtimes for
    whatever models were already in the cache, never at whether the currently
    requested model set even matched."""

    def _touch_trace(self, traces_dir, name, mtime=None):
        path = traces_dir / f'{name}.nc'
        path.write_text('placeholder')
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def test_cache_hit_when_model_set_and_mtimes_match(self, tmp_path):
        for name in ('A', 'B'):
            self._touch_trace(tmp_path, name)
        df = pd.DataFrame({'elpd': [1.0, 2.0]}, index=['A', 'B'])
        _save_comparison_cache(tmp_path, df, ['A', 'B'])

        loaded = _load_comparison_cache(tmp_path, ['A', 'B'])

        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)

    def test_cache_miss_when_a_new_model_is_requested(self, tmp_path):
        for name in ('A', 'B', 'C'):
            self._touch_trace(tmp_path, name)
        df = pd.DataFrame({'elpd': [1.0, 2.0]}, index=['A', 'B'])
        _save_comparison_cache(tmp_path, df, ['A', 'B'])

        loaded = _load_comparison_cache(tmp_path, ['A', 'B', 'C'])

        assert loaded is None

    def test_cache_miss_when_fewer_models_requested(self, tmp_path):
        for name in ('A', 'B'):
            self._touch_trace(tmp_path, name)
        df = pd.DataFrame({'elpd': [1.0, 2.0]}, index=['A', 'B'])
        _save_comparison_cache(tmp_path, df, ['A', 'B'])

        loaded = _load_comparison_cache(tmp_path, ['A'])

        assert loaded is None

    def test_cache_miss_when_a_cached_trace_changed(self, tmp_path):
        for name in ('A', 'B'):
            self._touch_trace(tmp_path, name)
        df = pd.DataFrame({'elpd': [1.0, 2.0]}, index=['A', 'B'])
        _save_comparison_cache(tmp_path, df, ['A', 'B'])

        original_mtime = (tmp_path / 'A.nc').stat().st_mtime
        self._touch_trace(tmp_path, 'A', mtime=original_mtime + 100)
        loaded = _load_comparison_cache(tmp_path, ['A', 'B'])

        assert loaded is None
