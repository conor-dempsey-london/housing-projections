"""Tests for housing_projections.cli — argument parsing only, no I/O."""
import pytest

from housing_projections.cli import _build_parser, _parse_model_list


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
