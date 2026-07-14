"""
Smoke test for housing_projections.html_report.generate_report — the CLI-facing report
builder (see html_report.py's own module docstring for how this differs from
reporting.full_report). Uses synthetic trace/data from conftest; writes to tmp_path so no
real results/ file is touched.
"""
import matplotlib

matplotlib.use('Agg')  # non-interactive backend so plt.show() is a no-op

from housing_projections.html_report import generate_report


class TestGenerateReport:
    def test_runs_for_single_model(self, mock_trace, data_dict, tmp_path):
        output_path = tmp_path / 'report.html'
        generate_report(
            data=data_dict,
            traces={'M0h': mock_trace},
            output_path=str(output_path),
        )
        assert output_path.exists()
        html = output_path.read_text(encoding='utf-8')
        assert '<html' in html.lower()

    def test_runs_for_multiple_models(self, mock_trace, mock_trace_with_divergences, data_dict, tmp_path):
        # len(traces) > 1 exercises the LOO comparison / sensitivity branches too.
        output_path = tmp_path / 'report.html'
        generate_report(
            data=data_dict,
            traces={'M0h': mock_trace, 'M1h': mock_trace_with_divergences},
            output_path=str(output_path),
        )
        assert output_path.exists()
