"""
housing-projections CLI

Commands
--------
  run-models   Sample one or more models and save traces.
  compare      Load saved traces and run LOO + sensitivity comparison.
  report       Generate a self-contained HTML analysis report.
"""
import argparse
import os
import sys
from pathlib import Path

import arviz as az

from housing_projections.data import load_data, make_data_dict, validate_data_path
from housing_projections.diagnostics import compute_model_comparison
from housing_projections.html_report import generate_report
from housing_projections.models import M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M0h, M5b
from housing_projections.outliers import apply_outlier_exclusion
from housing_projections.sensitivity import (
    compute_model_agreement_matrix,
    compute_z_model_sensitivity,
)

_ALL_MODELS = {m.name: m for m in [M0, M0h, M1, M2, M3, M4, M5, M5b, M6, M7, M8, M9]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_model_list(s):
    """Parse 'M0,M3,M5' → ['M0', 'M3', 'M5']."""
    return [x.strip() for x in s.split(',') if x.strip()]


def _load_traces(traces_dir, model_names):
    """Load saved .nc traces. Silently skips models with no saved file."""
    traces = {}
    for name in model_names:
        path = Path(traces_dir) / f'{name}.nc'
        if path.exists():
            print(f'  Loading {name} from {path}')
            traces[name] = az.from_netcdf(str(path))
        else:
            print(f'  [skip] {name}: no trace found at {path}')
    return traces


def _discover_traces(traces_dir):
    """Return model names for all .nc files found in traces_dir."""
    d = Path(traces_dir)
    if not d.exists():
        return []
    return [p.stem for p in sorted(d.glob('*.nc'))]


# ── run-models ────────────────────────────────────────────────────────────────

def cmd_run_models(args):
    model_names = _parse_model_list(args.models) if args.models else list(_ALL_MODELS)
    invalid = [n for n in model_names if n not in _ALL_MODELS]
    if invalid:
        print(f'Unknown model(s): {invalid}. Available: {list(_ALL_MODELS)}', file=sys.stderr)
        sys.exit(1)

    print(f'\n── Loading data from {args.data_path} ───────────────────────────')
    validate_data_path(args.data_path)
    gdf = load_data(args.data_path)
    gdf, _ = apply_outlier_exclusion(gdf)
    data = make_data_dict(gdf, n_areas=args.n_areas)

    print(f'   {data["n_areas"]} LSOAs, {data["n_years"]} inference years')

    os.makedirs(args.traces_dir, exist_ok=True)

    for name in model_names:
        ModelClass = _ALL_MODELS[name]
        # M8 requires borough_idx — skip unless user provides borough data
        if name == 'M8' and 'borough_idx' not in data:
            print('\n  [skip] M8 requires borough_idx in data dict. '
                  'Add it manually and run M8 separately.')
            continue

        print(f'\n── Sampling {name}: {ModelClass.description} ──')
        m = ModelClass(data)
        m.build()
        m.sample(use_nutpie=not args.no_nutpie)
        m.save(results_dir=args.traces_dir)
        print(f'  {name} saved.')

    print('\nDone. Run `housing-projections compare` to see LOO results.')


# ── compare ───────────────────────────────────────────────────────────────────

def cmd_compare(args):
    model_names = (_parse_model_list(args.models) if args.models
                   else _discover_traces(args.traces_dir))
    if not model_names:
        print(f'No traces found in {args.traces_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'\n── Loading traces from {args.traces_dir} ───────────────────────')
    traces = _load_traces(args.traces_dir, model_names)
    if not traces:
        print('No traces loaded. Run `housing-projections run-models` first.', file=sys.stderr)
        sys.exit(1)

    print(f'\n── LOO model comparison ({len(traces)} models) ──────────────────')
    comparison = compute_model_comparison(traces, verbose=True)

    print('\n── Model agreement (z posterior correlation) ────────────────────')
    corr = compute_model_agreement_matrix(traces)
    print(corr.to_string(float_format=lambda x: f'{x:.4f}'))

    print('\n── Z model sensitivity ──────────────────────────────────────────')
    summary, _ = compute_z_model_sensitivity(traces)
    std_col = summary['z_std_across_models']
    print(f'  Mean z std across models: {std_col.mean():.3f} dwellings/year')
    print(f'  Median z std:             {std_col.median():.3f}')
    print(f'  90th pct z std:           {std_col.quantile(0.90):.3f}')
    print(f'  Max z std (most sensitive LSOA): {std_col.max():.3f}')

    return comparison


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args):
    print(f'\n── Loading data from {args.data_path} ───────────────────────────')
    validate_data_path(args.data_path)
    gdf = load_data(args.data_path)
    gdf, _ = apply_outlier_exclusion(gdf)
    data = make_data_dict(gdf, n_areas=args.n_areas)

    model_names = (_parse_model_list(args.models) if args.models
                   else _discover_traces(args.traces_dir))
    if not model_names:
        print(f'No traces found in {args.traces_dir}. '
              f'Run `housing-projections run-models` first.', file=sys.stderr)
        sys.exit(1)

    print('\n── Loading traces ───────────────────────────────────────────────')
    traces = _load_traces(args.traces_dir, model_names)

    model_classes = {name: _ALL_MODELS[name] for name in traces if name in _ALL_MODELS}

    print(f'\n── Generating report → {args.output} ────────────────────────────')
    os.makedirs(Path(args.output).parent, exist_ok=True)
    generate_report(
        data=data,
        traces=traces,
        model_classes=model_classes,
        output_path=args.output,
        title=args.title,
    )
    print(f'\nReport written to {args.output}')


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(
        prog='housing-projections',
        description='Bayesian dwelling projection tools for London LSOAs.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── run-models ──────────────────────────────────────────────────────────
    p_run = sub.add_parser('run-models', help='Sample models and save traces.')
    p_run.add_argument('--data-path', required=True,
                       help='Root directory of raw data files.')
    p_run.add_argument('--models', default=None,
                       help='Comma-separated model names, e.g. M0,M3,M5 (default: all).')
    p_run.add_argument('--n-areas', type=int, default=None,
                       help='Subsample to N LSOAs for faster runs.')
    p_run.add_argument('--traces-dir', default='results/traces',
                       help='Directory to save .nc trace files (default: results/traces).')
    p_run.add_argument('--no-nutpie', action='store_true',
                       help='Disable nutpie sampler, fall back to PyMC.')

    # ── compare ─────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser('compare', help='LOO comparison and z sensitivity report.')
    p_cmp.add_argument('--traces-dir', default='results/traces',
                       help='Directory containing saved .nc trace files.')
    p_cmp.add_argument('--models', default=None,
                       help='Comma-separated model names (default: all found in traces-dir).')

    # ── report ──────────────────────────────────────────────────────────────
    p_rep = sub.add_parser('report', help='Generate self-contained HTML analysis report.')
    p_rep.add_argument('--data-path', required=True,
                       help='Root directory of raw data files (required for EDA).')
    p_rep.add_argument('--traces-dir', default='results/traces',
                       help='Directory containing saved .nc trace files.')
    p_rep.add_argument('--models', default=None,
                       help='Comma-separated model names to include (default: all found).')
    p_rep.add_argument('--n-areas', type=int, default=None,
                       help='Subsample to N LSOAs (must match traces if sub-sampled).')
    p_rep.add_argument('--output', default='results/report.html',
                       help='Output HTML path (default: results/report.html).')
    p_rep.add_argument('--title', default='Housing Projections: Model Analysis Report',
                       help='Report title.')

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'run-models':
        cmd_run_models(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)
