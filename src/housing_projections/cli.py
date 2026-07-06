"""
housing-projections CLI

Commands
--------
  run-models   Sample one or more models and save traces.
  compare      Load saved traces and run LOO + sensitivity comparison.
  report       Generate a self-contained HTML analysis report.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import arviz as az
import pandas as pd

from housing_projections.data import (
    load_data,
    make_data_dict,
    select_spatial_sample,
    validate_data_path,
)
from housing_projections.diagnostics import compute_model_comparison, diagnostics_summary
from housing_projections.html_report import generate_report
from housing_projections.models import M0, M0h, M1, M2, M3, M4, M5, M6, M7, M8
from housing_projections.outliers import apply_outlier_exclusion
from housing_projections.sensitivity import (
    compute_decomposed_uncertainty,
    compute_model_agreement_matrix,
    compute_z_model_sensitivity,
)

_ALL_MODELS = {m.name: m for m in [M0, M0h, M1, M2, M3, M4, M5, M6, M7, M8]}

_COMPARISON_CSV  = 'comparison.csv'
_COMPARISON_META = 'comparison_meta.json'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_model_list(s):
    """Parse 'M0,M1,M3' → ['M0', 'M1', 'M3']."""
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


def _trace_mtimes(traces_dir, model_names):
    """Return {name: mtime} for each trace file that exists."""
    return {
        name: Path(traces_dir, f'{name}.nc').stat().st_mtime
        for name in model_names
        if Path(traces_dir, f'{name}.nc').exists()
    }


def _load_comparison_cache(traces_dir):
    """
    Load cached LOO comparison if it is still valid (all trace files unchanged).

    Returns pd.DataFrame or None.
    """
    meta_path = Path(traces_dir) / _COMPARISON_META
    csv_path  = Path(traces_dir) / _COMPARISON_CSV
    if not meta_path.exists() or not csv_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        current_mtimes = _trace_mtimes(traces_dir, list(meta['mtimes']))
        if current_mtimes == meta['mtimes']:
            df = pd.read_csv(csv_path, index_col=0)
            print('  LOO comparison loaded from cache.')
            return df
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_comparison_cache(traces_dir, comparison_df, model_names):
    """Persist LOO comparison results and trace mtimes to disk."""
    meta = {'mtimes': _trace_mtimes(traces_dir, model_names)}
    Path(traces_dir, _COMPARISON_META).write_text(json.dumps(meta))
    comparison_df.to_csv(Path(traces_dir) / _COMPARISON_CSV)


def _data_matching_traces(gdf, traces, n_areas_hint=None):
    """
    Build a data dict whose rows exactly match the areas the traces were sampled on.

    Reads LSOA codes from the 'area' coordinate embedded in the trace by
    DwellingModel._default_coords() at sampling time, then filters and reorders
    gdf to match.  Falls back to iloc[:n_areas] for traces that pre-date the
    coordinate embedding (no 'area' coord present).
    """
    first_trace = next(iter(traces.values()))
    z_posterior = first_trace.posterior['z']

    if 'area' in z_posterior.coords:
        lsoa_codes = z_posterior.coords['area'].values.tolist()
        subset = gdf[gdf['LSOA21CD'].isin(lsoa_codes)].copy()
        subset = subset.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
        return make_data_dict(subset)

    # Legacy fallback: traces sampled before coordinate embedding
    trace_n_areas = z_posterior.shape[2]
    if n_areas_hint is not None and n_areas_hint != trace_n_areas:
        print(f'  Note: --n-areas={n_areas_hint} ignored; '
              f'traces have {trace_n_areas} areas.')
    print('  Warning: trace has no area coordinates — data may not match trace LSOAs.')
    return make_data_dict(gdf, n_areas=trace_n_areas)


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
    gdf_sample = select_spatial_sample(gdf, n_areas=args.n_areas or 200)
    data = make_data_dict(gdf_sample)

    print(f'   {data["n_areas"]} LSOAs, {data["n_years"]} inference years')

    os.makedirs(args.traces_dir, exist_ok=True)

    for name in model_names:
        ModelClass = _ALL_MODELS[name]
        # M7 requires borough_idx — skip unless user provides borough data
        if name == 'M7' and 'borough_idx' not in data:
            print('\n  [skip] M7 requires borough_idx in data dict. '
                  'Add it manually and run M7 separately.')
            continue

        print(f'\n── Sampling {name}: {ModelClass.description} ──')
        m = ModelClass(data)
        m.build()
        m.sample(use_nutpie=not args.no_nutpie)
        m.save(results_dir=args.traces_dir)
        print(f'  {name} saved.')

    # Invalidate comparison cache since traces have changed
    for p in [Path(args.traces_dir) / _COMPARISON_CSV,
              Path(args.traces_dir) / _COMPARISON_META]:
        p.unlink(missing_ok=True)

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
    comparison = _load_comparison_cache(args.traces_dir)
    if comparison is None:
        comparison = compute_model_comparison(traces, verbose=True)
        _save_comparison_cache(args.traces_dir, comparison, list(traces))
    else:
        display_cols = [c for c in ('elpd', 'se', 'p', 'elpd_diff', 'weight')
                        if c in comparison.columns]
        print(comparison[display_cols].to_string())
        print(f'\n  Best model: {comparison.index[0]}')

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

    data = None
    if hasattr(args, 'data_path') and args.data_path:
        try:
            validate_data_path(args.data_path)
            gdf  = load_data(args.data_path)
            gdf, _ = apply_outlier_exclusion(gdf)
            data = _data_matching_traces(gdf, traces)
        except Exception as exc:  # noqa: BLE001
            print(f'\n  Warning: could not load data for coverage ({exc}), skipping.')

    print('\n── Sampling diagnostics ─────────────────────────────────────────')
    diag = diagnostics_summary(traces, data=data)
    fmt  = {'max_rhat': '{:.4f}'.format, 'divergences': '{:d}'.format}
    if 'plan_cov_90' in diag.columns:
        fmt['plan_cov_90'] = '{:.3f}'.format
        fmt['ben_cov_90']  = '{:.3f}'.format
    print(diag.to_string(formatters={k: fmt[k] for k in fmt if k in diag.columns}))
    n_bad_rhat = int((diag['max_rhat'] > 1.01).sum())
    n_divs     = int(diag['divergences'].sum())
    if n_bad_rhat:
        print(f'\n  *** {n_bad_rhat} model(s) with max r-hat > 1.01 ***')
    if n_divs:
        print(f'  *** {n_divs} total divergences across all models ***')

    return comparison


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args):
    print(f'\n── Loading data from {args.data_path} ───────────────────────────')
    validate_data_path(args.data_path)
    gdf = load_data(args.data_path)
    gdf, _ = apply_outlier_exclusion(gdf)

    model_names = (_parse_model_list(args.models) if args.models
                   else _discover_traces(args.traces_dir))
    if not model_names:
        print(f'No traces found in {args.traces_dir}. '
              f'Run `housing-projections run-models` first.', file=sys.stderr)
        sys.exit(1)

    print('\n── Loading traces ───────────────────────────────────────────────')
    traces = _load_traces(args.traces_dir, model_names)

    # Build data dict matching the n_areas the traces were actually sampled on
    data = _data_matching_traces(gdf, traces, n_areas_hint=args.n_areas)
    print(f'   {data["n_areas"]} LSOAs, {data["n_years"]} inference years')

    # Load or compute LOO comparison (shared cache with `compare` command)
    comparison_df = None
    if len(traces) > 1:
        comparison_df = _load_comparison_cache(args.traces_dir)
        if comparison_df is None:
            print('  Computing LOO comparison (this may take a while)...')
            try:
                comparison_df = compute_model_comparison(traces, verbose=False)
                _save_comparison_cache(args.traces_dir, comparison_df, list(traces))
            except Exception as exc:  # noqa: BLE001
                print(f'  Warning: LOO comparison failed ({exc}), skipping.')

    model_classes = {name: _ALL_MODELS[name] for name in traces if name in _ALL_MODELS}

    print(f'\n── Generating report → {args.output} ────────────────────────────')
    os.makedirs(Path(args.output).parent, exist_ok=True)
    generate_report(
        data=data,
        traces=traces,
        model_classes=model_classes,
        output_path=args.output,
        title=args.title,
        comparison_df=comparison_df,
    )
    print(f'\nReport written to {args.output}')

    # Export uncertainty CSV alongside the report
    if len(traces) > 1 and comparison_df is not None:
        lsoa_codes = data['gdf']['LSOA21CD'].values if 'LSOA21CD' in data['gdf'].columns else None
        unc_df = compute_decomposed_uncertainty(
            traces, comparison_df=comparison_df, lsoa_codes=lsoa_codes,
        )
        csv_path = Path(args.output).with_suffix('.uncertainty.csv')
        unc_df.to_csv(csv_path, index=False)
        print(f'Uncertainty estimates written to {csv_path}')


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(
        prog='housing-projections',
        description='Bayesian dwelling projection tools for London LSOAs.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── run-models ──────────────────────────────────────────────────────────
    p_run = sub.add_parser('run-models', help='Sample models and save traces.')
    p_run.add_argument('--data-path', default='data',
                       help='Root directory of raw data files (default: data).')
    p_run.add_argument('--models', default=None,
                       help='Comma-separated model names, e.g. M0,M1,M3 (default: all).')
    p_run.add_argument('--n-areas', type=int, default=None,
                       help='Subsample to N LSOAs for faster runs.')
    p_run.add_argument('--traces-dir', default='results/traces',
                       help='Directory to save .nc trace files (default: results/traces).')
    p_run.add_argument('--no-nutpie', action='store_true',
                       help='Disable nutpie sampler, fall back to PyMC.')

    # ── compare ─────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser('compare', help='LOO comparison and z sensitivity report.')
    p_cmp.add_argument('--data-path', default='data',
                       help='Root directory of raw data files — used for coverage diagnostics (default: data).')
    p_cmp.add_argument('--traces-dir', default='results/traces',
                       help='Directory containing saved .nc trace files.')
    p_cmp.add_argument('--models', default=None,
                       help='Comma-separated model names (default: all found in traces-dir).')

    # ── report ──────────────────────────────────────────────────────────────
    p_rep = sub.add_parser('report', help='Generate self-contained HTML analysis report.')
    p_rep.add_argument('--data-path', default='data',
                       help='Root directory of raw data files (default: data).')
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
