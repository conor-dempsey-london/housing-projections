"""
housing-projections CLI

Commands
--------
  run-models           Sample one or more models and save traces.
  compare              Load saved traces and run LOO + sensitivity comparison.
  diagnose             Quick per-model sampling diagnostics (r-hat/ESS/coverage).
                       --adjust-for-multimodality excludes genuine, expected per-area
                       multimodality (see check-multimodality) from the headline r-hat/ESS.
  check-multimodality  Classify per-area lag-category multimodality and report
                       adjusted r-hat/ESS (see docs/multimodality-diagnostic-pipeline.md).
  report               Generate a self-contained HTML analysis report.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import arviz as az
import pandas as pd

from housing_projections.analysis import compute_model_comparison
from housing_projections.data import (
    load_data,
    make_borough_idx,
    make_data_dict,
    select_spatial_sample,
    validate_data_path,
)
from housing_projections.diagnostics import diagnostics_summary, z_identifiability_summary
from housing_projections.html_report import generate_report
from housing_projections.models import ALL_MODELS
from housing_projections.multimodality import (
    adjusted_diagnostics_summary,
    classify_multimodality,
    derive_loglik_var,
    lag_vars_in_trace,
    multimodality_report,
    resolve_stuck_areas,
    verify_resolution,
)
from housing_projections.outliers import apply_outlier_exclusion
from housing_projections.sensitivity import (
    compute_decomposed_uncertainty,
    compute_model_agreement_matrix,
    compute_z_model_sensitivity,
)

_ALL_MODELS = ALL_MODELS

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


def _var_names_by_model(traces, data):
    """
    Each model's own scalar `var_names` (shared by `diagnose` and `check-multimodality`) —
    restricts r-hat/ESS/multimodality scanning to that short, curated list instead of every
    posterior variable (z, delta, resp_*, *_pointwise, ...). See diagnostics_summary's own
    docstring: unrestricted, this is ~98% of diagnose's runtime for even a couple of small
    models — and, now, also the scope classify_scalar_multimodality would otherwise scan.
    Models not found in _ALL_MODELS, or whose var_names needs data to resolve (e.g. M5) and
    no data was loaded, are silently omitted — callers fall back to an unrestricted scope for
    those specific models (see adjusted_diagnostics_summary's own var_names=None handling).
    """
    var_names_by_model = {}
    for name in traces:
        ModelClass = _ALL_MODELS.get(name)
        if ModelClass is None:
            continue
        vn = ModelClass.var_names
        if isinstance(vn, property):  # e.g. M5 — depends on an instance attribute
            if data is None:
                continue
            try:
                vn = ModelClass(data).var_names
            except Exception:  # noqa: BLE001 — fall back to unrestricted for this model
                continue
        var_names_by_model[name] = vn
    return var_names_by_model


def _load_resolved_traces(traces_dir, model_names):
    """Load `{name}_resolved.nc` traces saved by `check-multimodality --resolve`, for
    every model in model_names that has one. Silently skips models with none — most
    won't, since --resolve is only run for models check-multimodality flagged."""
    resolved = {}
    for name in model_names:
        path = Path(traces_dir) / f'{name}_resolved.nc'
        if path.exists():
            print(f'  Loading resolved trace for {name} from {path}')
            resolved[name] = az.from_netcdf(str(path))
    return resolved


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

    borough_idx, n_boroughs, borough_codes = make_borough_idx(data['gdf'])
    data['borough_idx']   = borough_idx
    data['n_boroughs']    = n_boroughs
    data['borough_codes'] = borough_codes

    print(f'   {data["n_areas"]} LSOAs, {data["n_years"]} inference years, '
          f'{n_boroughs} boroughs')

    os.makedirs(args.traces_dir, exist_ok=True)

    for name in model_names:
        ModelClass = _ALL_MODELS[name]

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

    return comparison


# ── diagnose ──────────────────────────────────────────────────────────────────

def cmd_diagnose(args):
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

    data = None
    try:
        validate_data_path(args.data_path)
        gdf  = load_data(args.data_path)
        gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
        data = _data_matching_traces(gdf, traces)
    except Exception as exc:  # noqa: BLE001
        print(f'  Warning: could not load data ({exc}), skipping coverage.')

    var_names_by_model = _var_names_by_model(traces, data)

    print(f'\n── Diagnostics ({len(traces)} model(s)) ──────────────────────────')
    if args.adjust_for_multimodality:
        resolved_traces = _load_resolved_traces(args.traces_dir, list(traces))
        diag = adjusted_diagnostics_summary(
            traces, resolved_traces=resolved_traces, data=data,
            var_names=var_names_by_model, rhat_threshold=args.rhat_threshold,
            exclude_reviewed=args.exclude_reviewed)
    else:
        diag = diagnostics_summary(traces, data=data, rhat_threshold=args.rhat_threshold,
                                   var_names=var_names_by_model)

    col_fmts = {
        'max_rhat':       '{:.4f}'.format,
        'mean_rhat':      '{:.4f}'.format,
        'n_bad_rhat':     '{:d}'.format,
        'divergences':    '{:d}'.format,
        'min_ess':        '{:d}'.format,
        'flagged_chains': '{:d}'.format,
    }
    if 'n_lambda_weights_vars' in diag.columns:
        col_fmts['n_lambda_weights_vars']  = '{:d}'.format
        col_fmts['n_flagged_multimodal']   = '{:d}'.format
        col_fmts['n_resolved']             = '{:d}'.format
        col_fmts['n_needs_deep_dive']      = '{:d}'.format
        col_fmts['n_not_multimodal']       = '{:d}'.format
        col_fmts['best_case_max_rhat'] = lambda x: 'n/a' if pd.isna(x) else f'{x:.4f}'
        col_fmts['best_case_min_ess']  = lambda x: 'n/a' if pd.isna(x) else f'{x:.0f}'
    if 'plan_cov_90' in diag.columns:
        col_fmts['plan_cov_90'] = '{:.3f}'.format
        col_fmts['ben_cov_90']  = '{:.3f}'.format
    if 'frac_flat_despite_active' in diag.columns:
        col_fmts['frac_flat_despite_active'] = '{:.1%}'.format

    # raw_max_rhat/raw_min_ess are kept in `diag` for the per-model verdict below (which
    # needs them), but dropped from the printed table — max_rhat/min_ess above already ARE
    # the adjusted numbers, so showing raw alongside would just duplicate what the verdict
    # spells out per model.
    display_diag = diag.drop(columns=['raw_max_rhat', 'raw_min_ess'], errors='ignore')
    print(display_diag.to_string(
        formatters={k: col_fmts[k] for k in col_fmts if k in display_diag.columns}))

    n_bad     = int((diag['max_rhat'] > args.rhat_threshold).sum())
    n_divs    = int(diag['divergences'].sum())
    n_flagged = int(diag['flagged_chains'].sum())
    if 'frac_flat_despite_active' in diag.columns:
        flat_models = diag.index[diag['frac_flat_despite_active'] > 0.10].tolist()
        if flat_models:
            print(f"\n  *** z is flat despite active P/E signal in >10% of areas for: "
                  f"{flat_models} — see z_flatness_summary() per-area detail ***")
    print()
    if n_bad:
        bad_models = diag.index[diag['max_rhat'] > args.rhat_threshold].tolist()
        print(f'  *** {n_bad} model(s) with max r-hat > {args.rhat_threshold}: '
              f'{", ".join(bad_models)} ***')
    else:
        print(f'  All models have max r-hat ≤ {args.rhat_threshold}')
    if n_divs:
        print(f'  *** {n_divs} total divergences ***')
    else:
        print('  No divergences.')
    if n_flagged:
        flagged_models = diag.index[diag['flagged_chains'] > 0].tolist()
        print(f'  *** {n_flagged} chain(s) appear trapped in a distinct mode: '
              f'{", ".join(flagged_models)} — see chain agreement diagnostics ***')
    else:
        print('  No chains flagged as trapped in a distinct mode.')

    if 'n_flagged_multimodal' in diag.columns:
        n_flagged_mm  = int(diag['n_flagged_multimodal'].sum())
        n_deep_dive   = int(diag['n_needs_deep_dive'].sum())
        n_mm_resolved = int(diag['n_resolved'].sum())
        n_not_mm      = int(diag['n_not_multimodal'].sum()) if 'n_not_multimodal' in diag.columns else 0
        print(f'\n  Multimodality adjustment: {n_flagged_mm} area/cell finding(s) flagged '
              f'across {len(traces)} model(s), {n_mm_resolved} confirmed resolved from a '
              f'saved *_resolved.nc trace.')
        if n_not_mm:
            print(f'  *** {n_not_mm} scalar cell(s) have bad r-hat/ESS NOT attributable to '
                  f'multimodality at all — see `check-multimodality` for which, and '
                  f'ess-rhat-diagnostic-guide.md for how to investigate ***')
        if n_deep_dive:
            print(f'  *** {n_deep_dive} area(s) still need a manual deep-dive — see '
                  f'`check-multimodality` for per-area detail ***')
        else:
            print('  No areas need a manual deep-dive.')

        best_rhat = diag['best_case_max_rhat'].dropna()
        best_ess  = diag['best_case_min_ess'].dropna()
        if len(best_rhat):
            print(f'  Best case (excludes ALL flagged areas, every model): '
                  f'worst max r-hat={best_rhat.max():.4f}  worst min ESS={best_ess.min():.0f}'
                  + ('' if args.exclude_reviewed else
                     ' — a preview; the table above used --no-exclude-reviewed, so it still '
                     'includes reviewed-but-unproven areas'))
            if n_deep_dive:
                print(f'  *** not yet a justified number while {n_deep_dive} area(s) still '
                      f'need a deep-dive — see the docstring for adjusted_diagnostics_report ***')

        print('\n  Per-model verdict — did the best case actually improve on raw, and can '
              'multimodality be named as the cause?')
        for name in diag.index:
            raw_rhat  = diag.loc[name, 'raw_max_rhat']
            raw_ess   = diag.loc[name, 'raw_min_ess']
            best_rhat = diag.loc[name, 'best_case_max_rhat']
            best_ess  = diag.loc[name, 'best_case_min_ess']
            if pd.isna(raw_rhat):
                continue
            deep_dive = int(diag.loc[name, 'n_needs_deep_dive'])
            caveat = (f' — tentative while {deep_dive} area(s) still need a deep-dive'
                      if deep_dive else '')
            best_rhat_str = 'n/a' if pd.isna(best_rhat) else f'{best_rhat:.4f}'
            best_ess_str  = 'n/a' if pd.isna(best_ess)  else f'{best_ess:.0f}'
            # best_rhat can be NaN not because nothing was measured, but because EVERY cell
            # in the raw scope got excluded (100% attributable) — that counts as fully
            # explains, not a skip, since best_case shares raw's scope by construction.
            if raw_rhat <= args.rhat_threshold:
                print(f'    {name}: raw max r-hat already ≤ {args.rhat_threshold} — '
                      f'multimodality adjustment moot here')
            elif pd.isna(best_rhat) or best_rhat <= args.rhat_threshold:
                print(f'    {name}: multimodality DIRECTLY EXPLAINS the bad r-hat/ESS — best '
                      f'case resolves it (max r-hat {raw_rhat:.4f} -> {best_rhat_str}, min ESS '
                      f'{raw_ess:.0f} -> {best_ess_str}){caveat}')
            elif best_rhat < raw_rhat:
                print(f'    {name}: multimodality PARTIALLY explains it — best case improves '
                      f'max r-hat ({raw_rhat:.4f} -> {best_rhat_str}, min ESS {raw_ess:.0f} -> '
                      f'{best_ess_str}) but stays above {args.rhat_threshold} — the remainder is '
                      f'likely not multimodality, see ess-rhat-diagnostic-guide.md{caveat}')
            else:
                print(f'    {name}: *** multimodality does NOT explain the bad r-hat/ESS — best '
                      f'case is unchanged or worse (max r-hat {raw_rhat:.4f} -> {best_rhat:.4f}, '
                      f'min ESS {raw_ess:.0f} -> {best_ess:.0f}) after excluding every flagged '
                      f'cell; investigate via ess-rhat-diagnostic-guide.md, not this pipeline ***')

    print('\n── Per-area year-allocation confidence ───────────────────────────')
    for name, trace in traces.items():
        if 'z' not in trace.posterior:
            continue
        ident = z_identifiability_summary(trace, rhat_threshold=args.rhat_threshold)
        n_low = int((~ident['confident']).sum())
        pct   = 100 * n_low / len(ident) if len(ident) else 0.0
        print(f'  {name}: {n_low}/{len(ident)} areas ({pct:.1f}%) have low '
              f'confidence in which year(s) absorbed their change '
              f'(total change per area remains reliable — see '
              f'z_identifiability_summary() for per-area detail)')


# ── check-multimodality ───────────────────────────────────────────────────────

def cmd_check_multimodality(args):
    """
    Run the multimodality diagnostic pipeline (housing_projections.multimodality) for every
    model that has at least one `*_lambda_weights` variable OR a registered `var_names` list
    — see docs/multimodality-diagnostic-pipeline.md for the full walkthrough and category
    definitions. By default checks EVERY `*_lambda_weights` variable a model's trace has
    (planning and BEN independently, if both exist) PLUS every scalar in that model's own
    `var_names` (same convention `diagnose` itself uses), and combines them into ONE report —
    so raw/adjusted/best_case r-hat/ESS reflect the model's FULL diagnostic scope, the exact
    same scope `diagnose --adjust-for-multimodality` uses, not just whichever one lag var
    happened to be passed via --lag-var. Pass --lag-var to restrict the lag-var side to a
    specific subset. With --resolve, also attempts the validated informed-init fix for the
    resolvable (stuck_fixable) subset of ONE specified --lag-var (resolving means resampling
    once for a single lag hierarchy, so it can't default to "all" the way the report itself
    does — and it never applies to scalars, which have no automated resolution path).
    """
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

    gdf = None
    data = None
    try:
        validate_data_path(args.data_path)
        gdf = load_data(args.data_path)
        gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
        data = _data_matching_traces(gdf, traces)
    except Exception as exc:  # noqa: BLE001
        print(f'  Warning: could not load data ({exc}) — scalar var_names for models that '
              f'need it (e.g. M5) will be skipped, and --resolve will be unavailable.')

    var_names_by_model = _var_names_by_model(traces, data)

    requested_lag_vars = _parse_model_list(args.lag_var) if args.lag_var else None

    if args.resolve and (requested_lag_vars is None or len(requested_lag_vars) != 1):
        print('--resolve requires --lag-var to name exactly one variable (resolving '
              'resamples the model once, for one lag hierarchy at a time) — the report '
              'itself checks every lag var by default, but --resolve cannot.',
              file=sys.stderr)
        sys.exit(1)
    if args.resolve and gdf is None:
        print('--resolve requires --data-path to load successfully.', file=sys.stderr)
        sys.exit(1)

    applicable = {}
    for name, tr in traces.items():
        available = lag_vars_in_trace(tr)
        lag_vars = ([v for v in requested_lag_vars if v in available] if requested_lag_vars
                   else available)
        scalar_names = var_names_by_model.get(name)
        if lag_vars or scalar_names:
            applicable[name] = (tr, lag_vars, scalar_names)
    if not applicable:
        wanted = requested_lag_vars if requested_lag_vars else 'any *_lambda_weights variable'
        print(f"\nNone of {list(traces)} have {wanted}, or a registered var_names list, in "
              f"their posterior — nothing to check.")
        return

    for name, (trace, lag_vars, scalar_names) in applicable.items():
        checked_desc = ", ".join(lag_vars) if lag_vars else "no lag vars"
        if scalar_names:
            checked_desc += f" + {len(scalar_names)} scalar var(s)"
        print(f'\n── {name}: multimodality check on {checked_desc} ───────────────')

        resolved_trace = None
        if args.resolve:
            if not lag_vars:
                print(f'  Skipping --resolve for {name}: requested --lag-var not found here.')
            else:
                lag_var = lag_vars[0]
                ModelClass = _ALL_MODELS.get(name)
                if ModelClass is None:
                    print(f'  Skipping --resolve for {name}: not a registered model class.')
                else:
                    model_data = _data_matching_traces(gdf, {name: trace})
                    classification_df = classify_multimodality(
                        trace, lag_var, derive_loglik_var(lag_var),
                        rhat_threshold=args.rhat_threshold)
                    new_trace, seeded_areas = resolve_stuck_areas(
                        ModelClass, model_data, classification_df, lag_var,
                        chains=args.resolve_chains)
                    if new_trace is None:
                        print('  No stuck_fixable areas found — nothing to reseed.')
                    else:
                        verification = verify_resolution(new_trace, lag_var, seeded_areas)
                        n_resolved = int(verification['resolved'].sum())
                        print(f'  Reseeded {len(seeded_areas)} stuck_fixable area(s) with '
                              f'{args.resolve_chains} chains: {n_resolved}/{len(seeded_areas)} confirmed resolved.')
                        out_path = Path(args.traces_dir) / f'{name}_resolved.nc'
                        new_trace.to_netcdf(str(out_path))
                        print(f'  Reseeded trace saved to {out_path} (original {name}.nc left untouched).')
                        resolved_trace = new_trace

        report = multimodality_report(
            trace, lag_vars=lag_vars, var_names=scalar_names, rhat_threshold=args.rhat_threshold,
            resolved_trace=resolved_trace)

        if report['lag_vars_skipped']:
            print(f"  Note: {', '.join(report['lag_vars_skipped'])} found in the posterior "
                  f"but has no matching log_likelihood entry to classify against (see "
                  f"derive_loglik_var) — included unfiltered in every figure below rather "
                  f"than dropped.")

        print(f"\n  Flagged: {report['n_flagged']} area/cell finding(s) across "
              f"{report['n_areas_total']} area(s), {len(report['lag_vars_checked'])} lag "
              f"var(s) + {len(scalar_names) if scalar_names else 0} scalar var(s) checked")
        print(f"    hard_genuine (irreducible tie, no fix):       {report['n_hard_genuine']}")
        print(f"    stuck_fixable (fixable via reseeding):        {report['n_stuck_fixable']}"
              + (f"  ({report['n_resolved']} confirmed resolved this run)" if args.resolve else ""))
        print(f"    round_tripping (shallow/overlapping, benign): {report['n_round_tripping']}")
        print(f"    mixed (tie + stuck straggler, needs a call):  {report['n_mixed']}")
        print(f"    needs_review (doesn't cleanly classify):      {report['n_needs_review']}")
        if report['n_not_multimodal']:
            print(f"    not_multimodal (scalar-only — bad r-hat/ESS NOT attributable to "
                  f"multimodality at all): {report['n_not_multimodal']}")
        print(f"\n  *** {report['n_needs_deep_dive']} finding(s) still need a manual "
              f"deep-dive ***" if report['n_needs_deep_dive']
              else "\n  No findings need a manual deep-dive.")
        if report['n_not_multimodal']:
            print(f"  *** {report['n_not_multimodal']} scalar cell(s) have bad r-hat/ESS "
                  f"that ISN'T multimodality at all — see ess-rhat-diagnostic-guide.md, not "
                  f"this pipeline, for those ***")

        print(f"\n  Raw        max r-hat={report['raw_max_rhat']:.4f}  min ESS={report['raw_min_ess']:.0f}")
        print(f"  Adjusted   max r-hat={report['adjusted_max_rhat']:.4f}  min ESS={report['adjusted_min_ess']:.0f}"
              f"  (excludes hard_genuine/round_tripping cells — their disagreement is expected)")
        print(f"  Best case  max r-hat={report['best_case_max_rhat']:.4f}  min ESS={report['best_case_min_ess']:.0f}"
              f"  (excludes every flagged area/cell EXCEPT not_multimodal ones — the reading "
              f"for the rest of the model once everything attributable to multimodality has "
              f"been triaged; not yet justified while n_needs_deep_dive > 0 — see the "
              f"docstring). Scoped to *_lambda_weights cells UNION the {len(scalar_names) if scalar_names else 0} "
              f"scalar var(s) checked above — the SAME scope `raw` uses, so the two are "
              f"directly comparable.")

        if len(report['lag_vars_checked']) > 1:
            print("\n  Per-lag-var breakdown (P and E can have very different profiles for "
                  "the same model):")
            for lag_var, sub in report['by_lag_var'].items():
                print(f"    {lag_var}: {sub['n_flagged']} flagged  (hard_genuine="
                      f"{sub['n_hard_genuine']} stuck_fixable={sub['n_stuck_fixable']} "
                      f"round_tripping={sub['n_round_tripping']} mixed={sub['n_mixed']} "
                      f"needs_review={sub['n_needs_review']})  raw={sub['raw_max_rhat']:.4f}/"
                      f"{sub['raw_min_ess']:.0f}  best_case={sub['best_case_max_rhat']:.4f}/"
                      f"{sub['best_case_min_ess']:.0f}")

        if report['within_chain_ess']:
            print('\n  Within-chain ESS (each chain checked in isolation — confirms the '
                  'sampler itself is healthy even where cross-chain agreement is not '
                  'expected; keyed "lag_var:area"):')
            for key, ess_list in report['within_chain_ess'].items():
                print(f"    {key}: {[round(e, 0) for e in ess_list]}")

        if not report['scalar_classification_df'].empty:
            print('\n  Scalar/vector findings (var, cell — see classify_scalar_multimodality):')
            print(report['scalar_classification_df'].drop(columns=['gaps'])
                 .to_string(index=False))

        if not report['classification_df'].empty:
            print('\n  Per-area detail:')
            print(report['classification_df'].drop(columns=['gaps']).to_string(index=False))


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

    # Export per-area year-allocation confidence, one CSV per model with z.
    # Areas flagged low-confidence still have a reliable total change (from
    # the census constraint) but an unreliable year-by-year breakdown.
    for name, trace in traces.items():
        if 'z' not in trace.posterior:
            continue
        ident_df = z_identifiability_summary(trace)
        ident_path = Path(args.output).with_suffix(f'.{name}.identifiability.csv')
        ident_df.to_csv(ident_path, index=False)
        n_low = int((~ident_df['confident']).sum())
        print(f'{name} year-allocation confidence written to {ident_path} '
              f'({n_low}/{len(ident_df)} areas low-confidence)')


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
    p_cmp.add_argument('--traces-dir', default='results/traces',
                       help='Directory containing saved .nc trace files.')
    p_cmp.add_argument('--models', default=None,
                       help='Comma-separated model names (default: all found in traces-dir).')

    # ── diagnose ────────────────────────────────────────────────────────────
    p_diag = sub.add_parser('diagnose', help='Quick per-model sampling diagnostics.')
    p_diag.add_argument('--data-path', default='data',
                        help='Root directory of raw data files — used for coverage (default: data).')
    p_diag.add_argument('--traces-dir', default='results/traces',
                        help='Directory containing saved .nc trace files (default: results/traces).')
    p_diag.add_argument('--models', default=None,
                        help='Comma-separated model names (default: all found in traces-dir).')
    p_diag.add_argument('--rhat-threshold', type=float, default=1.01,
                        help='R-hat threshold for flagging bad convergence (default: 1.01).')
    p_diag.add_argument('--adjust-for-multimodality', action='store_true',
                        help='Classify per-area hierarchical lag-category ambiguity (see '
                             'docs/multimodality-diagnostic-pipeline.md) and exclude cells '
                             'whose bad r-hat/ESS is genuine, expected multimodality rather '
                             'than a real sampling problem from the reported max/mean '
                             'r-hat and min ESS. Picks up any {model}_resolved.nc saved by '
                             '`check-multimodality --resolve` in --traces-dir automatically.')
    p_diag.add_argument('--exclude-reviewed', action=argparse.BooleanOptionalAction, default=True,
                        help='With --adjust-for-multimodality, also exclude mixed/'
                             'needs_review/still-unresolved-stuck_fixable areas (default: '
                             'True) — you have already seen them via `check-multimodality`, '
                             'so the headline number should not keep being dragged down by '
                             'a status that is already tracked. Pass --no-exclude-reviewed '
                             'for a more conservative reading that only credits areas as '
                             'proven benign or proven fixed.')

    # ── check-multimodality ─────────────────────────────────────────────────
    p_mm = sub.add_parser('check-multimodality',
                          help='Classify per-area lag-category multimodality and report '
                               'adjusted r-hat/ESS (see docs/multimodality-diagnostic-pipeline.md).')
    p_mm.add_argument('--traces-dir', default='results/traces',
                      help='Directory containing saved .nc trace files (default: results/traces).')
    p_mm.add_argument('--models', default=None,
                      help='Comma-separated model names (default: all found in traces-dir).')
    p_mm.add_argument('--lag-var', default=None,
                      help='Comma-separated hierarchical lag-category simplex variable(s) '
                           'to check, e.g. lag_P_lambda_weights,lag_E_lambda_weights '
                           '(default: every *_lambda_weights variable found in each trace — '
                           'checking only one, when a model has both a planning and a BEN '
                           'lag hierarchy, silently misses the other\'s flagged areas). The '
                           'matching log-likelihood variable is derived automatically by '
                           'naming convention (derive_loglik_var) — lag_P_lambda_weights -> '
                           'P_like, etc.')
    p_mm.add_argument('--rhat-threshold', type=float, default=1.01,
                      help='R-hat threshold for flagging an area (default: 1.01).')
    p_mm.add_argument('--resolve', action='store_true',
                      help='Also attempt the informed-init reseeding fix for stuck_fixable '
                           'areas and verify whether it resolved them. Requires --data-path '
                           'and --lag-var naming EXACTLY ONE variable (resolving resamples '
                           'the model once, for one lag hierarchy at a time — it cannot '
                           'default to "every lag var" the way the report itself does), and '
                           're-samples the model (slow) — saves the result as '
                           '{model}_resolved.nc alongside the original trace, which is left '
                           'untouched.')
    p_mm.add_argument('--resolve-chains', type=int, default=16,
                      help='Chain count for the --resolve reseeded run (default: 16, '
                           'matching the validated AZ1d experiment).')
    p_mm.add_argument('--data-path', default='data',
                      help='Root directory of raw data files — only needed with --resolve '
                           '(default: data).')

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
    # Windows consoles default to a cp1252-family codec that can't encode the
    # box-drawing characters (e.g. '─') used throughout this package's output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'run-models':
        cmd_run_models(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'diagnose':
        cmd_diagnose(args)
    elif args.command == 'check-multimodality':
        cmd_check_multimodality(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)
