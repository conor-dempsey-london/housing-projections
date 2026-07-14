"""
Full-dataset trace characterization — reusable methodology.

Loads one model's full-scale trace (e.g. results/traces_full/AZ3.nc, 4987 areas)
exactly once, computes every summary table this report needs, and writes them as
CSVs plus pre-rendered PNGs to --output-dir. The trace itself is only ever read by
this script — regenerating the HTML report afterwards (e.g. after a wording or
layout tweak) should use build_report.py against the CSVs/PNGs already written
here, not re-run this script. See docs/az3-full-dataset-report-method.md.

Usage
-----
    pixi run python scripts/full_dataset_characterization.py \\
        --trace-path results/traces_full/AZ3.nc \\
        --output-dir results/artifacts/az3_full_characterization \\
        --model-name AZ3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from housing_projections.config import DATA_PATH  # noqa: E402
from housing_projections.data import (  # noqa: E402
    load_data,
    make_borough_idx,
    make_data_dict,
)
from housing_projections.diagnostics import (  # noqa: E402
    _check_calibration,
    _check_census_constraint,
    _check_morans_i,
    detect_z_multimodality,
    z_flatness_summary,
    z_identifiability_summary,
)
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402
from housing_projections.plots.core import (  # noqa: E402
    REFERENCE_AREAS,
    select_spike_tracking_areas,
)
from housing_projections.spatial import (  # noqa: E402
    build_weights_libpysal,
    compute_morans_i_by_year,
)


# -- Trace loading (skips warmup groups — halves memory footprint) ------------─

def load_trace_no_warmup(trace_path):
    """
    Load a saved trace via az.from_netcdf (a thin wrapper around
    xr.open_datatree in this arviz version). Every group, including
    `warmup_posterior`/`warmup_sample_stats`, is opened lazily — the backend
    only reads a variable's bytes off disk when something actually calls
    `.values`/`.compute()` on it, so as long as this module never touches the
    warmup groups (it doesn't — no diagnostic here needs them), their ~50%
    memory/time cost is never paid despite `from_netcdf` "loading" them.
    Kept as a named wrapper (rather than inlining az.from_netcdf everywhere)
    so that guarantee is documented once, next to where it matters most (the
    24GB+ full-dataset trace).
    """
    return az.from_netcdf(trace_path)


def data_matching_trace(gdf, trace):
    """Subset/reorder gdf to the LSOA order embedded in the trace's 'area' coord."""
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    subset = gdf[gdf['LSOA21CD'].isin(lsoa_codes)].copy()
    subset = subset.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
    return make_data_dict(subset)


# -- Per-area / per-cell summaries ----------------------------------------------

def compute_resp_noise_summary(trace, data):
    """
    Per-area mean resp_noise on active P/E cells, and the fraction of
    active cells classified >0.5 noise-probability — the area-level
    aggregation used in the AZ3 Phase 3 cross-area pattern check
    (docs/az-family-work-plan.md), reused here at full scale.
    Returns None for any resp_noise_* variable missing from this trace
    (the noise mixture is AZ3-specific machinery).
    """
    if 'resp_noise_P' not in trace.posterior:
        return None

    P_obs, E_obs = data['P_obs'], data['E_obs']
    resp_P = trace.posterior['resp_noise_P'].mean(dim=('chain', 'draw')).values
    resp_E = trace.posterior['resp_noise_E'].mean(dim=('chain', 'draw')).values

    active_P = np.abs(P_obs) > 3.0
    active_E = np.abs(E_obs) > 3.0

    def _masked_mean(arr, mask):
        out = np.full(arr.shape[0], np.nan)
        any_active = mask.any(axis=1)
        out[any_active] = np.array([
            arr[i, mask[i]].mean() for i in np.where(any_active)[0]
        ])
        return out

    def _masked_frac_high(arr, mask):
        out = np.full(arr.shape[0], np.nan)
        any_active = mask.any(axis=1)
        out[any_active] = np.array([
            (arr[i, mask[i]] > 0.5).mean() for i in np.where(any_active)[0]
        ])
        return out

    return pd.DataFrame({
        'mean_resp_noise_P':      _masked_mean(resp_P, active_P),
        'mean_resp_noise_E':      _masked_mean(resp_E, active_E),
        'frac_active_P_high_noise': _masked_frac_high(resp_P, active_P),
        'frac_active_E_high_noise': _masked_frac_high(resp_E, active_E),
    })


def compute_pareto_k_summary(trace, n_areas, n_years):
    """Per-area max Pareto-k for P_like/E_like, if the trace has a usable
    log_likelihood group. Returns None otherwise (mirrors
    select_spike_tracking_areas' own graceful-degradation pattern)."""
    if '/log_likelihood' not in trace.groups or 'P_like' not in trace.log_likelihood:
        return None
    out = {}
    for src, var in [('P', 'P_like'), ('E', 'E_like')]:
        try:
            loo = az.loo(trace, var_name=var, pointwise=True)
            k = np.asarray(loo.pareto_k.values).reshape(n_areas, n_years)
            out[f'max_pareto_k_{src}'] = k.max(axis=1)
            out[f'frac_bad_k_{src}'] = (k > 0.7).mean(axis=1)
        except Exception as exc:  # noqa: BLE001 — Pareto-k is a bonus diagnostic
            print(f'  Pareto-k for {var} unavailable: {exc}')
    return pd.DataFrame(out) if out else None


def compute_multimodality_summary(trace, thin):
    """
    Full (area, year) multimodality scan via detect_z_multimodality, thinned
    to keep runtime tractable at 4987 areas x 10 years (49,870 KDE fits).
    Thinning draws (not chains) trades some resolution in each cell's KDE for
    a large constant-factor speedup; validated at the 200-area scale in
    az-family-work-plan.md Phase 3 with the full draw count, so this is a
    scale-driven compute tradeoff, not an untested method.
    """
    thinned = type('T', (), {})()
    thinned.posterior = trace.posterior.isel(draw=slice(None, None, thin))
    return detect_z_multimodality(thinned)


# -- Deep-dive example selection ------------------------------------------------

def select_deep_dive_examples(trace, data, area_df, n_examples=6):
    """
    Contrasting example areas for the deep-dive section: reuse
    select_spike_tracking_areas' category-based selection (under-tracked,
    Pareto-k worst, disagreeing spike years, well-tracked) plus REFERENCE_AREAS,
    then add explicitly multimodality-driven contrasts not covered by that
    selector: the most-multimodal area, a confident (zero-multimodal-year)
    heavily-noise-flagged area, and the area with the single worst per-cell
    Pareto-k in this full run specifically.
    """
    selected = select_spike_tracking_areas(trace, data, n_examples=n_examples,
                                            reference_areas=REFERENCE_AREAS)
    used_codes = {area_df.iloc[idx]['area'] for idx, _ in selected}

    if 'n_multimodal_years' in area_df.columns:
        most_multimodal = area_df.sort_values('n_multimodal_years', ascending=False)
        for _, row in most_multimodal.iterrows():
            if row['area'] not in used_codes and row['n_multimodal_years'] >= 2:
                idx = int(row.name)
                selected.append((idx, f"most multimodal: {int(row['n_multimodal_years'])} "
                                       f"of 10 years show >=2 z modes"))
                used_codes.add(row['area'])
                break

    if 'mean_resp_noise_P' in area_df.columns:
        confident_noisy = area_df[
            (area_df.get('n_multimodal_years', 0) == 0) &
            (area_df['mean_resp_noise_P'] > 0.6)
        ].sort_values('mean_resp_noise_P', ascending=False)
        for _, row in confident_noisy.iterrows():
            if row['area'] not in used_codes:
                idx = int(row.name)
                selected.append((idx, f"unimodal but heavily noise-flagged: "
                                       f"mean resp_noise_P={row['mean_resp_noise_P']:.2f}"))
                used_codes.add(row['area'])
                break

    return selected


def area_timeseries_frame(trace, data, area_idx, area_code, reason):
    """Per-year data for one example area: raw P/E obs, z posterior mean+90%CI,
    resp_noise means — everything build_report.py needs to redraw this area's
    panel without touching the trace."""
    z = trace.posterior['z'].isel(area=area_idx).values  # (chain, draw, year)
    z_flat = z.reshape(-1, z.shape[-1])
    n_years = z_flat.shape[-1]

    rows = {
        'area':        area_code,
        'reason':      reason,
        'year_idx':    np.arange(n_years),
        'P_obs':       data['P_obs'][area_idx],
        'E_obs':       data['E_obs'][area_idx],
        'z_mean':      z_flat.mean(axis=0),
        'z_lo':        np.percentile(z_flat, 5, axis=0),
        'z_hi':        np.percentile(z_flat, 95, axis=0),
    }
    if 'resp_noise_P' in trace.posterior:
        rows['resp_noise_P'] = trace.posterior['resp_noise_P'].isel(
            area=area_idx).mean(dim=('chain', 'draw')).values
        rows['resp_noise_E'] = trace.posterior['resp_noise_E'].isel(
            area=area_idx).mean(dim=('chain', 'draw')).values
    return pd.DataFrame(rows)


def plot_mode_decomposition_if_available(trace, area_idx, area_code, output_dir):
    """Save a static PNG of the whole-draw mode decomposition (plot_z_area_modes)
    for one example area, if that area has any multimodal year. Kept as a
    pre-rendered image (not CSV data) since it needs the full per-draw z array,
    which is deliberately not carried into the CSVs."""
    try:
        from housing_projections.plots.core import plot_z_area_modes
    except ImportError:
        return None
    z_post = trace.posterior['z'].values
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_z_area_modes(ax, z_post, area_idx)
    ax.set_title(f'{area_code} — mode decomposition')
    fname = output_dir / f'mode_decomposition_{area_code}.png'
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fname.name


# -- Main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace-path', default='results/traces_full/AZ3.nc')
    ap.add_argument('--data-path', default=str(DATA_PATH) if DATA_PATH else 'data')
    ap.add_argument('--output-dir', default='results/artifacts/az3_full_characterization')
    ap.add_argument('--model-name', default='AZ3')
    ap.add_argument('--thin-multimodality', type=int, default=4,
                     help='keep every Nth draw for the full-scale multimodality scan')
    ap.add_argument('--n-example-areas', type=int, default=6)
    ap.add_argument('--skip-multimodality', action='store_true')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f'-- Loading trace: {args.trace_path} --')
    trace = load_trace_no_warmup(args.trace_path)
    print(f'   loaded in {time.time() - t0:.0f}s, '
          f'z shape {trace.posterior["z"].shape}')

    print(f'-- Loading data: {args.data_path} --')
    gdf = load_data(args.data_path)
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
    data = data_matching_trace(gdf, trace)
    n_areas, n_years = data['n_areas'], data['n_years']
    print(f'   {n_areas} areas x {n_years} years matched to trace')

    borough_idx, n_boroughs, borough_codes = make_borough_idx(data['gdf'])
    data['borough_idx'], data['n_boroughs'], data['borough_codes'] = \
        borough_idx, n_boroughs, borough_codes

    # -- Area-level summary --------------------------------------------------
    print('-- z flatness / identifiability --')
    flatness = z_flatness_summary(trace, data)
    identifiability = z_identifiability_summary(trace)

    area_df = flatness.merge(identifiability, on='area', suffixes=('', '_ident'))
    area_df['borough'] = borough_codes[borough_idx]
    area_df['D'] = data['D']

    print('-- resp_noise summary --')
    resp_noise_df = compute_resp_noise_summary(trace, data)
    if resp_noise_df is not None:
        area_df = pd.concat([area_df, resp_noise_df], axis=1)

    print('-- Pareto-k summary --')
    pareto_df = compute_pareto_k_summary(trace, n_areas, n_years)
    if pareto_df is not None:
        area_df = pd.concat([area_df, pareto_df], axis=1)

    multimodal_cells = pd.DataFrame()
    if not args.skip_multimodality:
        print(f'-- Multimodality scan (thin={args.thin_multimodality}) --')
        tm0 = time.time()
        mm = compute_multimodality_summary(trace, args.thin_multimodality)
        print(f'   scanned {len(mm)} cells in {time.time() - tm0:.0f}s')
        n_multimodal_per_area = (
            mm[mm['n_modes'] >= 2].groupby('area_idx').size()
            .reindex(range(n_areas), fill_value=0)
        )
        area_df['n_multimodal_years'] = n_multimodal_per_area.values
        multimodal_cells = mm[mm['n_modes'] >= 2].copy()

    area_df.to_csv(output_dir / 'area_summary.csv', index=False)
    multimodal_cells.to_csv(output_dir / 'multimodal_cells.csv', index=False)
    print(f'   wrote area_summary.csv ({len(area_df)} rows), '
          f'multimodal_cells.csv ({len(multimodal_cells)} rows)')

    # -- Borough-level summary ----------------------------------------------─
    print('-- Borough summary --')
    agg = {'flat_despite_active': 'mean', 'is_flat': 'mean', 'has_active_year': 'mean',
           'confident': 'mean', 'D': 'mean'}
    for col in ['mean_resp_noise_P', 'mean_resp_noise_E', 'n_multimodal_years',
                'max_pareto_k_P', 'max_pareto_k_E']:
        if col in area_df.columns:
            agg[col] = 'mean'
    borough_df = area_df.groupby('borough').agg(agg)
    borough_df['n_areas'] = area_df.groupby('borough').size()
    borough_df = borough_df.reset_index()
    borough_df.to_csv(output_dir / 'borough_summary.csv', index=False)
    print(f'   wrote borough_summary.csv ({len(borough_df)} boroughs)')

    # -- Scalar / whole-run summary ------------------------------------------
    print('-- Scalar diagnostics (calibration, census constraint, Moran\'s I) --')
    calibration = _check_calibration(trace, data)
    census = _check_census_constraint(trace, data)
    morans = _check_morans_i(trace, data)

    scalar_summary = {
        'model_name': args.model_name,
        'n_areas': n_areas,
        'n_years': n_years,
        'frac_flat_despite_active': float(flatness.attrs['summary']['frac_flat_despite_active']),
        'frac_flat': float(flatness.attrs['summary']['frac_flat']),
        'frac_active': float(flatness.attrs['summary']['frac_active']),
        'frac_low_year_confidence': float((~identifiability['confident']).mean()),
        'max_rhat_z': float(np.nanmax(area_df['max_rhat'])),
        'plan_coverage_90': calibration['planning'],
        'ben_coverage_90': calibration['ben'],
        'census_mean_violation': census['mean_violation'],
        'census_max_violation': census['max_violation'],
        'morans_i_planning_residual': morans['planning']['I'],
        'morans_i_planning_p': morans['planning']['p_value'],
        'morans_i_ben_residual': morans['ben']['I'],
        'morans_i_ben_p': morans['ben']['p_value'],
    }
    for scalar in ['rho_P', 'rho_E', 'sigma_plan', 'sigma_ben',
                   'sigma_noise_P', 'sigma_noise_E']:
        if scalar in trace.posterior:
            vals = trace.posterior[scalar].values
            scalar_summary[f'{scalar}_mean'] = float(vals.mean())
            scalar_summary[f'{scalar}_sd'] = float(vals.std())
    if not args.skip_multimodality:
        scalar_summary['frac_areas_any_multimodal_year'] = float(
            (area_df['n_multimodal_years'] > 0).mean())
        scalar_summary['frac_cells_multimodal'] = float(
            len(multimodal_cells) / (n_areas * n_years))

    (output_dir / 'scalar_summary.json').write_text(json.dumps(scalar_summary, indent=2))
    print('   wrote scalar_summary.json')

    # -- Spatial: Moran's I by year on resp_noise (spatial clustering of ambiguity) --
    if resp_noise_df is not None:
        print('-- Moran\'s I by year on resp_noise --')
        w = build_weights_libpysal(data['gdf'])
        resp_P_by_year = trace.posterior['resp_noise_P'].mean(dim=('chain', 'draw')).values
        morans_by_year = compute_morans_i_by_year(resp_P_by_year, w)
        morans_by_year['year'] = range(n_years)
        morans_by_year.to_csv(output_dir / 'morans_i_resp_noise_by_year.csv', index=False)
        print('   wrote morans_i_resp_noise_by_year.csv')

    # -- Deep-dive examples --------------------------------------------------
    print('-- Selecting deep-dive example areas --')
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    area_df_indexed = area_df.copy()
    area_df_indexed.index = range(len(area_df_indexed))
    examples = select_deep_dive_examples(trace, data, area_df_indexed,
                                          n_examples=args.n_example_areas)

    example_meta, example_ts = [], []
    for idx, reason in examples:
        code = lsoa_codes[idx]
        example_meta.append({'area_idx': idx, 'area': code, 'reason': reason})
        example_ts.append(area_timeseries_frame(trace, data, idx, code, reason))
        plot_mode_decomposition_if_available(trace, idx, code, output_dir)

    pd.DataFrame(example_meta).to_csv(output_dir / 'example_areas.csv', index=False)
    pd.concat(example_ts, ignore_index=True).to_csv(
        output_dir / 'example_areas_timeseries.csv', index=False)
    print(f'   wrote example_areas.csv / example_areas_timeseries.csv '
          f'({len(example_meta)} examples)')

    manifest = {
        'model_name': args.model_name,
        'trace_path': str(args.trace_path),
        'data_path': str(args.data_path),
        'thin_multimodality': args.thin_multimodality if not args.skip_multimodality else None,
        'files': sorted(p.name for p in output_dir.iterdir()),
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f'\nDone in {time.time() - t0:.0f}s. Building report...')

    # build_report.py's HTML-assembly logic lives once, at the canonical
    # artifact location — load it from there regardless of --output-dir so
    # e.g. a scratch/smoke-test run still exercises the real report builder.
    import importlib.util
    report_module_path = (Path(__file__).resolve().parent.parent /
                           'results' / 'artifacts' / 'az3_full_characterization' /
                           'build_report.py')
    spec = importlib.util.spec_from_file_location('build_report', report_module_path)
    build_report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_report)
    build_report.build_report(output_dir)
    print(f'Report written to {output_dir / "report.html"}')


if __name__ == '__main__':
    main()
