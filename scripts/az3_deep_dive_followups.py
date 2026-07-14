"""
One-time investigative follow-ups on AZ3's full-scale trace -- resolves the
"needs a trace read" items deferred in docs/az3-report-review-plan.md, plus
adds a handful of confident/active example areas requested in the second
review round. Opens results/traces_full/AZ3.nc exactly once, read-only (via
full_dataset_characterization's own load_trace_no_warmup wrapper), and reuses
area_summary.csv/example_areas.csv already on disk rather than recomputing
anything already cached there.

Tasks, in the order they run:
  1. E01004686 -- why does the report claim 2 modes but the plot show 1?
     Rerun plot_z_area_modes with min_cluster_frac=0 (nothing dropped) at
     n_clusters=2 and 3, print the actual scenario weights and the internal
     concentration diagnostic.
  2. E01002794 -- why does one mode dump mass into 2013 instead of the
     observed 2016/2020 spike years? Rerun at n_clusters=2/3/4, and compute
     pairwise draw-correlation among the 5 individually-flagged years to test
     whether they move together or independently across draws.
  3. Adaptive multimodality re-characterization -- for every area with >=3
     individually-flagged multimodal years (per-cell KDE scan), test k=2/3/4
     whole-draw clusters and record the largest k where every resulting
     cluster clears a minimum-size bar. Answers "which areas genuinely need
     more than 2 scenarios" at full scale, not just for 1-2 hand-picked
     examples. Written to mode_recharacterization.csv.
  4. Regenerate mode-decomposition PNGs for the existing multimodal example
     areas using the now-fixed plot_mode_decomposition_if_available (D/P/E
     passthrough, no title clobber, plain P/E markers).
  5. Select and add a few new "confident and active" example areas -- high
     activity, zero low-confidence years, zero multimodal years -- as a
     contrast to the mostly-ambiguous examples already in the report.

Run: pixi run python scripts/az3_deep_dive_followups.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_SCRIPTS_DIR))

import full_dataset_characterization as fdc  # noqa: E402
from housing_projections.data import load_data  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402
from housing_projections.plots.core import plot_z_area_modes  # noqa: E402

OUTPUT_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_full_characterization'
TRACE_PATH = _REPO_ROOT / 'results' / 'traces_full' / 'AZ3.nc'
DATA_PATH = str(fdc.DATA_PATH) if fdc.DATA_PATH else 'data'

THRESH_YEARS_FOR_RECHAR = 3   # areas with >= this many flagged years get re-clustered
MIN_CLUSTER_FRAC = 0.05       # a cluster must clear this to count as "real" here
N_NEW_CONFIDENT_EXAMPLES = 4


def investigate_cluster_drop(z_post, lsoa_codes, area_code, idx, k_values=(2, 3)):
    """Task 1/2 shared helper -- rerun the whole-draw decomposition with
    nothing dropped (min_cluster_frac=0) at several k, print weights/title."""
    print(f'\n-- {area_code} (idx={idx}) --')
    for k in k_values:
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_z_area_modes(ax, z_post, idx, n_clusters=k, min_cluster_frac=0.0,
                           show_legend=True, lsoa_codes=lsoa_codes)
        handles, labels = ax.get_legend_handles_labels()
        scenario_labels = [lbl for lbl in labels if lbl.startswith('Scenario')]
        print(f'  k={k}: {ax.get_title()}')
        print(f'  k={k}: scenario weights = {scenario_labels}')
        plt.close(fig)


def characterize_area_modes(z_area_flat, k_values=(2, 3, 4), min_frac=MIN_CLUSTER_FRAC):
    """Whole-draw k-means at several k; for each k report each cluster's
    weight and whether every cluster clears min_frac (a 'real' cluster, not
    a k-means sliver), plus the top-k-year concentration diagnostic
    (see plot_z_area_modes docstring) as a check against pure exchangeability."""
    std = z_area_flat.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    norm = (z_area_flat - z_area_flat.mean(axis=0)) / std_safe

    argmax_year = z_area_flat.argmax(axis=1)
    year_mass = np.bincount(argmax_year, minlength=z_area_flat.shape[1]) / len(z_area_flat)

    out = {}
    for k in k_values:
        rng = np.random.default_rng(0)
        _, labels = kmeans2(norm, k, minit='++', seed=rng)
        weights = np.sort([(labels == i).mean() for i in range(k)])[::-1]
        n_real = int((weights >= min_frac).sum())
        concentration = float(np.sort(year_mass)[::-1][:k].sum())
        out[k] = {
            'weights': weights.round(3).tolist(),
            'n_real_clusters': n_real,
            'concentration': concentration,
        }
    return out


def main():
    print(f'-- Loading trace (read-only): {TRACE_PATH} --')
    trace = fdc.load_trace_no_warmup(str(TRACE_PATH))
    print(f'   z shape {trace.posterior["z"].shape}')

    print(f'-- Loading data: {DATA_PATH} --')
    gdf = load_data(DATA_PATH)
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
    data = fdc.data_matching_trace(gdf, trace)
    n_years = data['n_years']
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    idx_by_code = {code: i for i, code in enumerate(lsoa_codes)}

    area_df = pd.read_csv(OUTPUT_DIR / 'area_summary.csv')
    mismatches = sum(1 for i, code in enumerate(area_df['area']) if idx_by_code.get(code) != i)
    print(f'   area_summary.csv row/trace-index correspondence check: {mismatches} mismatches '
          f'(0 expected)')

    z_post = trace.posterior['z'].values
    print(f'   z_post loaded, {z_post.nbytes / 1e9:.2f} GB')

    # ── Task 1: E01004686 -----------------------------------------------------
    print('\n== Task 1: E01004686 -- "two modes claimed, one shown" ==')
    investigate_cluster_drop(z_post, lsoa_codes, 'E01004686', idx_by_code['E01004686'])

    # ── Task 2: E01002794 ------------------------------------------------------
    print('\n== Task 2: E01002794 -- mode dumps mass into 2013, not 2016/2020 ==')
    idx2 = idx_by_code['E01002794']
    investigate_cluster_drop(z_post, lsoa_codes, 'E01002794', idx2, k_values=(2, 3, 4))

    flagged_years = [1, 4, 5, 8, 9]  # year_idx for 2013, 2016, 2017, 2020, 2021
    z_area_e01002794 = z_post[:, :, idx2, :].reshape(-1, n_years)
    corr = np.corrcoef(z_area_e01002794[:, flagged_years].T)
    corr_df = pd.DataFrame(corr, index=flagged_years, columns=flagged_years)
    print('  pairwise draw-correlation among the 5 flagged years '
          '(year_idx 1=2013, 4=2016, 5=2017, 8=2020, 9=2021):')
    print(corr_df.round(2).to_string())

    # ── Task 3: adaptive multimodality re-characterization ---------------------
    print(f'\n== Task 3: adaptive re-clustering for areas with '
          f'>={THRESH_YEARS_FOR_RECHAR} flagged years ==')
    candidates = area_df[area_df['n_multimodal_years'] >= THRESH_YEARS_FOR_RECHAR]
    print(f'  {len(candidates)} candidate areas')

    thin = 4
    z_thin = trace.posterior['z'].isel(draw=slice(None, None, thin)).values
    print(f'  thinned z loaded for recharacterization, {z_thin.nbytes / 1e9:.2f} GB')

    rows = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        code = row['area']
        idx_c = idx_by_code[code]
        z_flat = z_thin[:, :, idx_c, :].reshape(-1, n_years)
        res = characterize_area_modes(z_flat)
        best_k = 1
        for k in (2, 3, 4):
            if res[k]['n_real_clusters'] == k:
                best_k = k
        rows.append({
            'area': code, 'borough': row['borough'], 'D': row['D'],
            'n_multimodal_years': row['n_multimodal_years'],
            'best_k': best_k,
            'k2_weights': res[2]['weights'], 'k2_concentration': res[2]['concentration'],
            'k3_weights': res[3]['weights'], 'k3_concentration': res[3]['concentration'],
            'k4_weights': res[4]['weights'], 'k4_concentration': res[4]['concentration'],
        })
        if (i + 1) % 200 == 0:
            print(f'  ...{i + 1}/{len(candidates)}')
    del z_thin

    recharacterization_df = pd.DataFrame(rows)
    recharacterization_df.to_csv(OUTPUT_DIR / 'mode_recharacterization.csv', index=False)
    n_need_3plus = int((recharacterization_df['best_k'] >= 3).sum())
    print(f'  wrote mode_recharacterization.csv ({len(recharacterization_df)} rows); '
          f'{n_need_3plus} areas genuinely need 3+ scenarios')

    # ── Task 4: regenerate mode-decomposition PNGs for existing examples -------
    print('\n== Task 4: regenerating mode-decomposition PNGs (fixed function) ==')
    example_meta = pd.read_csv(OUTPUT_DIR / 'example_areas.csv')
    for _, row in example_meta.iterrows():
        code = row['area']
        idx_e = idx_by_code[code]
        n_modes = area_df.loc[area_df['area'] == code, 'n_multimodal_years'].iloc[0]
        if n_modes > 0:
            fdc.plot_mode_decomposition_if_available(trace, data, idx_e, code, OUTPUT_DIR,
                                                       z_post=z_post)
            print(f'  regenerated mode_decomposition_{code}.png ({int(n_modes)} multimodal years)')
        else:
            print(f'  skipped {code} (0 multimodal years, per report-review item 5)')

    # ── Task 5: new confident + active example areas ---------------------------
    print(f'\n== Task 5: selecting {N_NEW_CONFIDENT_EXAMPLES} confident + active example areas ==')
    P_obs, E_obs = data['P_obs'], data['E_obs']
    activity = np.maximum(np.abs(P_obs).max(axis=1), np.abs(E_obs).max(axis=1))
    activity_by_code = dict(zip(lsoa_codes, activity))
    area_df['activity'] = area_df['area'].map(activity_by_code)

    used_codes = set(example_meta['area'])
    confident_active = area_df[
        area_df['has_active_year']
        & (area_df['n_low_confidence_years'] == 0)
        & (area_df['n_multimodal_years'] == 0)
        & (~area_df['area'].isin(used_codes))
    ].sort_values('activity', ascending=False)

    new_examples = confident_active.head(N_NEW_CONFIDENT_EXAMPLES)
    print(new_examples[['area', 'borough', 'D', 'activity', 'max_rhat']].to_string())

    new_meta_rows, new_ts_rows = [], []
    for _, row in new_examples.iterrows():
        code = row['area']
        idx_n = idx_by_code[code]
        reason = (f'confident & active (for contrast): peak |P or E|={row["activity"]:.0f}, '
                  f'D={row["D"]:.0f}, 0 low-confidence years, 0 multimodal years')
        new_meta_rows.append({'area_idx': idx_n, 'area': code, 'reason': reason})
        new_ts_rows.append(fdc.area_timeseries_frame(trace, data, idx_n, code, reason))
        # confident & unimodal by construction -- no mode-decomposition PNG needed

    example_meta_full = pd.concat([example_meta, pd.DataFrame(new_meta_rows)], ignore_index=True)
    example_ts_full = pd.concat(
        [pd.read_csv(OUTPUT_DIR / 'example_areas_timeseries.csv')] + new_ts_rows,
        ignore_index=True)
    example_meta_full.to_csv(OUTPUT_DIR / 'example_areas.csv', index=False)
    example_ts_full.to_csv(OUTPUT_DIR / 'example_areas_timeseries.csv', index=False)
    print(f'  wrote example_areas.csv / example_areas_timeseries.csv '
          f'({len(example_meta_full)} examples total, '
          f'{len(new_meta_rows)} newly added)')

    # ── Update manifest ----------------------------------------------------------
    manifest = json.loads((OUTPUT_DIR / 'manifest.json').read_text())
    manifest['files'] = sorted(p.name for p in OUTPUT_DIR.iterdir())
    manifest['followup_note'] = (
        'mode_recharacterization.csv and 4 new confident/active example areas '
        'added by scripts/az3_deep_dive_followups.py -- see '
        'docs/az3-report-review-plan.md'
    )
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # ── Rebuild report -------------------------------------------------------------
    print('\n-- Rebuilding report --')
    sys.path.insert(0, str(OUTPUT_DIR))
    import build_report as br
    br.build_report(OUTPUT_DIR)
    print(f'Report written to {OUTPUT_DIR / "report.html"}')


if __name__ == '__main__':
    main()
