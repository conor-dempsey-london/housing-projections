"""
Full per-(area, year) numerical estimates export for AZ3 -- Phase A of
docs/estimates-dashboard-report-plan.md. Produces the one set of CSVs every
downstream deliverable (dashboard, stakeholder report) reads from, so the
24GB trace at results/traces_full/AZ3.nc is only ever opened here.

Applies the three-tier reporting scheme from
docs/model-stopping-criteria-and-communication.md Sec 4 to every one of the
4987 areas x 10 years, not just the hand-picked examples in
results/artifacts/az3_full_characterization/:

  Tier 1 (confident)   -- z_identifiability_summary.confident AND
                          0 KDE-flagged multimodal years (area_summary.csv's
                          n_multimodal_years == 0). Point mean + 90% CI.
  Tier 2 (ambiguous)   -- otherwise, provided the area has any active P/E
                          year. See "Scenario decomposition" below for how
                          resolved/unresolved is decided.
  Tier 3 (diffuse)     -- no active P/E year across the whole decade
                          (area_summary.csv's has_active_year == False).
                          Reported as total D only, no year breakdown.

Reuses results/artifacts/az3_full_characterization/area_summary.csv and
multimodal_cells.csv (already on disk) for the tier-classification inputs,
rather than re-running the expensive per-cell KDE scan
(detect_z_multimodality) a second time.

Scenario decomposition -- method note
--------------------------------------
The first version of this script clustered each area's FULL 10-year
standardized z vector via whole-draw k-means (the same method as
scripts/az3_deep_dive_followups.py's characterize_area_modes /
plots/core.py's plot_z_area_modes). Verified against this project's own two
headline documented ground-truth areas before trusting it -- and it got both
backwards: E01002702 (confirmed genuine bimodality, per
docs/az-family-work-plan.md Phase 3) came out "unresolved"; E01002794
(confirmed spurious/diffuse, per docs/az3-report-review-plan.md's own
pairwise-correlation investigation) came out "resolved" with a confident
4-scenario split.

Root cause, confirmed by direct inspection: whole-vector k-means clusters on
all 10 years' Euclidean distance, so a real bimodality localized to the
specific year(s) the per-cell KDE scan (detect_z_multimodality) already
flagged can be swamped by unrelated variance in large-but-unambiguous other
years -- and conversely k-means will always partition even genuinely diffuse
data into k roughly-balanced wedges, which is indistinguishable from a real
split by cluster-balance alone. (This also means
results/artifacts/az3_full_characterization/mode_recharacterization.csv's
"734/999 areas need 3+ scenarios" finding, built with the same whole-vector
method, is likely unreliable -- flagged as a known follow-up in
docs/estimates-dashboard-report-plan.md, not fixed here per user instruction.)

Fix used here: restrict the clustering signal to exactly the years
multimodal_cells.csv already flagged as per-cell multimodal for that area
(the KDE scan is validated; trust it to say WHICH years are ambiguous, then
only look at those dimensions):
  - 1 flagged year: no cross-year clustering needed -- reuse
    diagnostics._detect_modes' own validated per-cell KDE modes directly for
    that single cell.
  - >=2 flagged years: resolved only if the flagged years show a real
    anti-correlated (either-or) relationship (min pairwise correlation among
    them <= -0.3) -- confirmed against ground truth: E01002702's 2
    flagged years correlate at -0.89 (clears it); E01002794's 5 flagged
    years correlate at -0.07 to -0.18 (does not, matching its own
    already-documented "closer to diffuse/near-exchangeable" conclusion).
    If resolved, a 2-way k-means restricted to just those flagged
    dimensions gives the scenario split.

Usage
-----
    pixi run python scripts/az3_year_estimates.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_SCRIPTS_DIR))

import gla_data._ons  # noqa: E402
import full_dataset_characterization as fdc  # noqa: E402
from housing_projections.config import INFER_YEARS  # noqa: E402
from housing_projections.data import load_data, make_borough_idx  # noqa: E402
from housing_projections.diagnostics import _detect_modes  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402

OUTPUT_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_year_estimates'
CHAR_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_full_characterization'
TRACE_PATH = _REPO_ROOT / 'results' / 'traces_full' / 'AZ3.nc'
DATA_PATH = str(fdc.DATA_PATH) if fdc.DATA_PATH else 'data'

MIN_CLUSTER_FRAC = 0.03      # a cluster must clear this share of draws to get its own label
MIN_CORR_BAR = -0.3          # min pairwise correlation among an area's flagged years to
                              # call the ambiguity a real either-or split (see module docstring)
CONCENTRATION_BAR = 0.5      # top-2-flagged-year argmax share, same bar as plots/core.py's
                              # plot_z_area_modes -- rules out the combinatorial-exchangeability
                              # case (many flagged years, none individually dominant) that
                              # min_corr alone doesn't catch, see module docstring
SCENARIO_LETTERS = 'ABCD'


def classify_tier(row):
    if not row['has_active_year']:
        return 'tier3'
    if row['confident'] and row['n_multimodal_years'] == 0:
        return 'tier1'
    return 'tier2_candidate'


def decompose_area(z_area_full, flagged_years, min_frac=MIN_CLUSTER_FRAC,
                    min_corr_bar=MIN_CORR_BAR, concentration_bar=CONCENTRATION_BAR, seed=0):
    """
    Decide whether an area's flagged year-allocation ambiguity resolves into
    a small number of genuine labelled scenarios. See module docstring for
    the method and why it replaced whole-vector k-means.

    Requires BOTH a real anti-correlated pair (min_corr_bar) AND a small
    dominant subset of the flagged years (concentration_bar) -- min_corr
    alone is not sufficient once an area has many (5+) flagged years: found
    by direct inspection of E01035709 (9 flagged years, all near-identical
    ~0-or-~90 cells) that its pairwise correlations are just noise around
    the -1/(n-1) baseline the zero-sum constraint mechanically induces among
    ANY set of interchangeable candidate years (observed mean -0.12, matching
    -1/8 almost exactly) -- one pair sampling to -0.55 by chance is enough to
    clear min_corr_bar despite there being no real 2-group structure at all
    (off-diagonal correlations scattered -0.55..+0.32, no block pattern; the
    number of "high" years per draw is tightly fixed at 3-4 of 9, consistent
    with genuine combinatorial exchangeability -- confident about HOW MANY
    years absorbed the change, uncertain about WHICH -- not 2 distinguishable
    stories). Its flagged-year argmax concentration is only 31%, correctly
    below the bar, where E01002702/E01004686 (genuine cases) score 100%/86%.

    Parameters
    ----------
    z_area_full   : (n_draws, n_years) full-precision posterior draws for
                    one area (all years, not just flagged ones -- needed so
                    reported scenarios carry each cluster's full 10-year
                    profile, not just the flagged dimensions).
    flagged_years : list[int] -- year indices multimodal_cells.csv flagged
                    for this area.

    Returns
    -------
    dict with 'resolved' (bool) and, if resolved, 'clusters': a list of
    {'weight', 'peak_year_idx', 'peak_year_z', 'year_profile'} sorted by
    weight descending.
    """
    if len(flagged_years) == 0:
        return {'resolved': False}

    if len(flagged_years) == 1:
        y = flagged_years[0]
        vals = z_area_full[:, y]
        locs, _, n_modes = _detect_modes(vals)
        if n_modes < 2:
            return {'resolved': False}
        assign = np.argmin(np.abs(vals[:, None] - locs[None, :]), axis=1)
        clusters = []
        for m in range(n_modes):
            mask = assign == m
            w = float(mask.mean())
            if w < min_frac:
                continue
            clusters.append({
                'weight': w, 'peak_year_idx': y, 'peak_year_z': float(locs[m]),
                'year_profile': z_area_full[mask].mean(axis=0),
            })
        clusters.sort(key=lambda c: -c['weight'])
        return {'resolved': len(clusters) >= 2, 'clusters': clusters}

    sub = z_area_full[:, flagged_years]
    corr = np.corrcoef(sub.T)
    min_corr = float(corr[np.triu_indices_from(corr, k=1)].min())

    argmax_flagged = sub.argmax(axis=1)
    mass = np.bincount(argmax_flagged, minlength=len(flagged_years)) / len(sub)
    concentration = float(np.sort(mass)[::-1][:2].sum())

    if min_corr > min_corr_bar or concentration < concentration_bar:
        return {'resolved': False, 'min_corr': min_corr, 'concentration': concentration}

    std = sub.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    norm = (sub - sub.mean(axis=0)) / std_safe
    rng = np.random.default_rng(seed)
    _, labels = kmeans2(norm, 2, minit='++', seed=rng)

    clusters = []
    for i in range(2):
        mask = labels == i
        w = float(mask.mean())
        if w < min_frac:
            continue
        year_profile = z_area_full[mask].mean(axis=0)
        peak_year_idx = flagged_years[int(np.argmax(np.abs(year_profile[flagged_years])))]
        clusters.append({
            'weight': w, 'peak_year_idx': peak_year_idx,
            'peak_year_z': float(year_profile[peak_year_idx]),
            'year_profile': year_profile,
        })
    clusters.sort(key=lambda c: -c['weight'])
    return {'resolved': len(clusters) >= 2, 'min_corr': min_corr,
            'concentration': concentration, 'clusters': clusters}


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'-- Loading trace (read-only): {TRACE_PATH} --')
    trace = fdc.load_trace_no_warmup(str(TRACE_PATH))
    print(f'   z shape {trace.posterior["z"].shape}, loaded in {time.time() - t0:.0f}s')

    print(f'-- Loading data: {DATA_PATH} --')
    gdf = load_data(DATA_PATH)
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
    data = fdc.data_matching_trace(gdf, trace)
    n_areas, n_years = data['n_areas'], data['n_years']
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    years = list(INFER_YEARS[:n_years])
    print(f'   {n_areas} areas x {n_years} years matched to trace')

    borough_idx, n_boroughs, borough_codes = make_borough_idx(data['gdf'])

    print('-- Borough name lookup (gla_data ONS geography) --')
    geo_lookup = gla_data._ons.fetch_geography_lookup(2021, 'lsoa')
    geo_lookup = geo_lookup[geo_lookup['LSOA21CD'].isin(lsoa_codes)].copy()
    code_to_name = (geo_lookup.drop_duplicates('LAD22CD')
                     .set_index('LAD22CD')['LAD22NM'].to_dict())
    borough_name_by_area = [code_to_name.get(borough_codes[i], borough_codes[i])
                             for i in borough_idx]

    print('-- Loading existing tier-classification inputs (area_summary.csv, '
          'multimodal_cells.csv) --')
    area_df = pd.read_csv(CHAR_DIR / 'area_summary.csv')
    idx_by_code = {code: i for i, code in enumerate(lsoa_codes)}
    mismatches = sum(1 for i, code in enumerate(area_df['area']) if idx_by_code.get(code) != i)
    if mismatches:
        raise RuntimeError(f'area_summary.csv row order does not match trace area order '
                            f'({mismatches} mismatches) -- regenerate it against this trace first')
    area_df['tier'] = area_df.apply(classify_tier, axis=1)
    print('  ', area_df['tier'].value_counts().to_dict())

    mm_df = pd.read_csv(CHAR_DIR / 'multimodal_cells.csv')
    flagged_by_area_idx = mm_df.groupby('area_idx')['year_idx'].apply(list).to_dict()

    # -- Full-precision per-(area, year) point estimate + 90% CI, all areas --------
    print('-- z posterior mean / 90% CI, full draws, all areas x years --')
    z_post = trace.posterior['z'].values  # (chain, draw, area, year)
    C, S, A, T = z_post.shape
    z_flat_all = z_post.reshape(C * S, A, T)
    z_mean_all = z_flat_all.mean(axis=0)
    z_lo_all = np.percentile(z_flat_all, 5, axis=0)
    z_hi_all = np.percentile(z_flat_all, 95, axis=0)
    print(f'   z_post {z_post.nbytes / 1e9:.2f} GB loaded')

    # -- Tier 2: scenario decomposition on flagged years only -----------------------
    tier2_idx = area_df.index[area_df['tier'] == 'tier2_candidate'].tolist()
    print(f'-- Tier 2 scenario decomposition for {len(tier2_idx)} candidate areas --')

    scenario_rows = []
    area_df['tier_subtype'] = None
    area_df['n_flagged_years'] = 0
    area_df['min_flagged_corr'] = np.nan
    area_df['flagged_concentration'] = np.nan

    for n, area_idx in enumerate(tier2_idx):
        flagged = flagged_by_area_idx.get(area_idx, [])
        area_df.loc[area_idx, 'n_flagged_years'] = len(flagged)
        z_area_full = z_flat_all[:, area_idx, :]
        result = decompose_area(z_area_full, flagged)

        if 'min_corr' in result:
            area_df.loc[area_idx, 'min_flagged_corr'] = result['min_corr']
        if 'concentration' in result:
            area_df.loc[area_idx, 'flagged_concentration'] = result['concentration']
        area_df.loc[area_idx, 'tier_subtype'] = 'resolved' if result['resolved'] else 'unresolved'

        if result['resolved']:
            code = lsoa_codes[area_idx]
            for scenario_i, cluster in enumerate(result['clusters']):
                scenario_rows.append({
                    'area': code,
                    'scenario_label': f'Scenario {SCENARIO_LETTERS[scenario_i]}',
                    'weight': cluster['weight'],
                    'peak_year': years[cluster['peak_year_idx']],
                    'peak_year_z': cluster['peak_year_z'],
                    'year_profile': json.dumps(np.round(cluster['year_profile'], 1).tolist()),
                })
        if (n + 1) % 400 == 0:
            print(f'   ...{n + 1}/{len(tier2_idx)}')

    area_df.loc[area_df['tier'] == 'tier2_candidate', 'tier'] = 'tier2'
    n_resolved = (area_df['tier_subtype'] == 'resolved').sum()
    n_unresolved = (area_df['tier_subtype'] == 'unresolved').sum()
    print(f'   tier2: {n_resolved} resolved (clean scenario split), '
          f'{n_unresolved} unresolved (ambiguous, no clean split)')

    area_df['borough_name'] = borough_name_by_area

    # -- (area, year) grain output ---------------------------------------------------
    print('-- Assembling area_year_estimates.csv --')
    year_rows = []
    for area_idx, code in enumerate(lsoa_codes):
        tier = area_df.loc[area_idx, 'tier']
        tier_subtype = area_df.loc[area_idx, 'tier_subtype']
        for year_idx, year in enumerate(years):
            year_rows.append({
                'area': code,
                'year': year,
                'tier': tier,
                'tier_subtype': tier_subtype,
                'z_mean': z_mean_all[area_idx, year_idx],
                'z_lo90': z_lo_all[area_idx, year_idx],
                'z_hi90': z_hi_all[area_idx, year_idx],
            })
    year_estimates_df = pd.DataFrame(year_rows)
    year_estimates_df.to_csv(OUTPUT_DIR / 'area_year_estimates.csv', index=False)
    print(f'   wrote area_year_estimates.csv ({len(year_estimates_df)} rows)')

    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_csv(OUTPUT_DIR / 'area_scenarios.csv', index=False)
    print(f'   wrote area_scenarios.csv ({len(scenarios_df)} rows, '
          f'{scenarios_df["area"].nunique() if len(scenarios_df) else 0} resolved areas)')

    tier_cols = ['area', 'borough_name', 'borough', 'D', 'tier', 'tier_subtype',
                 'n_low_confidence_years', 'max_rhat', 'n_multimodal_years',
                 'n_flagged_years', 'min_flagged_corr', 'flagged_concentration',
                 'has_active_year']
    area_df[tier_cols].to_csv(OUTPUT_DIR / 'area_tier_summary.csv', index=False)
    print(f'   wrote area_tier_summary.csv ({len(area_df)} rows)')
    print('  ', area_df['tier'].value_counts().to_dict())

    # -- Borough / London aggregate rollup, per year + full-decade total ------------
    print('-- Borough / London aggregate rollup --')
    borough_name_arr = np.array(borough_name_by_area)
    agg_rows = []
    for geo_label, mask in [('London', np.ones(A, dtype=bool))] + [
        (name, borough_name_arr == name) for name in sorted(set(borough_name_by_area))
    ]:
        idx = np.where(mask)[0]
        for year_idx, year in enumerate(years):
            grp = z_flat_all[:, idx, year_idx].sum(axis=1)
            agg_rows.append({
                'geography': geo_label, 'year': year, 'n_areas': len(idx),
                'z_total_mean': grp.mean(),
                'z_total_lo90': np.percentile(grp, 5),
                'z_total_hi90': np.percentile(grp, 95),
            })
        grp_total = z_flat_all[:, idx, :].sum(axis=(1, 2))
        agg_rows.append({
            'geography': geo_label, 'year': 'total', 'n_areas': len(idx),
            'z_total_mean': grp_total.mean(),
            'z_total_lo90': np.percentile(grp_total, 5),
            'z_total_hi90': np.percentile(grp_total, 95),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUTPUT_DIR / 'borough_london_totals.csv', index=False)
    print(f'   wrote borough_london_totals.csv ({len(agg_df)} rows, '
          f'{len(set(borough_name_by_area))} boroughs + London)')

    manifest = {
        'model_name': 'AZ3',
        'trace_path': str(TRACE_PATH),
        'data_path': DATA_PATH,
        'n_areas': n_areas, 'n_years': n_years,
        'min_cluster_frac': MIN_CLUSTER_FRAC,
        'min_corr_bar': MIN_CORR_BAR,
        'method_note': 'scenario decomposition restricted to per-cell-flagged years '
                        '(see script docstring) -- NOT whole-vector k-means',
        'files': sorted(p.name for p in OUTPUT_DIR.iterdir()),
    }
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nDone in {time.time() - t0:.0f}s.')


if __name__ == '__main__':
    main()
