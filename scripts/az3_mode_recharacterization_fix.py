"""
Recomputes results/artifacts/az3_full_characterization/mode_recharacterization.csv
using the corrected per-flagged-year decomposition method, replacing the original
whole-vector k-means version built by scripts/az3_deep_dive_followups.py (Task 3).

Why this was wrong
-------------------
The original `characterize_area_modes` ran k-means on an area's FULL 10-year
standardized z vector. scripts/az3_year_estimates.py's `decompose_area` (built
later, for the per-(area,year) estimates export) found and fixed the same bug in
a sibling method: whole-vector k-means is dominated by whichever years have the
most variance, which can easily NOT be the years the per-cell KDE scan
(detect_z_multimodality) actually flagged as ambiguous -- and it always partitions
even genuinely diffuse data into k roughly-balanced wedges, indistinguishable from
a real split by cluster-balance alone (the E01035709 cautionary case). The
original mode_recharacterization.csv's headline finding ("734/999 areas genuinely
need 3+ scenarios") used this flawed method and was flagged in
docs/estimates-dashboard-report-plan.md as "likely unreliable" -- a known
follow-up, not fixed at the time.

The fix, and a second bug caught while building it
---------------------------------------------------
Restrict clustering to exactly the years multimodal_cells.csv already flagged as
per-cell multimodal for that area (same fix as `decompose_area`), and reuse
`decompose_area`'s own validated gates -- `min_corr_bar` (>=1 pair of flagged
years genuinely anti-correlated, <= -0.3, across draws) and `concentration_bar`
(top-k argmax-share among the flagged years, not the whole vector, clears 0.5) --
rather than the original's naive "first k where every cluster clears a floor" rule.

**This script originally generalised the k=2 method to k=3/4 directly (test each
k against the same fixed 0.5 concentration bar) -- checked against the E01035709
ground-truth case before trusting it, and it FAILED**: E01035709 (the canonical
documented false positive -- 9 near-exchangeable flagged years, already confirmed
to have no real 2-group structure) came back `best_k=4` with k=4 concentration
54.6%, clearing the 0.5 bar. Root cause: "top-k argmax share" has a mechanically
rising baseline as k grows relative to the number of flagged years (top-4-of-9
covers ~44% of mass under pure exchangeability alone, before any real structure);
a FIXED 0.5 bar is only a meaningful filter at small k relative to n_flagged
(where decompose_area validated it), not in general. At the k=n_flagged boundary
concentration is trivially 1.0 regardless of real structure, so it can't
discriminate there at all.

**Scope of the actual fix, given that**: this script now tests ONLY k=2 with the
validated method (reproducing decompose_area's exact gates on the flagged-years
subspace) and reports whether each area resolves into a genuine 2-way split.
Raw k=3/k=4 diagnostics (weights, concentration) are still recorded for
visibility/future work, but are NOT used for any headline claim -- properly
generalising the concentration check to k>2 (e.g. comparing against the
exchangeable-null baseline k/n_flagged with a calibrated margin, not a fixed 0.5)
is real, undone work, not something to invent and publish untested in the same
pass that just caught the k=2 version's own analogous bug. This deliberately
retracts the original "734/999 areas need 3+ scenarios" claim rather than
replacing it with an equally untested different number.

Reuses area_summary.csv / multimodal_cells.csv already on disk (same candidate
selection as the original: areas with >=3 individually-flagged multimodal years).
Opens results/traces_full/AZ3.nc exactly once, read-only, thinned the same way
(every 4th draw) as the original for tractability at ~1000 candidate areas.

Usage
-----
    pixi run python scripts/az3_mode_recharacterization_fix.py
"""
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

import full_dataset_characterization as fdc  # noqa: E402

OUTPUT_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_full_characterization'
TRACE_PATH = _REPO_ROOT / 'results' / 'traces_full' / 'AZ3.nc'

THRESH_YEARS_FOR_RECHAR = 3   # areas with >= this many flagged years get re-clustered
                              # (same threshold as the original, superseded pass)
MIN_CLUSTER_FRAC = 0.05       # a cluster must clear this to count as "real" here
                              # (same bar as the original, superseded pass)
MIN_CORR_BAR = -0.3           # same validated gate as az3_year_estimates.py's decompose_area
CONCENTRATION_BAR = 0.5       # same validated gate as az3_year_estimates.py's decompose_area
THIN = 4


def characterize_area_modes_fixed(z_area_full, flagged_years, k_values=(2, 3, 4),
                                   min_frac=MIN_CLUSTER_FRAC, min_corr_bar=MIN_CORR_BAR,
                                   concentration_bar=CONCENTRATION_BAR, seed=0):
    """Corrected version of az3_deep_dive_followups.characterize_area_modes --
    restricts clustering + the concentration diagnostic to exactly
    `flagged_years`, not the whole 10-year vector. See module docstring."""
    sub = z_area_full[:, flagged_years]
    n_flagged = len(flagged_years)

    if n_flagged > 1:
        corr = np.corrcoef(sub.T)
        min_corr = float(corr[np.triu_indices_from(corr, k=1)].min())
    else:
        min_corr = np.nan  # single flagged year -- handled the same way

    std = sub.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    norm = (sub - sub.mean(axis=0)) / std_safe

    argmax_flagged = sub.argmax(axis=1)
    year_mass = np.bincount(argmax_flagged, minlength=n_flagged) / len(sub)

    out = {'min_corr': min_corr}
    for k in k_values:
        if k > n_flagged:
            out[k] = {'weights': [], 'n_real_clusters': 0, 'concentration': 0.0}
            continue
        rng = np.random.default_rng(seed)
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


def resolved_2way_from(res, min_corr_bar=MIN_CORR_BAR, concentration_bar=CONCENTRATION_BAR):
    """Does this area's flagged years resolve into a genuine 2-way split? Exactly
    az3_year_estimates.py's decompose_area decision rule, evaluated here on the
    same flagged-years subspace. k=3/4 are deliberately NOT tested for a
    headline verdict -- see module docstring."""
    if not np.isnan(res['min_corr']) and res['min_corr'] > min_corr_bar:
        return False
    return res[2]['n_real_clusters'] == 2 and res[2]['concentration'] >= concentration_bar


def main():
    t0 = time.time()
    print(f'-- Loading trace (read-only): {TRACE_PATH} --')
    trace = fdc.load_trace_no_warmup(str(TRACE_PATH))
    print(f'   z shape {trace.posterior["z"].shape}, loaded in {time.time() - t0:.0f}s')

    area_df = pd.read_csv(OUTPUT_DIR / 'area_summary.csv')
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    idx_by_code = {code: i for i, code in enumerate(lsoa_codes)}
    mismatches = sum(1 for i, code in enumerate(area_df['area']) if idx_by_code.get(code) != i)
    print(f'   area_summary.csv row/trace-index correspondence check: {mismatches} mismatches '
          f'(0 expected)')

    mm_df = pd.read_csv(OUTPUT_DIR / 'multimodal_cells.csv')
    flagged_by_area_idx = mm_df.groupby('area_idx')['year_idx'].apply(sorted).to_dict()

    candidates = area_df[area_df['n_multimodal_years'] >= THRESH_YEARS_FOR_RECHAR]
    print(f'-- {len(candidates)} candidate areas (same selection as the superseded pass) --')

    z_thin = trace.posterior['z'].isel(draw=slice(None, None, THIN)).values
    n_years = z_thin.shape[-1]
    print(f'   thinned z loaded, {z_thin.nbytes / 1e9:.2f} GB')

    rows = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        code = row['area']
        idx_c = idx_by_code[code]
        flagged = flagged_by_area_idx.get(idx_c, [])
        if len(flagged) < THRESH_YEARS_FOR_RECHAR:
            # area_summary.csv's n_multimodal_years and multimodal_cells.csv's own
            # per-area flagged-year list should always agree -- guard rather than
            # silently mis-scope an area if they ever don't.
            raise RuntimeError(f'{code}: n_multimodal_years={row["n_multimodal_years"]} but '
                                f'only {len(flagged)} rows in multimodal_cells.csv')
        z_flat = z_thin[:, :, idx_c, :].reshape(-1, n_years)
        res = characterize_area_modes_fixed(z_flat, flagged)
        rows.append({
            'area': code, 'borough': row['borough'], 'D': row['D'],
            'n_multimodal_years': row['n_multimodal_years'],
            'n_flagged_years': len(flagged),
            'min_corr': res['min_corr'],
            'resolved_2way': resolved_2way_from(res),
            'k2_weights': res[2]['weights'], 'k2_concentration': res[2]['concentration'],
            'k3_weights_untested': res[3]['weights'],
            'k3_concentration_untested': res[3]['concentration'],
            'k4_weights_untested': res[4]['weights'],
            'k4_concentration_untested': res[4]['concentration'],
        })
        if (i + 1) % 200 == 0:
            print(f'   ...{i + 1}/{len(candidates)}')
    del z_thin

    recharacterization_df = pd.DataFrame(rows)
    recharacterization_df.to_csv(OUTPUT_DIR / 'mode_recharacterization.csv', index=False)
    n_resolved = int(recharacterization_df['resolved_2way'].sum())
    n_unresolved = len(recharacterization_df) - n_resolved
    print(f'\n   wrote mode_recharacterization.csv ({len(recharacterization_df)} rows)')
    print(f'   resolved_2way=True (genuine 2-way split among flagged years): {n_resolved}')
    print(f'   resolved_2way=False (no real split found -- may still be genuinely '
          f'multimodal, just not a clean 2-way story): {n_unresolved}')
    print(f'\nDone in {time.time() - t0:.0f}s.')


if __name__ == '__main__':
    main()
