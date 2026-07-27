"""
Generates one PNG per confidence tier for docs/az3-stakeholder-summary.md's
"Example areas" section -- the four areas already named in that doc's text,
rendered against the underlying P_obs/E_obs records so a reader can see what
each tier actually looks like.

Opens results/traces_full/AZ3.nc exactly once, read-only, then slices out just
these 4 areas before plotting rather than materializing the full
(chain, draw, 4987, 10) z array -- per load_trace_no_warmup's own stated
memory-cost discipline for this trace.

Run: pixi run python scripts/stakeholder_example_plots.py
"""
import ast
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_SCRIPTS_DIR))

from housing_projections.config import COLOURS, DATA_PATH, INFER_YEARS  # noqa: E402
from housing_projections.data import load_data  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402
from housing_projections.plots.core import _SCENARIO_COLOURS, plot_z_area  # noqa: E402

import full_dataset_characterization as fdc  # noqa: E402

# A first version of this script used the internal technical report's binary
# hatch convention (>50% noise probability = hatched, per
# docs/az3-report-review-plan.md's B1 finding). Replaced after review: a
# binary cutoff hides exactly the cases that matter most -- e.g. a bar at 51%
# noise probability that a resolved scenario still substantively treats as
# real change looked identical to one at 99%. Faded continuously by
# resp_noise instead (opaque = trusted, faded = likely anomaly) so the actual
# probability is visible, not just which side of one threshold it fell on.
ALPHA_TRUSTED = 0.9    # bar fill alpha at resp_noise == 0
ALPHA_ANOMALY = 0.15   # bar fill alpha at resp_noise == 1

TRACE_PATH = _REPO_ROOT / 'results' / 'traces_full' / 'AZ3.nc'
ESTIMATES_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_year_estimates'
OUTPUT_DIR = _REPO_ROOT / 'results' / 'artifacts' / 'az3_stakeholder_examples'
DATA_PATH_ARG = str(DATA_PATH) if DATA_PATH else 'data'

# (LSOA code, output filename suffix, plot style) -- matching
# docs/az3-stakeholder-summary.md's "Example areas" bullets exactly. Identities
# confirmed beforehand against area_tier_summary.csv / area_scenarios.csv:
#   E01034220 -- Newham, D=719, tier1 (4 clean bursts: 2015/2016/2018/2020)
#   E01002702 -- Islington, D=479, tier2/resolved (63%/2021 vs 37%/2019)
#   E01000251 -- Barnet, D=133, tier2/unresolved-minor (2018/2021 dominant,
#                only 2012/2013 -- 14.7% of magnitude -- unresolved)
#   E01035667 -- Tower Hamlets, D=379, tier3 (no active P/E year at all)
EXAMPLES = [
    ('E01034220', 'tier1_confident', 'plain'),
    ('E01002702', 'tier2_resolved_stories', 'scenarios'),
    ('E01000251', 'tier2_mostly_confident', 'plain'),
    ('E01035667', 'tier3_diffuse', 'plain'),
]

_NUMBER_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def _scenario_title(area_code, D_val, scenarios):
    """
    Computed (not hardcoded) so it stays correct for any area, not just the
    4 pre-selected examples -- e.g. if area_scenarios.csv is ever regenerated
    with different weights/peak years, or this is reused for an
    arbitrary user-selected area (as in the dashboard's multi-LSOA view).
    """
    word = _NUMBER_WORDS.get(len(scenarios), str(len(scenarios)))
    parts = [f'{row.weight:.0%} likely {int(row.peak_year)}'
             for row in scenarios.itertuples()]
    return (f'{area_code}  (D={D_val:.0f}): {word} possible year patterns '
            f'({", ".join(parts)})')


def _noise_to_alpha(noise):
    return ALPHA_TRUSTED - (ALPHA_TRUSTED - ALPHA_ANOMALY) * noise


def draw_pe_bars(ax, years, P_obs, E_obs, resp_noise_P=None, resp_noise_E=None):
    """
    Draws P_obs/E_obs as plain-coloured bars, faded continuously by the
    model's noise-mixture posterior probability that each observation is a
    data anomaly rather than a real change (opaque = trusted, faded = likely
    anomaly) -- this is what lets a reader see directly why z sometimes
    doesn't move to match an observed spike, and how confident that judgement
    actually was, not just whether it crossed some fixed cutoff. Every bar
    keeps a solid black outline regardless of fill fade, so its height (the
    observed magnitude) stays legible even when heavily faded.

    Returns True if any resp_noise array was supplied (so the caller knows
    whether to add the fade-meaning entries to the legend).
    """
    p_bars = ax.bar([y - 0.15 for y in years], P_obs, width=0.3, label='Planning',
                    color=COLOURS['planning'], edgecolor='black', linewidth=0.5)
    e_bars = ax.bar([y + 0.15 for y in years], E_obs, width=0.3, label='UPRN',
                    color=COLOURS['uprn'], edgecolor='black', linewidth=0.5)
    has_noise_info = False
    for bars, resp_noise in ((p_bars, resp_noise_P), (e_bars, resp_noise_E)):
        if resp_noise is None:
            continue
        has_noise_info = True
        for bar, noise in zip(bars, resp_noise):
            bar.set_alpha(_noise_to_alpha(noise))
    return has_noise_info


FADE_LEGEND_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def finish_legend(ax, has_noise_info):
    handles, labels = ax.get_legend_handles_labels()
    if has_noise_info:
        for pct in FADE_LEGEND_STEPS:
            handles.append(Patch(facecolor='grey', edgecolor='black',
                                  alpha=_noise_to_alpha(pct)))
            labels.append(f'{pct:.0%} likely a data anomaly')
    ax.legend(handles, labels, fontsize=7)


def plot_scenarios_from_csv(ax, area_code, P_obs, E_obs, resp_noise_P, resp_noise_E, D, idx):
    """
    Tier-2-resolved scenario plot for one area, sourced from
    results/artifacts/az3_year_estimates/area_scenarios.csv +
    area_year_estimates.csv -- i.e. the SAME validated scenario split the
    dashboard itself renders (notebooks/8.0-cd-az3_estimates_dashboard.py).

    Deliberately does not re-derive scenarios via plot_z_area_modes's
    whole-draw k-means: that method clusters on the full 10-year vector and
    can disagree with the flagged-years-restricted method that actually
    produced area_scenarios.csv (see
    docs/estimates-dashboard-report-plan.md's Phase A method note) -- checked
    directly for E01002702 and the two methods do disagree (whole-vector
    k-means gives ~98%/2%, not the documented 63%/37%), so this reuses the
    already-validated CSV numbers instead of a fresh, less trustworthy
    reclustering.
    """
    scenarios = pd.read_csv(ESTIMATES_DIR / 'area_scenarios.csv')
    scenarios = scenarios[scenarios['area'] == area_code].sort_values(
        'weight', ascending=False)
    year_est = pd.read_csv(ESTIMATES_DIR / 'area_year_estimates.csv')
    year_est = year_est[year_est['area'] == area_code].sort_values('year')
    years = year_est['year'].tolist()

    ax.fill_between(years, year_est['z_lo90'], year_est['z_hi90'],
                     alpha=0.15, color='gray', label='90% CI (overall)')

    has_noise_flag = draw_pe_bars(ax, INFER_YEARS, P_obs[idx], E_obs[idx],
                                  resp_noise_P[idx], resp_noise_E[idx])

    for rank, row in enumerate(scenarios.itertuples()):
        profile = ast.literal_eval(row.year_profile)
        ax.plot(years, profile, color=_SCENARIO_COLOURS[rank % len(_SCENARIO_COLOURS)],
                 marker='o', linewidth=1.8, zorder=3,
                 label=f'{row.scenario_label} ({row.weight:.0%} of draws)')

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net dwelling change')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(_scenario_title(area_code, D[idx], scenarios))
    finish_legend(ax, has_noise_flag)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'-- Loading trace: {TRACE_PATH} --')
    trace = fdc.load_trace_no_warmup(TRACE_PATH)

    print(f'-- Loading data: {DATA_PATH_ARG} --')
    gdf = load_data(DATA_PATH_ARG)
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
    data = fdc.data_matching_trace(gdf, trace)

    lsoa_codes_full = trace.posterior['z'].coords['area'].values.tolist()
    idx_by_code = {code: i for i, code in enumerate(lsoa_codes_full)}

    codes = [code for code, _, _ in EXAMPLES]
    global_idxs = [idx_by_code[code] for code in codes]

    # Slice just these 4 areas out of the trace/data rather than materializing
    # the full 4987-area z array.
    z_sub = trace.posterior['z'].isel(area=global_idxs).values  # (chain, draw, 4, year)
    P_sub = data['P_obs'][global_idxs]
    E_sub = data['E_obs'][global_idxs]
    D_sub = data['D'][global_idxs]
    resp_noise_P_sub = trace.posterior['resp_noise_P'].isel(
        area=global_idxs).mean(dim=('chain', 'draw')).values
    resp_noise_E_sub = trace.posterior['resp_noise_E'].isel(
        area=global_idxs).mean(dim=('chain', 'draw')).values

    for local_idx, (code, fname, style) in enumerate(EXAMPLES):
        fig, ax = plt.subplots(figsize=(9, 4))
        if style == 'scenarios':
            plot_scenarios_from_csv(ax, code, P_sub, E_sub, resp_noise_P_sub,
                                     resp_noise_E_sub, D_sub, local_idx)
        else:
            # P_obs/E_obs=None -- plot_z_area draws only z's line/CI/D-baseline
            # here; the faded bars below replace its own plain P/E lines.
            # show_post_sum=False: AZ3's zero-sum prior forces the posterior
            # sum to equal D exactly on every draw, so that comparison (a
            # holdover from the M-family, where the census total was only a
            # soft/probabilistic constraint) is always a trivial zero-width
            # match here -- not informative for this model.
            plot_z_area(ax, z_sub, local_idx, P_obs=None, E_obs=None, D=D_sub,
                        lsoa_codes=codes, show_legend=False, show_post_sum=False)
            has_noise_flag = draw_pe_bars(
                ax, INFER_YEARS, P_sub[local_idx], E_sub[local_idx],
                resp_noise_P_sub[local_idx], resp_noise_E_sub[local_idx])
            finish_legend(ax, has_noise_flag)
        fig.tight_layout()

        out_path = OUTPUT_DIR / f'{code}_{fname}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
