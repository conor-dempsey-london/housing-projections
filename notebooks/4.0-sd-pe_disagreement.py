# %% Imports
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from housing_projections.config import DATA_PATH, INFER_YEARS
from housing_projections.data import load_data, make_data_dict
from housing_projections.outliers import apply_outlier_exclusion

# %% Configuration
ZERO_THRESHOLD  = 0.5   # |value| below this treated as zero
MIN_NONZERO     = 3     # min non-zero years to include area in ratio/lag analysis
MAX_LAG         = 5     # years to scan in cross-correlation

# %% Load data
gdf_raw        = load_data(DATA_PATH)
gdf_clean, _   = apply_outlier_exclusion(gdf_raw)
data           = make_data_dict(gdf_clean)

P = data['P_obs']   # (n_areas, n_years)
E = data['E_obs']   # (n_areas, n_years)
n_areas, n_years = P.shape
years = np.array(INFER_YEARS)

print(f'Areas: {n_areas}   Years: {n_years}   ({years[0]}–{years[-1]})')

# %% ── 1. Zero-rate analysis ───────────────────────────────────────────────────
# Classify each (area, year) cell into one of four joint states.

P_zero = np.abs(P) < ZERO_THRESHOLD
E_zero = np.abs(E) < ZERO_THRESHOLD

both_zero    = ( P_zero &  E_zero).sum()
both_nonzero = (~P_zero & ~E_zero).sum()
p_only_zero  = ( P_zero & ~E_zero).sum()   # P missing, E present
e_only_zero  = (~P_zero &  E_zero).sum()   # E missing, P present
total        = n_areas * n_years

zero_summary = pd.DataFrame({
    'state':   ['both zero', 'both non-zero', 'P=0 E≠0', 'P≠0 E=0'],
    'count':   [both_zero, both_nonzero, p_only_zero, e_only_zero],
    'pct':     [100 * x / total for x in
                [both_zero, both_nonzero, p_only_zero, e_only_zero]],
})
print('\n── Zero-rate joint states ───────────────────────────────────────────')
print(zero_summary.to_string(index=False, float_format='{:.1f}'.format))

# Zero rates by year
p_zero_by_year = P_zero.mean(axis=0)
e_zero_by_year = E_zero.mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(years, p_zero_by_year * 100, 'o-', color='darkorange', label='P_obs zero rate')
ax.plot(years, e_zero_by_year * 100, 's-', color='forestgreen', label='E_obs zero rate')
ax.set_ylabel('% of areas with zero observation')
ax.set_xlabel('Year')
ax.set_title('Zero rate by year: P (planning) vs E (BEN)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Zero rates by area — histogram
p_zero_by_area = P_zero.mean(axis=1)
e_zero_by_area = E_zero.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(p_zero_by_area * 100, bins=20, color='darkorange', edgecolor='none')
axes[0].set_title('P_obs: fraction of years with zero (per area)')
axes[0].set_xlabel('% years zero')
axes[1].hist(e_zero_by_area * 100, bins=20, color='forestgreen', edgecolor='none')
axes[1].set_title('E_obs: fraction of years with zero (per area)')
axes[1].set_xlabel('% years zero')
for ax in axes:
    ax.set_ylabel('# areas')
plt.tight_layout()
plt.show()

# %% ── 2. P vs E when both non-zero: scale and sign ──────────────────────────

mask_both = ~P_zero & ~E_zero
P_nz = P[mask_both]
E_nz = E[mask_both]
diff = P_nz - E_nz
ratio = P_nz / E_nz

print('\n── P vs E when both non-zero ────────────────────────────────────────')
print(f'  N observations:  {mask_both.sum()}')
print(f'  P mean:          {P_nz.mean():.2f}   E mean: {E_nz.mean():.2f}')
print(f'  Diff (P-E) mean: {diff.mean():.2f}   std: {diff.std():.2f}')
print(f'  Ratio P/E: median={np.median(ratio):.2f}  p25={np.percentile(ratio,25):.2f}  p75={np.percentile(ratio,75):.2f}')
print(f'  Sign agreement: {(np.sign(P_nz) == np.sign(E_nz)).mean()*100:.1f}% of both-nonzero obs')

fig, axes = plt.subplots(1, 3, figsize=(13, 3))

# scatter
lim = np.percentile(np.abs(np.concatenate([P_nz, E_nz])), 98)
axes[0].scatter(E_nz, P_nz, alpha=0.15, s=5, color='steelblue')
axes[0].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='P=E')
axes[0].set_xlim(-lim, lim)
axes[0].set_ylim(-lim, lim)
axes[0].set_xlabel('E_obs')
axes[0].set_ylabel('P_obs')
axes[0].set_title('P vs E (both non-zero)')
axes[0].legend(fontsize=8)

# difference distribution
axes[1].hist(diff, bins=60, color='steelblue', edgecolor='none')
axes[1].axvline(0, color='red', linewidth=1)
axes[1].axvline(diff.mean(), color='black', linewidth=1, linestyle='--',
                label=f'mean={diff.mean():.1f}')
axes[1].set_xlabel('P_obs − E_obs')
axes[1].set_title('Difference distribution (P−E)')
axes[1].legend(fontsize=8)

# ratio distribution (clipped)
r_clip = np.clip(ratio, 0.05, 20)
axes[2].hist(np.log10(r_clip), bins=60, color='steelblue', edgecolor='none')
axes[2].axvline(0, color='red', linewidth=1, label='P/E = 1')
axes[2].set_xlabel('log₁₀(P/E)')
axes[2].set_title('P/E ratio (log scale)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% ── 2b. Isolate the log₁₀(P/E) ≈ −1.5 spike ─────────────────────────────
# The P/E ratio plot shows a secondary spike around log10(P/E) ≈ -1.5 (P/E ≈ 0.03),
# which is too consistent to be random noise. Isolate these records and examine
# what areas and years they come from.

LOW_RATIO_LO = -2.0   # log10(P/E) lower bound for spike
LOW_RATIO_HI = -1.0   # log10(P/E) upper bound for spike

log_ratio = np.where(mask_both, np.log10(np.abs(P / np.where(E == 0, np.nan, E))), np.nan)
spike_mask = mask_both & (log_ratio >= LOW_RATIO_LO) & (log_ratio <= LOW_RATIO_HI)

n_spike = spike_mask.sum()
print(f'\n── log₁₀(P/E) spike [{LOW_RATIO_LO}, {LOW_RATIO_HI}] ─────────────────────────────')
print(f'  Records in spike:  {n_spike}  ({100*n_spike/mask_both.sum():.1f}% of both-nonzero)')

if n_spike > 0:
    P_spike = P[spike_mask]
    E_spike = E[spike_mask]
    print(f'  P in spike:  mean={P_spike.mean():.2f}  median={np.median(P_spike):.2f}  '
          f'p25={np.percentile(P_spike,25):.2f}  p75={np.percentile(P_spike,75):.2f}')
    print(f'  E in spike:  mean={E_spike.mean():.2f}  median={np.median(E_spike):.2f}  '
          f'p25={np.percentile(E_spike,25):.2f}  p75={np.percentile(E_spike,75):.2f}')

    # Which years are overrepresented?
    spike_by_year = spike_mask.sum(axis=0)
    both_by_year  = mask_both.sum(axis=0)
    print(f'\n  Spike records by year:')
    for y, yr in enumerate(years):
        pct = 100 * spike_by_year[y] / both_by_year[y] if both_by_year[y] > 0 else 0
        print(f'    {yr}: {spike_by_year[y]:4d} / {both_by_year[y]:4d}  ({pct:.1f}%)')

    # How many distinct areas are affected?
    areas_with_spike = spike_mask.any(axis=1).sum()
    always_spike = (spike_mask.sum(axis=1) == (~E_zero & ~P_zero).sum(axis=1)) & \
                   ((~E_zero & ~P_zero).sum(axis=1) > 0)
    print(f'\n  Distinct areas with ≥1 spike record: {areas_with_spike}  '
          f'({100*areas_with_spike/n_areas:.1f}%)')
    print(f'  Areas where ALL both-nonzero obs are in spike: {always_spike.sum()}')

    # Distribution of spike records vs non-spike in P/E scatter
    non_spike_mask = mask_both & ~spike_mask
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    lim = np.percentile(np.abs(np.concatenate([P[mask_both], E[mask_both]])), 98)
    axes[0].scatter(E[non_spike_mask], P[non_spike_mask],
                    alpha=0.1, s=4, color='steelblue', label='Normal')
    axes[0].scatter(E[spike_mask], P[spike_mask],
                    alpha=0.4, s=8, color='firebrick', label=f'Spike (log P/E ∈ [{LOW_RATIO_LO},{LOW_RATIO_HI}])')
    axes[0].plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel('E_obs')
    axes[0].set_ylabel('P_obs')
    axes[0].set_title('P vs E: spike records highlighted')
    axes[0].legend(fontsize=8)

    axes[1].bar(years, spike_by_year, color='firebrick', label='Spike records')
    axes[1].bar(years, both_by_year - spike_by_year, bottom=spike_by_year,
                color='steelblue', label='Normal both-nonzero')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('# records')
    axes[1].set_title('Spike records by year')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# %% ── 3. Disagreement typology per area ─────────────────────────────────────
# Classify each area by its dominant P/E disagreement pattern.

p_miss_rate  = (P_zero & ~E_zero).sum(axis=1) / np.maximum((~E_zero).sum(axis=1), 1)
e_miss_rate  = (~P_zero &  E_zero).sum(axis=1) / np.maximum((~P_zero).sum(axis=1), 1)

# Conditional bias: mean(P - E) when both non-zero, per area
cond_diff = np.where(mask_both, P - E, np.nan)
area_bias  = np.nanmean(cond_diff, axis=1)
area_bias_abs = np.nanmean(np.abs(cond_diff), axis=1)

# Fraction of years where both are non-zero
both_nonzero_rate = mask_both.mean(axis=1)

area_df = pd.DataFrame({
    'p_miss_rate':       p_miss_rate,
    'e_miss_rate':       e_miss_rate,
    'both_nonzero_rate': both_nonzero_rate,
    'area_bias':         area_bias,
    'area_bias_abs':     area_bias_abs,
})

print('\n── Per-area disagreement summary ────────────────────────────────────')
print(area_df.describe().round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3))

axes[0].hist(p_miss_rate * 100, bins=20, color='darkorange', edgecolor='none')
axes[0].set_title('P missingness rate\n(P=0 when E≠0, per area)')
axes[0].set_xlabel('% of E-active years where P=0')
axes[0].set_ylabel('# areas')

axes[1].hist(e_miss_rate * 100, bins=20, color='forestgreen', edgecolor='none')
axes[1].set_title('E missingness rate\n(E=0 when P≠0, per area)')
axes[1].set_xlabel('% of P-active years where E=0')
axes[1].set_ylabel('# areas')

clip = 50
area_bias_clipped = np.clip(area_bias, -clip, clip)
axes[2].hist(area_bias_clipped, bins=40, color='steelblue', edgecolor='none')
axes[2].axvline(0, color='red', linewidth=1)
n_clipped = (np.abs(area_bias) > clip).sum()
axes[2].set_title(f'Conditional bias per area\nmean(P − E | both non-zero), clipped ±{clip}\n'
                  f'({n_clipped} areas outside ±{clip} not shown)')
axes[2].set_xlabel('Mean P − E')
axes[2].set_ylabel('# areas')

plt.tight_layout()
plt.show()

# %% ── 4. Lag signature: cross-correlation of P and E within areas ────────────
# For each area with enough non-zero years, compute cross-correlation
# of P_obs[a,:] and E_obs[a,:] at lags -MAX_LAG to +MAX_LAG.
# Positive lag k means P leads E by k years (P records earlier events later).

eligible = both_nonzero_rate >= (MIN_NONZERO / n_years)
print(f'\n── Lag analysis: {eligible.sum()} of {n_areas} areas have ≥{MIN_NONZERO} jointly non-zero years')

lags = np.arange(-MAX_LAG, MAX_LAG + 1)
xcorr_matrix = np.full((eligible.sum(), len(lags)), np.nan)

for i, a in enumerate(np.where(eligible)[0]):
    p = P[a] - P[a].mean()
    e = E[a] - E[a].mean()
    norm = np.sqrt((p**2).sum() * (e**2).sum())
    if norm < 1e-8:
        continue
    for j, lag in enumerate(lags):
        if lag >= 0:
            xcorr_matrix[i, j] = (p[lag:] * e[:n_years - lag]).sum() / norm if n_years > lag else np.nan
        else:
            k = -lag
            xcorr_matrix[i, j] = (p[:n_years - k] * e[k:]).sum() / norm if n_years > k else np.nan

mean_xcorr = np.nanmean(xcorr_matrix, axis=0)
p25_xcorr  = np.nanpercentile(xcorr_matrix, 25, axis=0)
p75_xcorr  = np.nanpercentile(xcorr_matrix, 75, axis=0)
peak_lag   = lags[np.argmax(mean_xcorr)]

print(f'  Mean cross-correlation peaks at lag = {peak_lag} year(s)')
print(f'  (positive lag: E leads P, i.e. BEN records completions earlier than planning does)')

fig, ax = plt.subplots(figsize=(8, 3))
ax.fill_between(lags, p25_xcorr, p75_xcorr, alpha=0.25, color='steelblue', label='IQR across areas')
ax.plot(lags, mean_xcorr, 'o-', color='steelblue', linewidth=1.5, label='Mean xcorr')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.axvline(peak_lag, color='red', linewidth=1, linestyle=':', label=f'Peak lag={peak_lag}')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Lag (years, positive = E leads P / planning lags behind BEN)')
ax.set_ylabel('Cross-correlation')
ax.set_title('P–E cross-correlation by lag\n(mean ± IQR across areas)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% ── 5. Heterogeneity: do areas cluster into disagreement types? ────────────
# Scatter each area by its (p_miss_rate, conditional bias) to see if
# there are natural groups.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sc = axes[0].scatter(
    area_df['p_miss_rate'] * 100,
    area_df['area_bias'],
    c=area_df['both_nonzero_rate'],
    cmap='viridis', alpha=0.6, s=15,
)
plt.colorbar(sc, ax=axes[0], label='Both non-zero rate')
axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
axes[0].set_xlabel('P missingness rate (%)')
axes[0].set_ylabel('Conditional bias (mean P−E)')
axes[0].set_title('Area-level disagreement space')

# Rank areas by P missingness — show E and P profiles for representative areas
miss_ranks = area_df['p_miss_rate'].values
low_miss   = np.argsort(miss_ranks)[:3]
high_miss  = np.argsort(miss_ranks)[-3:]
sample_areas = np.concatenate([low_miss, high_miss])
labels = ['low-miss'] * 3 + ['high-miss'] * 3

for idx, (a, label) in enumerate(zip(sample_areas, labels)):
    color = 'steelblue' if 'low' in label else 'darkorange'
    axes[1].plot(years, P[a], '-', color=color, alpha=0.7, linewidth=1)
    axes[1].plot(years, E[a], '--', color=color, alpha=0.7, linewidth=1)

axes[1].plot([], [], '-',  color='steelblue',   label='P (low miss areas)')
axes[1].plot([], [], '--', color='steelblue',   label='E (low miss areas)')
axes[1].plot([], [], '-',  color='darkorange',  label='P (high miss areas)')
axes[1].plot([], [], '--', color='darkorange',  label='E (high miss areas)')
axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Observations')
axes[1].set_title('P and E profiles: low vs high P-missingness areas')
axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.show()

# %% ── 6. Systematic vs random missingness ───────────────────────────────────
# For each area, examine whether P=0 years cluster (systematic absence)
# or are scattered (random year-by-year dropout).
#
# Two diagnostics:
#   a) Run-length distribution: how many areas have long unbroken runs of P=0
#      when E is active — a sign of systematic rather than random absence.
#   b) Missingness autocorrelation: is P missing in year t predictive of
#      P missing in year t+1 (within the same area)?

def _run_lengths(binary_row):
    """Return lengths of all runs of True in a 1-D boolean array."""
    runs = []
    count = 0
    for v in binary_row:
        if v:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs

# Missingness indicator: P=0 when E≠0
miss_indicator = P_zero & ~E_zero   # (n_areas, n_years)

all_runs = []
for a in range(n_areas):
    all_runs.extend(_run_lengths(miss_indicator[a]))

all_runs = np.array(all_runs) if all_runs else np.array([0])

# Areas where P is missing for ALL E-active years (completely dark to planning)
e_active_years = (~E_zero).sum(axis=1)
p_miss_all     = (miss_indicator.sum(axis=1) == e_active_years) & (e_active_years > 0)
p_miss_never   = (miss_indicator.sum(axis=1) == 0) & (e_active_years > 0)
p_miss_partial = (~p_miss_all & ~p_miss_never) & (e_active_years > 0)

print('\n── Systematic vs random P missingness ───────────────────────────────')
print(f'  Areas where P is ALWAYS missing when E active:  '
      f'{p_miss_all.sum():4d}  ({100*p_miss_all.mean():.1f}%)')
print(f'  Areas where P is NEVER missing when E active:   '
      f'{p_miss_never.sum():4d}  ({100*p_miss_never.mean():.1f}%)')
print(f'  Areas with PARTIAL missingness:                 '
      f'{p_miss_partial.sum():4d}  ({100*p_miss_partial.mean():.1f}%)')
print(f'\n  Run-length distribution (consecutive P=0 years when E active):')
for length in range(1, min(n_years + 1, 8)):
    n = (all_runs == length).sum()
    print(f'    Run of {length}: {n:5d}  ({100*n/len(all_runs):.1f}%)')

# Missingness autocorrelation within areas
# lag-1: does P_miss[t] predict P_miss[t+1]?
miss_t  = miss_indicator[:, :-1].ravel().astype(float)
miss_t1 = miss_indicator[:,  1:].ravel().astype(float)
autocorr = np.corrcoef(miss_t, miss_t1)[0, 1]
print(f'\n  Lag-1 autocorrelation of P missingness within areas: {autocorr:.3f}')
print(f'  (0 = random year-to-year, 1 = perfectly persistent)')

fig, axes = plt.subplots(1, 2, figsize=(11, 3))

run_counts = [(all_runs == k).sum() for k in range(1, n_years + 1)]
axes[0].bar(range(1, n_years + 1), run_counts, color='darkorange', edgecolor='none')
axes[0].set_xlabel('Consecutive P=0 years (run length)')
axes[0].set_ylabel('# runs')
axes[0].set_title('Run-length distribution of P missingness\n(when E is active)')
axes[0].set_xticks(range(1, n_years + 1))

labels = ['Always\nmissing', 'Partial\nmissing', 'Never\nmissing']
sizes  = [p_miss_all.sum(), p_miss_partial.sum(), p_miss_never.sum()]
colors = ['#d62728', '#ff7f0e', '#2ca02c']
axes[1].bar(labels, sizes, color=colors, edgecolor='none')
axes[1].set_ylabel('# areas')
axes[1].set_title('Area-level P missingness type\n(relative to E-active years)')

plt.tight_layout()
plt.show()

# %% ── 7. How much missingness could lag explain? ─────────────────────────────
# For each cell where P=0 and E≠0 (a "P missing" event), check whether a
# non-zero P appears in the same area within the next MAX_LAG_EXPLAIN years.
# If so, the lag hypothesis *could* explain the zero — the completion may have
# been recorded late rather than missing permanently.
#
# This is an upper bound: a nearby non-zero P doesn't prove it's the same
# completion, but persistent zeros over many years clearly cannot be lag.

MAX_LAG_EXPLAIN = 3

lag_explainable = np.zeros(n_areas, dtype=float)
lag_unexplainable = np.zeros(n_areas, dtype=float)

for a in range(n_areas):
    for t in range(n_years):
        if not miss_indicator[a, t]:
            continue
        # Check if a non-zero P appears within MAX_LAG_EXPLAIN years
        future_p = P[a, t+1 : t+1+MAX_LAG_EXPLAIN]
        if len(future_p) > 0 and np.any(np.abs(future_p) >= ZERO_THRESHOLD):
            lag_explainable[a] += 1
        else:
            lag_unexplainable[a] += 1

total_missing    = miss_indicator.sum()
n_lag_explain    = lag_explainable.sum()
n_lag_unexplain  = lag_unexplainable.sum()

# Areas where ALL missingness is lag-unexplainable (never a future P recovery)
always_unexplained = (lag_explainable == 0) & ((lag_explainable + lag_unexplainable) > 0)

print(f'\n── Lag-explainable P missingness (within {MAX_LAG_EXPLAIN} years) ──────────────')
print(f'  Total P-missing cells:                   {total_missing:5d}')
print(f'  Followed by non-zero P within {MAX_LAG_EXPLAIN} yrs:    {int(n_lag_explain):5d}  '
      f'({100*n_lag_explain/total_missing:.1f}%)  ← upper bound on lag-explainable')
print(f'  No recovery within {MAX_LAG_EXPLAIN} yrs:               {int(n_lag_unexplain):5d}  '
      f'({100*n_lag_unexplain/total_missing:.1f}%)  ← cannot be lag')
print(f'  Areas with zero P-recovery ever:         {always_unexplained.sum():5d}  '
      f'({100*always_unexplained.mean():.1f}%)')

# Break down by run length: short runs more likely to be lag
fig, axes = plt.subplots(1, 2, figsize=(11, 3))

# Per-area: fraction of P-missing cells that are lag-explainable
total_miss_per_area = lag_explainable + lag_unexplainable
frac_explainable = np.where(total_miss_per_area > 0,
                             lag_explainable / total_miss_per_area, np.nan)
frac_valid = frac_explainable[~np.isnan(frac_explainable)]

axes[0].hist(frac_valid * 100, bins=20, color='steelblue', edgecolor='none')
axes[0].axvline(np.nanmean(frac_explainable) * 100, color='red', linewidth=1,
                linestyle='--', label=f'mean={np.nanmean(frac_explainable)*100:.0f}%')
axes[0].set_xlabel(f'% of P-missing cells with P recovery within {MAX_LAG_EXPLAIN} yrs')
axes[0].set_ylabel('# areas')
axes[0].set_title(f'Lag-explainability per area\n(upper bound, lag ≤ {MAX_LAG_EXPLAIN})')
axes[0].legend(fontsize=8)

# Lag-explainability by run length
run_explain = {k: {'yes': 0, 'no': 0} for k in range(1, n_years + 1)}
for a in range(n_areas):
    runs_with_pos = []
    count = 0
    start = None
    for t in range(n_years):
        if miss_indicator[a, t]:
            if count == 0:
                start = t
            count += 1
        else:
            if count > 0:
                future_p = P[a, t : t + MAX_LAG_EXPLAIN]
                has_recovery = len(future_p) > 0 and np.any(np.abs(future_p) >= ZERO_THRESHOLD)
                run_explain[count]['yes' if has_recovery else 'no'] += 1
                count = 0
    if count > 0:
        run_explain[count]['no'] += 1   # run extends to end of window, no future data

max_run_show = 8
ks    = list(range(1, max_run_show + 1))
yes_n = [run_explain[k]['yes'] for k in ks]
no_n  = [run_explain[k]['no']  for k in ks]
x = np.arange(len(ks))
w = 0.4
axes[1].bar(x - w/2, yes_n, w, label='Recovery found', color='steelblue', edgecolor='none')
axes[1].bar(x + w/2, no_n,  w, label='No recovery',    color='darkorange', edgecolor='none')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(k) for k in ks])
axes[1].set_xlabel('Run length (consecutive P=0 years)')
axes[1].set_ylabel('# runs')
axes[1].set_title('Lag-explainability by run length')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% ── 8. Dark areas: no P or E signal ───────────────────────────────────────
# Identify areas that have no useful observation from either source across the
# full inference window. These areas must rely entirely on the census to identify
# their mean rate — a good model should handle them by falling back to the area
# mean without affecting inference in observed areas.
#
# Three tiers:
#   Fully dark  — P=0 and E=0 for every year
#   P-dark only — P=0 for every year, E has some signal
#   E-dark only — E=0 for every year, P has some signal

p_ever_nonzero = (~P_zero).any(axis=1)
e_ever_nonzero = (~E_zero).any(axis=1)

fully_dark  = ~p_ever_nonzero & ~e_ever_nonzero
p_dark_only = ~p_ever_nonzero &  e_ever_nonzero
e_dark_only =  p_ever_nonzero & ~e_ever_nonzero
observed    =  p_ever_nonzero &  e_ever_nonzero

print('\n── Dark area classification ─────────────────────────────────────────')
print(f'  Fully dark  (no P, no E):  {fully_dark.sum():5d}  ({100*fully_dark.mean():.1f}%)')
print(f'  P-dark only (no P, has E): {p_dark_only.sum():5d}  ({100*p_dark_only.mean():.1f}%)')
print(f'  E-dark only (no E, has P): {e_dark_only.sum():5d}  ({100*e_dark_only.mean():.1f}%)')
print(f'  Observed    (has P and E): {observed.sum():5d}  ({100*observed.mean():.1f}%)')

D = data['D']

print('\n── Census D values by area type ─────────────────────────────────────')
for label, mask in [('Fully dark',  fully_dark),
                    ('P-dark only', p_dark_only),
                    ('E-dark only', e_dark_only),
                    ('Observed',    observed)]:
    if mask.sum() == 0:
        continue
    d = D[mask]
    print(f'  {label:12s}  n={mask.sum():5d}  '
          f'median={np.median(d):6.1f}  mean={d.mean():6.1f}  '
          f'p25={np.percentile(d,25):6.1f}  p75={np.percentile(d,75):6.1f}  '
          f'pct_zero={100*(d==0).mean():.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# D distribution by area type
bins = np.linspace(D.min(), np.percentile(D, 99), 50)
for label, mask, color in [
    ('Observed',    observed,    'steelblue'),
    ('P-dark only', p_dark_only, 'darkorange'),
    ('E-dark only', e_dark_only, 'forestgreen'),
    ('Fully dark',  fully_dark,  'firebrick'),
]:
    if mask.sum() == 0:
        continue
    axes[0].hist(D[mask], bins=bins, alpha=0.5, color=color,
                 label=f'{label} (n={mask.sum()})', density=True)

axes[0].set_xlabel('Census completions D (total over window)')
axes[0].set_ylabel('Density')
axes[0].set_title('D distribution by observation tier')
axes[0].legend(fontsize=8)

# CDF version for easier comparison
for label, mask, color in [
    ('Observed',    observed,    'steelblue'),
    ('P-dark only', p_dark_only, 'darkorange'),
    ('E-dark only', e_dark_only, 'forestgreen'),
    ('Fully dark',  fully_dark,  'firebrick'),
]:
    if mask.sum() == 0:
        continue
    d_sorted = np.sort(D[mask])
    axes[1].plot(d_sorted, np.linspace(0, 1, len(d_sorted)),
                 color=color, linewidth=1.5, label=label)

axes[1].set_xlabel('Census completions D (total over window)')
axes[1].set_ylabel('Cumulative fraction')
axes[1].set_title('CDF of D by observation tier')
axes[1].legend(fontsize=8)
axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
plt.show()

# Spatial pattern: are dark areas concentrated in particular boroughs?
# Use LSOA codes from the data if available.
lsoa_codes = data['gdf']['LSOA21CD'].values if 'gdf' in data else None
if lsoa_codes is not None:
    dark_df = pd.DataFrame({
        'LSOA21CD': lsoa_codes,
        'D':        D,
        'tier': np.where(fully_dark,  'fully_dark',
                np.where(p_dark_only, 'p_dark_only',
                np.where(e_dark_only, 'e_dark_only', 'observed'))),
    })
    # Borough code = first 9 chars of LSOA (E09xxxxx + LSOA suffix → use first 3 chars of suffix)
    # LAD code embedded in LSOA21CD as E01xxxxxx → borough not directly readable.
    # Approximate: count dark areas per first-6-char prefix (ward-level grouping).
    dark_df['prefix'] = dark_df['LSOA21CD'].str[:6]
    tier_by_prefix = dark_df.groupby('prefix')['tier'].value_counts().unstack(fill_value=0)
    if 'fully_dark' in tier_by_prefix.columns:
        tier_by_prefix['pct_dark'] = (
            tier_by_prefix.get('fully_dark', 0) + tier_by_prefix.get('p_dark_only', 0)
        ) / tier_by_prefix.sum(axis=1)
        top_dark = tier_by_prefix.nlargest(10, 'pct_dark')[['fully_dark', 'p_dark_only', 'observed', 'pct_dark']]
        print('\n── Top 10 prefixes by dark-area concentration ───────────────────')
        print(top_dark.to_string(float_format='{:.2f}'.format))

# %% ── 9. Summary table ───────────────────────────────────────────────────────
# Two versions: full data and trimmed (dropping top/bottom 1% of each series)
# to separate the picture for typical areas from extreme outliers.

def _trimmed_stats(x, pct=1.0):
    """Return dict of stats after dropping the outer `pct`% on each side."""
    lo, hi = np.percentile(x, [pct, 100 - pct])
    xt = x[(x >= lo) & (x <= hi)]
    return {
        'n':      len(xt),
        'mean':   xt.mean(),
        'median': np.median(xt),
        'std':    xt.std(),
        'p25':    np.percentile(xt, 25),
        'p75':    np.percentile(xt, 75),
    }

TRIM_PCT = 1.0   # drop outer 1% on each side

diff_stats_full    = {'n': len(diff),    'mean': diff.mean(),    'median': np.median(diff),
                      'std': diff.std(), 'p25': np.percentile(diff, 25),
                      'p75': np.percentile(diff, 75)}
diff_stats_trimmed = _trimmed_stats(diff, TRIM_PCT)

abs_diff_full    = np.abs(diff)
abs_diff_trimmed_stats = _trimmed_stats(abs_diff_full, TRIM_PCT)

bias_valid = area_bias[~np.isnan(area_bias)]
bias_stats_full    = {'n': len(bias_valid), 'mean': bias_valid.mean(),
                      'median': np.median(bias_valid), 'std': bias_valid.std(),
                      'p25': np.percentile(bias_valid, 25),
                      'p75': np.percentile(bias_valid, 75)}
bias_stats_trimmed = _trimmed_stats(bias_valid, TRIM_PCT)

D_obs_stats_full    = {'n': observed.sum(), 'mean': D[observed].mean(),
                       'median': np.median(D[observed]), 'std': D[observed].std(),
                       'p25': np.percentile(D[observed], 25),
                       'p75': np.percentile(D[observed], 75)}
D_obs_stats_trimmed = _trimmed_stats(D[observed], TRIM_PCT)

def _fmt(s):
    return (f"n={s['n']:5d}  mean={s['mean']:7.2f}  median={s['median']:7.2f}  "
            f"std={s['std']:7.2f}  IQR=[{s['p25']:.1f}, {s['p75']:.1f}]")

print('\n══ P/E Disagreement Summary ══════════════════════════════════════════')
print(f'\n  Total (area × year) cells: {total}')
print(f'  Both zero:                 {both_zero:5d}  ({100*both_zero/total:.1f}%)')
print(f'  Both non-zero:             {both_nonzero:5d}  ({100*both_nonzero/total:.1f}%)')
print(f'  P=0, E≠0 (P missing):      {p_only_zero:5d}  ({100*p_only_zero/total:.1f}%)')
print(f'  P≠0, E=0 (E missing):      {e_only_zero:5d}  ({100*e_only_zero/total:.1f}%)')

print(f'\n  When both non-zero — P−E (observation level):')
print(f'    Full:          {_fmt(diff_stats_full)}')
print(f'    Trimmed {TRIM_PCT:.0f}%:    {_fmt(diff_stats_trimmed)}')

print(f'\n  When both non-zero — |P−E|:')
print(f'    Full:          {_fmt({"n": len(abs_diff_full), "mean": abs_diff_full.mean(), "median": np.median(abs_diff_full), "std": abs_diff_full.std(), "p25": np.percentile(abs_diff_full,25), "p75": np.percentile(abs_diff_full,75)})}')
print(f'    Trimmed {TRIM_PCT:.0f}%:    {_fmt(abs_diff_trimmed_stats)}')

print(f'\n  Sign agree rate (both non-zero): {(np.sign(P_nz)==np.sign(E_nz)).mean()*100:.1f}%')

print(f'\n  Per-area conditional bias (mean P−E when both non-zero):')
print(f'    Full:          {_fmt(bias_stats_full)}')
print(f'    Trimmed {TRIM_PCT:.0f}%:    {_fmt(bias_stats_trimmed)}')
print(f'    Areas outside ±50: {(np.abs(bias_valid) > 50).sum()}')

print(f'\n  Census D for observed areas:')
print(f'    Full:          {_fmt(D_obs_stats_full)}')
print(f'    Trimmed {TRIM_PCT:.0f}%:    {_fmt(D_obs_stats_trimmed)}')

print(f'\n  Cross-correlation peak lag: {peak_lag} year(s)')

print(f'\n  P missingness (E-active years where P=0):')
print(f'    Median across areas:     {np.median(p_miss_rate)*100:.1f}%')
print(f'    p75 across areas:        {np.percentile(p_miss_rate,75)*100:.1f}%')
print(f'    p90 across areas:        {np.percentile(p_miss_rate,90)*100:.1f}%')

print(f'\n  Lag-explainability of P missingness (within {MAX_LAG_EXPLAIN} yrs):')
print(f'    Upper bound:   {100*n_lag_explain/total_missing:.1f}%  '
      f'({int(n_lag_explain)} of {total_missing} P-missing cells)')
print(f'    Unexplainable: {100*n_lag_unexplain/total_missing:.1f}%')
