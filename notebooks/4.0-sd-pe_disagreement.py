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

axes[2].hist(area_bias, bins=40, color='steelblue', edgecolor='none')
axes[2].axvline(0, color='red', linewidth=1)
axes[2].set_title('Conditional bias per area\nmean(P − E | both non-zero)')
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

# %% ── 6. Summary table ───────────────────────────────────────────────────────

print('\n══ P/E Disagreement Summary ══════════════════════════════════════════')
print(f'\n  Total (area × year) cells: {total}')
print(f'  Both zero:                 {both_zero:5d}  ({100*both_zero/total:.1f}%)')
print(f'  Both non-zero:             {both_nonzero:5d}  ({100*both_nonzero/total:.1f}%)')
print(f'  P=0, E≠0 (P missing):      {p_only_zero:5d}  ({100*p_only_zero/total:.1f}%)')
print(f'  P≠0, E=0 (E missing):      {e_only_zero:5d}  ({100*e_only_zero/total:.1f}%)')
print(f'\n  When both non-zero:')
print(f'    Mean P−E:                {diff.mean():.2f}')
print(f'    |P−E| median:            {np.median(np.abs(diff)):.2f}')
print(f'    Sign agree rate:         {(np.sign(P_nz)==np.sign(E_nz)).mean()*100:.1f}%')
print(f'\n  Cross-correlation peak lag: {peak_lag} year(s)')
print(f'\n  P missingness (E-active years where P=0):')
print(f'    Median across areas:     {np.median(p_miss_rate)*100:.1f}%')
print(f'    p75 across areas:        {np.percentile(p_miss_rate,75)*100:.1f}%')
print(f'    p90 across areas:        {np.percentile(p_miss_rate,90)*100:.1f}%')
