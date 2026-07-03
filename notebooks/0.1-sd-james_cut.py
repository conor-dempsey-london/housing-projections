# %% [markdown]
# > **ARCHIVED** — This notebook uses the old `get_dwellings()` API and will
# > not run. Kept for historical reference. See `1.2-sd-inference.py` for the
# > current workflow.

# %%
import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pld_database.bep import run_bespoke_pipeline
from housing_projections.data import get_dwellings

dwellings, df_lsoa, year_cols_ben, year_cols_completion = get_dwellings()


# %%
def scatter_fit_equal(data=None, x=None, y=None, **kwargs):
    fgrid=sns.lmplot(dwellings, x='intercensal_change', y='intercensal_completions')
    fgrid.set(ylim=(-200,5000))

    min_val = dwellings[['intercensal_change','intercensal_completions']].min().min()
    max_val = dwellings[['intercensal_change','intercensal_completions']].max().max()

    fgrid.set(xlim=(min_val, max_val))
    fgrid.set(ylim=(min_val, max_val))

    fgrid.set(xlabel="Census dwellings change 2011-2021")
    fgrid.set(ylabel="Net PLD completions 2011-2021")

    fgrid.set(aspect='equal', adjustable='box')

    fgrid.ax.axline((0, 0), slope=1, color='k', ls='--')
    fgrid.ax.grid(True, axis='both', ls=':')

    return fgrid

# %%

# compare Ben's baseline with the PLD completions cut and the intercensal change
dwellings_changes = dwellings[[ 'intercensal_change', 'intercensal_completions', 'total_change_2011_to_2021_ben']]

dwellings_changes.rename(columns=
                         {
                             'intercensal_change': 'intercensal dwellings change',
                             'intercensal_completions': 'total planning completions',
                             'total_change_2011_to_2021_ben': 'current estimate (from Ben)'
                         }, inplace=True)

def corrplot(x, y, **kwargs):
    ax = plt.gca()
    sns.regplot(x=x, y=y, ax=ax, **kwargs)
    
    mask = x.notna() & y.notna()
    r, _ = stats.pearsonr(x[mask], y[mask])
    
    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
    )

g = sns.PairGrid(
    dwellings_changes,
    diag_sharey=False, corner=True)
g.map_diag(sns.histplot)
g.map_offdiag(corrplot)
g.savefig('../figures/current_v_pld_v_intercensal.png')

# Compute the correlation matrix
corr = dwellings_changes.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0.76,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# %%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

X = dwellings

y = dwellings['intercensal_change']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_ben, X_test_ben = X_train[['total_change_2011_to_2021_ben']], X_test[['total_change_2011_to_2021_ben']]
X_train_pld, X_test_pld = X_train[['intercensal_completions']], X_test[['intercensal_completions']]
X_train_sums, X_test_sums = X_train[['total_change_2011_to_2021_ben', 'intercensal_completions' ]], X_test[['total_change_2011_to_2021_ben', 'intercensal_completions' ]]

drop_columns = ['LSOA21CD', 'intercensal_change', 'dwellings_2011', 'dwellings_2021', 'intercensal_completions', 'total_change_2011_to_2021_ben']

X_train = X_train.drop(
    columns=drop_columns
)

X_test = X_test.drop(
    columns=drop_columns
)


# build pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

# fit and evaluate
pipeline.fit(X_train_sums, y_train)
y_pred_sum = pipeline.predict(X_test_sums)

# fit and evaluate
pipeline.fit(X_train_ben, y_train)
y_pred_ben = pipeline.predict(X_test_ben)

# fit and evaluate
pipeline.fit(X_train_pld, y_train)
y_pred_pld = pipeline.predict(X_test_pld)

# fit and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

y_pred_full = pipeline.predict(
    X.drop(
        columns=drop_columns
    )
)

r2 = r2_score(y_test, y_pred)
r2_sum = r2_score(y_test, y_pred_sum)
r2_ben = r2_score(y_test, y_pred_ben)
r2_pld = r2_score(y_test, y_pred_pld)

score_str = (
    f"R²:            {r2:.3f}\n"
    f"R² totals:     {r2_sum:.3f}\n"
    f"R² current:    {r2_ben:.3f}\n"
    f"R² PLD sum:    {r2_pld:.3f}"
)
print(score_str)

# plot predicted vs actual
fig, ax = plt.subplots(figsize=(8, 6))

ax.text(
    0.05, 0.95,          # x, y in axes coordinates (0–1)
    score_str,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
)

sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)

# perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")

ax.set_xlabel("actual intercensal change")
ax.set_ylabel("predicted intercensal change")
ax.set_title("Predicted vs actual intercensal changes (across LSOAs)")
plt.tight_layout()
plt.show()

fig.savefig('../figures/predicted_intercensal_change_linreg.png')

# %%

coefs = pd.Series(
    pipeline.named_steps["model"].coef_,
    index=X_train.columns
).sort_values(key=abs, ascending=False)

print("Positive coefficients:")
print(coefs[coefs > 0])

print("\nNegative coefficients:")
print(coefs[coefs < 0])

# %%

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# residuals vs predicted
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[0])
axes[0].axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
axes[0].set_xlabel("predicted intercensal change")
axes[0].set_ylabel("residual")
axes[0].set_title("residuals vs predicted")

# distribution of residuals
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_xlabel("residual")
axes[1].set_title("residual distribution")

plt.tight_layout()
plt.show()

fig.savefig('../figures/residuals_analysis_intercensal_change_linreg.png')


# %%
import esda
import libpysal

df_lsoa['residuals'] = y - y_pred_full

df_lsoa_x = df_lsoa.loc[df_lsoa['target_id'].isin(X['LSOA21CD']), :]

ax = df_lsoa_x.plot(
    column="residuals",
    scheme="Quantiles",
    k=5,
    cmap="GnBu",
    legend=True,
    figsize=(9, 9),
)

ax.set_axis_off()
plt.title("Residuals (Quintiles)")

# %%

g = libpysal.graph.Graph.build_contiguity(df_lsoa_x, rook=False)
gr = g.transform("r")
ylag = gr.lag(df_lsoa_x["residuals"])

ax = df_lsoa_x.plot(
    ylag,
    cmap="GnBu",
    linewidth=0.1,
    edgecolor="white",
    scheme="quantiles",
    legend=True,
    figsize=(9, 9),
)

ax.set_axis_off()
plt.title("spatially smoothed residuals")

plt.show()
ax.figure.savefig('../figures/residuals_spatial.png')


mi = esda.moran.Moran(df_lsoa_x["residuals"], gr)

moran_str = (
    f"Moran's I: {mi.I:.3f}\n"
    f"p-value:   {mi.p_sim:.3f}"
)

print(moran_str)

ax=sns.kdeplot(mi.sim, fill=True)
plt.vlines(mi.I, 0, 1, color="r")
plt.vlines(mi.EI, 0, 1)
plt.xlabel("Moran's I")

ax.text(
    0.65, 0.95,          # x, y in axes coordinates (0–1)
    moran_str,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
)

plt.savefig('../figures/morans_i_test_on_residuals.png', bbox_inches='tight')


# %%
models = {
    "Linear combination": y_pred,
    "Just the two sums": y_pred_sum,
    "Current approach": y_pred_ben,
    "PLD completions": y_pred_pld,
}

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True, sharex=True)

min_val = min(y_test.min(), min(p.min() for p in models.values()))
max_val = max(y_test.max(), max(p.max() for p in models.values()))

for ax, (name, y_pred_next) in zip(axes, models.items()):
    r2_next = r2_score(y_test, y_pred_next)

    sns.scatterplot(x=y_test, y=y_pred_next, alpha=0.4, ax=ax)
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    ax.set_title(f"{name}\nR²: {r2_next:.3f}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted" if ax == axes[0] else "")

plt.tight_layout()
plt.show()

fig.savefig('../current_vs_pld_vs_combo_pred.png')

# %%

from libpysal.weights import Queen

dwellings_spatial = gpd.GeoDataFrame(
    pd.merge(
        dwellings,
        df_lsoa[['target_id', 'geometry']], left_on='LSOA21CD', right_on='target_id')
).drop(columns='target_id')

dwellings_min_spatial = dwellings_spatial[['LSOA21CD', 
                'total_change_2011_to_2021_ben', 
                'intercensal_completions', 
                'intercensal_change',
                'geometry']]

def add_spatial_lag_features(gdf, feature_cols, use_index=False):

    w = Queen.from_dataframe(gdf, use_index=use_index)
    w.transform = 'r'
    W_dense = libpysal.weights.full(w)[0]

    gdf_out = gdf.copy()
    for col in feature_cols:
        gdf_out[f'lag_{col}'] = W_dense @ gdf[col].values

    return gdf_out

lag_cols = (
    ['total_change_2011_to_2021_ben', 'intercensal_completions'] + 
    year_cols_ben + 
    year_cols_completion
)

dwellings_min_spatial = add_spatial_lag_features(dwellings_min_spatial, ['total_change_2011_to_2021_ben', 'intercensal_completions'])

dwellings_spatial = add_spatial_lag_features(dwellings_spatial, lag_cols)

# %%

def fit_eval(X, y):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit and evaluate
    pipeline.fit(X_train, y_train)
    y_pred_fit = pipeline.predict(X_test)
    r2_fit = r2_score(y_test, y_pred_fit)

    return r2_fit, pipeline


y = dwellings_min_spatial['intercensal_change']
X = dwellings_min_spatial[
    [
        'total_change_2011_to_2021_ben',
        'intercensal_completions',
        'lag_total_change_2011_to_2021_ben',
        'lag_intercensal_completions'
    ]
]

r2_spatial, _ = fit_eval(X, y)

X_spatial_full = dwellings_spatial.drop(
        columns=[
            'LSOA21CD',
            'dwellings_2011',	
            'dwellings_2021',	
            'intercensal_change',
            'geometry'
        ],
    )

r2_spatial_full, pipeline_spatial_full = fit_eval(X_spatial_full, y)

print(f"R²:   {r2:.4f}")
print(f"R² sum:   {r2_sum:.4f}")
print(f"R² current:   {r2_ben:.4f}")
print(f"R² PLD sum:   {r2_pld:.4f}")
print(f"R² spatial:   {r2_spatial:.4f}")
print(f"R² spatial full:   {r2_spatial_full:.4f}")

y_pred    = pipeline_spatial_full.predict(X_spatial_full)   # or however you generate predictions
residuals = y - y_pred

# Build W on your full GeoDataFrame
w = Queen.from_dataframe(dwellings_min_spatial, use_index=False)
w.transform = 'r'

# Moran's I on residuals
moran = esda.moran.Moran(residuals, w)
print(f"Moran's I: {moran.I:.4f}  p={moran.p_sim:.4f}")


# %%
years = list(range(2012, 2022))

comp_mean = dwellings[year_cols_completion].mean()
comp_std  = dwellings[year_cols_completion].std()
ben_mean  = dwellings[year_cols_ben].mean()
ben_std   = dwellings[year_cols_ben].std()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(years, comp_mean, label='Completions', color='steelblue')
ax.fill_between(years,
                comp_mean - comp_std,
                comp_mean + comp_std,
                alpha=0.2, color='steelblue')

ax.plot(years, ben_mean, label='BEN', color='coral')
ax.fill_between(years,
                ben_mean - ben_std,
                ben_mean + ben_std,
                alpha=0.2, color='coral')

ax.set_xticks(years)
ax.set_xlabel('Year')
ax.legend()
ax.set_title('Mean ± 1 SD across all areas')
plt.tight_layout()
plt.show()


# %%


def compare_distributions(
        series1, 
        series2,
        series_names=['series_names[0]', 'series_names[1]']):

    flat1 = series1.ravel()
    flat2 = series2.ravel()

    n = len(flat1)
    print(f"Observations per series: {n:,}")
    print(f"Areas: {series1.shape[0]}, Time points: {series1.shape[1]}\n")

    print("=== Summary Statistics ===")
    for label, arr in [(series_names[0], flat1), (series_names[1], flat2)]:
        print(f"{label}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"median={np.median(arr):.4f}, "
              f"[{np.percentile(arr, 5):.3f}, {np.percentile(arr, 95):.3f}] (5–95%)")

    print("\n=== Statistical Tests ===")

    # KS test
    ks_stat, ks_p = stats.ks_2samp(flat1, flat2)
    print(f"Kolmogorov-Smirnov:  statistic={ks_stat:.6f}, p={ks_p:.4g}")

    # Mann-Whitney U
    mw_stat, mw_p = stats.mannwhitneyu(flat1, flat2, alternative="two-sided")
    print(f"Mann-Whitney U:      statistic={mw_stat:.4g}, p={mw_p:.4g}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("overall distribution comparison current vs completions", fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    bins = np.linspace(
        -50,
        50,
        80
    )
    ax.hist(flat1, bins=bins, alpha=0.5, density=True, label=series_names[0], color="steelblue")
    ax.hist(flat2, bins=bins, alpha=0.5, density=True, label=series_names[1], color="tomato")
    ax.set_title("histograms")
    ax.set_xlabel("annual dwelling change in LSOA")
    ax.set_ylabel("probability density")
    ax.set_xlim(-20,20)
    ax.legend()

    ax = axes[1]
    for arr, label, color in [(flat1, series_names[0], "steelblue"), (flat2, series_names[1], "tomato")]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        step = max(1, len(sorted_arr) // 2000)
        ax.plot(sorted_arr[::step], cdf[::step], label=label, color=color, lw=1.5)
    ax.set_title("cumulative distribution functions")
    ax.set_xlabel("annual dwelling change in LSOA")
    ax.set_ylabel("cumulative probability")
    ax.set_xlim(-20,20)
    ax.legend()

    ax = axes[2]
    quantiles = np.linspace(0.01, 0.99, 500)
    q1 = np.quantile(flat1, quantiles)
    q2 = np.quantile(flat2, quantiles)
    ax.scatter(q1, q2, s=5, alpha=0.6, color="mediumpurple")
    lims = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
    ax.plot(lims, lims, "k--", lw=1, label="y = x (identical distributions)")
    ax.set_title(f"percentile-percentile plot")
    ax.set_xlabel(f"{series_names[0]} percentiles")
    ax.set_ylabel(f"{series_names[1]} percentiles")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig, axes


fig, _ = compare_distributions(
    dwellings[year_cols_ben].values, 
    dwellings[year_cols_completion].values,
    series_names=['current estimate', 'completions'])

fig.savefig('../figures/compare_current_vs_completions_distros_global.png')

# %%

correlations = pd.Series({
    yr: dwellings[comp_col].corr(dwellings[ben_col])
    for yr, comp_col, ben_col in zip(years, year_cols_completion, year_cols_ben)
})

fig, ax = plt.subplots(figsize=(10, 4))
correlations.plot(kind='bar', ax=ax, color='steelblue')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('correlation between completions and current estimates by year')
ax.set_xlabel('Year')
ax.set_ylabel('correlation coefficient')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

ax.figure.savefig('../figures/correlation_across_areas_by_year_current_vs_completions.png')

comp_flat = dwellings[year_cols_completion].values.ravel()
ben_flat  = dwellings[year_cols_ben].values.ravel()

r = pd.Series(comp_flat).corr(pd.Series(ben_flat))
print(f"overall correlation: {r:.4f}")

per_area_corr = pd.Series([
    pd.Series(row_comp).corr(pd.Series(row_ben))
    for row_comp, row_ben in zip(
        dwellings[year_cols_completion].values,
        dwellings[year_cols_ben].values
    )
])

stats_string = (
    f"mean per-area correlation:   {per_area_corr.mean():.2f}\n"
    f"median per-area correlation: {per_area_corr.median():.2f}\n"
    f"std:                         {per_area_corr.std():.2f}\n"
    f"% with r > 0.5:              {(per_area_corr > 0.5).mean()*100:.1f}%"
)

print(stats_string)

with sns.axes_style("white"):
    ax=per_area_corr.hist(bins=50)
    ax.text(
        0.5, 0.95,          # x, y in axes coordinates (0–1)
        stats_string,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )
    plt.xlabel('within-area correlation')
    plt.ylabel('# areas')
    plt.title("correlations between completions and Ben's current estimate across areas")
    plt.box(False)
    plt.show()
    ax.figure.savefig('../figures/correlations_between_change_series_by_area.png')


# %%

def mean_autocorrelation(df, cols, max_lag=5):
    """
    For each area, compute autocorrelation of its time series at each lag.
    Return the mean across areas.
    """
    series = df[cols].values   # shape (n_areas, 10)
    results = {}
    for lag in range(1, max_lag + 1):
        # correlate t with t+lag across all areas
        corrs = [
            pd.Series(series[:, t]).corr(pd.Series(series[:, t + lag]))
            for t in range(series.shape[1] - lag)
        ]
        results[lag] = np.mean(corrs)
    return pd.Series(results)

ac_comp = mean_autocorrelation(dwellings, year_cols_completion)
ac_ben  = mean_autocorrelation(dwellings, year_cols_ben)

fig, ax = plt.subplots(figsize=(8, 4))
ac_comp.plot(marker='o', label='completions', ax=ax)
ac_ben.plot(marker='o',  label='current estimate',         ax=ax)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('mean temporal autocorrelation across areas')
ax.set_xlabel('lag (years)')
ax.set_ylabel('mean corr. coeff.')
ax.legend()
plt.tight_layout()
plt.show()

ax.figure.savefig('../figures/autocorrelation_of_estimates.png')

# %%

def mean_cross_correlation(df, cols_a, cols_b, max_lag=5):
    """
    Cross-correlation between series A and series B at lags -max_lag to +max_lag.
    Positive lag: A leads B. Negative lag: B leads A.
    """
    a = df[cols_a].values
    b = df[cols_b].values
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corrs = [
                pd.Series(a[:, t]).corr(pd.Series(b[:, t + lag]))
                for t in range(a.shape[1] - lag)
            ]
        else:
            corrs = [
                pd.Series(a[:, t - lag]).corr(pd.Series(b[:, t]))
                for t in range(a.shape[1] + lag)
            ]
        results[lag] = np.mean(corrs)
    return pd.Series(results)

xcorr = mean_cross_correlation(dwellings, year_cols_completion, year_cols_ben)

fig, ax = plt.subplots(figsize=(10, 4))
xcorr.plot(kind='bar', ax=ax, color='steelblue')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('cross-correlation: completions vs current estimates (positive lag = completions lead)')
ax.set_xlabel('lag (years)')
ax.set_ylabel('mean corr. coeff.')
plt.tight_layout()
plt.show()

ax.figure.savefig('../figures/crosscorrelation_of_estimates.png')


# %%

def compute_autocorrelations(gdf, cols_a, cols_b, max_lag=5,
                              n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    def compute_ac(values):
        years       = values.shape[1]
        observed_ac = {}
        null_ac     = {lag: [] for lag in range(1, max_lag + 1)}
        for lag in range(1, max_lag + 1):
            obs_corrs = [
                np.corrcoef(values[:, t], values[:, t + lag])[0, 1]
                for t in range(years - lag)
            ]
            observed_ac[lag] = np.mean(obs_corrs)
            for _ in range(n_permutations):
                perm_corrs = [
                    np.corrcoef(values[:, t], rng.permutation(values[:, t + lag]))[0, 1]
                    for t in range(years - lag)
                ]
                null_ac[lag].append(np.mean(perm_corrs))
        return observed_ac, null_ac

    obs_a, null_a = compute_ac(gdf[cols_a].values)
    obs_b, null_b = compute_ac(gdf[cols_b].values)

    return {'obs_a': obs_a, 'null_a': null_a,
            'obs_b': obs_b, 'null_b': null_b}


def compute_crosscorrelations(gdf, cols_a, cols_b, max_lag=5,
                               n_permutations=1000, random_state=42):
    a   = gdf[cols_a].values
    b   = gdf[cols_b].values
    rng = np.random.default_rng(random_state)

    lags        = list(range(-max_lag, max_lag + 1))
    observed_xc = {}
    null_xc     = {lag: [] for lag in lags}

    for lag in lags:
        if lag >= 0:
            obs_corrs = [
                np.corrcoef(a[:, t], b[:, t + lag])[0, 1]
                for t in range(a.shape[1] - lag)
            ]
        else:
            obs_corrs = [
                np.corrcoef(a[:, t - lag], b[:, t])[0, 1]
                for t in range(b.shape[1] + lag)
            ]
        observed_xc[lag] = np.mean(obs_corrs)

        for _ in range(n_permutations):
            if lag >= 0:
                perm_corrs = [
                    np.corrcoef(a[:, t], rng.permutation(b[:, t + lag]))[0, 1]
                    for t in range(a.shape[1] - lag)
                ]
            else:
                perm_corrs = [
                    np.corrcoef(a[:, t - lag], rng.permutation(b[:, t]))[0, 1]
                    for t in range(b.shape[1] + lag)
                ]
            null_xc[lag].append(np.mean(perm_corrs))

    return {'observed': observed_xc, 'null': null_xc}

# %%

def plot_autocorrelations(ac_results, labels=('Series A', 'Series B'),
                           max_lag=5, alpha=0.05):
    lags = list(range(1, max_lag + 1))

    obs_vals_a = [ac_results['obs_a'][l] for l in lags]
    obs_vals_b = [ac_results['obs_b'][l] for l in lags]
    upper_a    = [np.percentile(ac_results['null_a'][l], 100 * (1 - alpha / 2)) for l in lags]
    upper_b    = [np.percentile(ac_results['null_b'][l], 100 * (1 - alpha / 2)) for l in lags]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lags, obs_vals_a, marker='o', color='steelblue', label=labels[0])
    ax.plot(lags, obs_vals_b, marker='o', color='coral',     label=labels[1])
    ax.plot(lags, upper_a,    linestyle='--', color='black', linewidth=1.0,
            label=f'{int((1-alpha)*100)}% threshold ({labels[0]})')
    ax.plot(lags, upper_b,    linestyle=':',  color='black', linewidth=1.0,
            label=f'{int((1-alpha)*100)}% threshold ({labels[1]})')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (years)')
    ax.set_ylabel('Mean correlation')
    ax.set_title('Temporal autocorrelation with permutation significance threshold')
    ax.set_xticks(lags)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_crosscorrelations(xc_results, labels=('Series A', 'Series B'),
                            max_lag=5, alpha=0.05):
    lags      = list(range(-max_lag, max_lag + 1))
    obs_vals  = [xc_results['observed'][l] for l in lags]
    upper     = [np.percentile(xc_results['null'][l], 100 * (1 - alpha / 2)) for l in lags]
    lower     = [np.percentile(xc_results['null'][l], 100 * (alpha / 2))     for l in lags]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, obs_vals, marker='o', color='steelblue', label='Observed')
    ax.plot(lags, upper,    linestyle='--', color='black', linewidth=1.0,
            label=f'{int((1-alpha)*100)}% threshold')
    ax.plot(lags, lower,    linestyle='--', color='black', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel(f'Lag (years)  —  positive: {labels[0]} leads {labels[1]}')
    ax.set_ylabel('Mean correlation')
    ax.set_title(f'Cross-correlation: {labels[0]} vs {labels[1]}')
    ax.set_xticks(lags)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax

# %%

ac_results = compute_autocorrelations(dwellings, year_cols_completion, year_cols_ben)
xc_results = compute_crosscorrelations(dwellings, year_cols_completion, year_cols_ben)

# %%
fig, _ = plot_autocorrelations(ac_results,  labels=('completions', 'current estimates'), alpha=0.01)
fig.savefig('../figures/estimates_autocorrelations.png')

fig, _ = plot_crosscorrelations(xc_results, labels=('completions', 'current estimates'), alpha=0.01)
fig.savefig('../figures/estimates_crosscorrelations.png')


# %%

def compute_crosscorrelations_prewhitened(gdf, cols_a, cols_b, max_lag=5,
                                           n_permutations=1000, random_state=42,
                                           method='difference'):
    """
    Computes cross-correlations after prewhitening to remove autocorrelation.

    method : 'difference' — first-difference each series (robust for short series)
             'ar'         — fit AR(1) per area and use residuals
    """
    a   = gdf[cols_a].values   # (n_areas, n_years)
    b   = gdf[cols_b].values
    rng = np.random.default_rng(random_state)

    def prewhiten(values):
        if method == 'difference':
            # First difference: x[t] - x[t-1], loses one time point
            return np.diff(values, axis=1)
        elif method == 'ar':
            # Fit AR(1) per area, return residuals
            residuals = np.zeros((values.shape[0], values.shape[1] - 1))
            for i in range(values.shape[0]):
                y     = values[i]
                x_lag = y[:-1].reshape(-1, 1)
                x_cur = y[1:]
                # Simple OLS AR(1)
                beta  = np.linalg.lstsq(
                    np.column_stack([np.ones(len(x_lag)), x_lag]),
                    x_cur, rcond=None
                )[0]
                residuals[i] = x_cur - beta[0] - beta[1] * x_lag.ravel()
            return residuals

    a_white = prewhiten(a)
    b_white = prewhiten(b)

    lags        = list(range(-max_lag, max_lag + 1))
    observed_xc = {}
    null_xc     = {lag: [] for lag in lags}

    for lag in lags:
        if lag >= 0:
            obs_corrs = [
                np.corrcoef(a_white[:, t], b_white[:, t + lag])[0, 1]
                for t in range(a_white.shape[1] - lag)
            ]
        else:
            obs_corrs = [
                np.corrcoef(a_white[:, t - lag], b_white[:, t])[0, 1]
                for t in range(b_white.shape[1] + lag)
            ]
        observed_xc[lag] = np.mean(obs_corrs)

        for _ in range(n_permutations):
            if lag >= 0:
                perm_corrs = [
                    np.corrcoef(a_white[:, t], rng.permutation(b_white[:, t + lag]))[0, 1]
                    for t in range(a_white.shape[1] - lag)
                ]
            else:
                perm_corrs = [
                    np.corrcoef(a_white[:, t - lag], rng.permutation(b_white[:, t]))[0, 1]
                    for t in range(b_white.shape[1] + lag)
                ]
            null_xc[lag].append(np.mean(perm_corrs))

    return {'observed': observed_xc, 'null': null_xc}


# Compute and plot both versions for comparison
xc_raw         = compute_crosscorrelations(dwellings, year_cols_completion, year_cols_ben)

# %%
xc_prewhitened = compute_crosscorrelations_prewhitened(
    dwellings, year_cols_completion,
    year_cols_ben, method='ar', max_lag=4)

# %%
plot_crosscorrelations(xc_raw,         labels=('Completions', 'current estimates'), max_lag=4)
plot_crosscorrelations(xc_prewhitened, labels=('Completions (differenced)',
                                                'current estimates (differenced)'), max_lag=4)

# %%

