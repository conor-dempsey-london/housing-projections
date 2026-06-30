import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from housing_projections.config import INFER_COLS_PLAN, INFER_COLS_BEN, INFER_YEARS


# ── Distribution over time ────────────────────────────────────────────────────

def plot_distributions_by_year(gdf):
    """
    Boxplots of planning and BEN distributions by year.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    gdf[INFER_COLS_PLAN].boxplot(ax=axes[0], showfliers=False)
    axes[0].set_title('Planning completions distribution by year')
    axes[0].set_ylabel('Net dwelling change')
    axes[0].spines[['top', 'right']].set_visible(False)

    gdf[INFER_COLS_BEN].boxplot(ax=axes[1], showfliers=False)
    axes[1].set_title('BEN estimates distribution by year')
    axes[1].set_ylabel('Net dwelling change')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_mean_trends(gdf):
    """
    Mean ± 1 SD trends for planning and BEN over time.
    """
    years = INFER_YEARS

    comp_mean = gdf[INFER_COLS_PLAN].mean()
    comp_std  = gdf[INFER_COLS_PLAN].std()
    ben_mean  = gdf[INFER_COLS_BEN].mean()
    ben_std   = gdf[INFER_COLS_BEN].std()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(years, comp_mean, color='steelblue', marker='o', label='Planning mean')
    ax.fill_between(years, comp_mean - comp_std, comp_mean + comp_std,
                    alpha=0.2, color='steelblue')

    ax.plot(years, ben_mean, color='coral', marker='o', label='BEN mean')
    ax.fill_between(years, ben_mean - ben_std, ben_mean + ben_std,
                    alpha=0.2, color='coral')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(years)
    ax.set_xlabel('Year')
    ax.set_ylabel('Net dwelling change')
    ax.set_title('Mean ± 1 SD across all areas')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_year_correlation(gdf):
    """
    Year-by-year cross-sectional correlation between planning and BEN.
    """
    correlations = pd.Series({
        yr: gdf[pc].corr(gdf[bc])
        for yr, pc, bc in zip(INFER_YEARS, INFER_COLS_PLAN, INFER_COLS_BEN)
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(INFER_YEARS, correlations, marker='o', color='steelblue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(INFER_YEARS)
    ax.set_xlabel('Year')
    ax.set_ylabel('Pearson r')
    ax.set_title('Cross-sectional correlation between planning and BEN by year')
    ax.set_ylim(-1, 1)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


# ── Autocorrelation ───────────────────────────────────────────────────────────

def compute_autocorrelations(gdf, cols_a, cols_b, max_lag=5,
                              n_permutations=1000, random_state=42):
    """
    Compute mean temporal autocorrelations for two series with
    permutation-based significance.

    Returns
    -------
    dict with keys 'obs_a', 'null_a', 'obs_b', 'null_b'
    """
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
                    np.corrcoef(values[:, t],
                                rng.permutation(values[:, t + lag]))[0, 1]
                    for t in range(years - lag)
                ]
                null_ac[lag].append(np.mean(perm_corrs))
        return observed_ac, null_ac

    obs_a, null_a = compute_ac(gdf[cols_a].values)
    obs_b, null_b = compute_ac(gdf[cols_b].values)

    return {'obs_a': obs_a, 'null_a': null_a,
            'obs_b': obs_b, 'null_b': null_b}


def plot_autocorrelations(ac_results, labels=('Series A', 'Series B'),
                           max_lag=5, alpha=0.05):
    """
    Line and marker plot of autocorrelations with permutation
    significance thresholds.
    """
    lags = list(range(1, max_lag + 1))

    obs_vals_a = [ac_results['obs_a'][l] for l in lags]
    obs_vals_b = [ac_results['obs_b'][l] for l in lags]
    upper_a    = [np.percentile(ac_results['null_a'][l],
                                100 * (1 - alpha / 2)) for l in lags]
    upper_b    = [np.percentile(ac_results['null_b'][l],
                                100 * (1 - alpha / 2)) for l in lags]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lags, obs_vals_a, marker='o', color='steelblue', label=labels[0])
    ax.plot(lags, obs_vals_b, marker='o', color='coral',     label=labels[1])
    ax.plot(lags, upper_a,   linestyle='--', color='black',  linewidth=1.0,
            label=f'{labels[0]} {int((1-alpha)*100)}% threshold')
    ax.plot(lags, upper_b,   linestyle=':',  color='black',  linewidth=1.0,
            label=f'{labels[1]} {int((1-alpha)*100)}% threshold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (years)')
    ax.set_ylabel('Mean correlation')
    ax.set_title('Temporal autocorrelation with permutation significance threshold')
    ax.set_xticks(lags)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Cross-correlation ─────────────────────────────────────────────────────────

def compute_crosscorrelations(gdf, cols_a, cols_b, max_lag=5,
                               n_permutations=1000, random_state=42):
    """
    Compute mean cross-correlations between two series with
    permutation-based significance.

    Returns
    -------
    dict with keys 'observed', 'null'
    """
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
                    np.corrcoef(a[:, t],
                                rng.permutation(b[:, t + lag]))[0, 1]
                    for t in range(a.shape[1] - lag)
                ]
            else:
                perm_corrs = [
                    np.corrcoef(a[:, t - lag],
                                rng.permutation(b[:, t]))[0, 1]
                    for t in range(b.shape[1] + lag)
                ]
            null_xc[lag].append(np.mean(perm_corrs))

    return {'observed': observed_xc, 'null': null_xc}


def compute_crosscorrelations_prewhitened(gdf, cols_a, cols_b, max_lag=5,
                                           n_permutations=1000,
                                           random_state=42, method='difference'):
    """
    Cross-correlations after prewhitening to remove autocorrelation.

    method : 'difference' — first-difference each series
             'ar'         — fit AR(1) per area and use residuals
    """
    a   = gdf[cols_a].values
    b   = gdf[cols_b].values
    rng = np.random.default_rng(random_state)

    def prewhiten(values):
        if method == 'difference':
            return np.diff(values, axis=1)
        elif method == 'ar':
            residuals = np.zeros((values.shape[0], values.shape[1] - 1))
            for i in range(values.shape[0]):
                y     = values[i]
                x_lag = y[:-1].reshape(-1, 1)
                x_cur = y[1:]
                beta  = np.linalg.lstsq(
                    np.column_stack([np.ones(len(x_lag)), x_lag]),
                    x_cur, rcond=None)[0]
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
                    np.corrcoef(a_white[:, t],
                                rng.permutation(b_white[:, t + lag]))[0, 1]
                    for t in range(a_white.shape[1] - lag)
                ]
            else:
                perm_corrs = [
                    np.corrcoef(a_white[:, t - lag],
                                rng.permutation(b_white[:, t]))[0, 1]
                    for t in range(b_white.shape[1] + lag)
                ]
            null_xc[lag].append(np.mean(perm_corrs))

    return {'observed': observed_xc, 'null': null_xc}


def plot_crosscorrelations(xc_results, labels=('Series A', 'Series B'),
                            max_lag=5, alpha=0.05):
    """
    Line and marker plot of cross-correlations with permutation
    significance thresholds (upper and lower).
    """
    lags     = list(range(-max_lag, max_lag + 1))
    obs_vals = [xc_results['observed'][l] for l in lags]
    upper    = [np.percentile(xc_results['null'][l],
                              100 * (1 - alpha / 2)) for l in lags]
    lower    = [np.percentile(xc_results['null'][l],
                              100 * (alpha / 2))     for l in lags]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, obs_vals, marker='o', color='steelblue', label='Observed')
    ax.plot(lags, upper,    linestyle='--', color='black', linewidth=1.0,
            label=f'{int((1-alpha)*100)}% threshold')
    ax.plot(lags, lower,    linestyle='--', color='black', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel(
        f'Lag (years)  —  positive: {labels[0]} leads {labels[1]}')
    ax.set_ylabel('Mean correlation')
    ax.set_title(f'Cross-correlation: {labels[0]} vs {labels[1]}')
    ax.set_xticks(lags)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()