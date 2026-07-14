"""
Grouped (leave-area-out) K-fold CV comparing AZ3 vs AZ0a vs the best M-class
model (M5, per the existing PSIS-LOO first pass — see
docs/az-family-work-plan.md Phase 6), on the 200-area development sample.

Why a hand-rolled SamplingWrapper rather than reusing the trace's own
`log_likelihood` group: none of these three models expose a per-area latent
`z` that stays meaningfully "in the model" once that area's own P/E cells are
removed from a refit — AZ0a/AZ3's z-prior is a fixed-form, per-area-independent
distribution (_build_zero_sum_z_prior: ZeroSumNormal(mu_area, sigma_delta),
both fully determined by that area's own D, not by anything fitted), and M5's
is a Normal(mu_area, sigma_slab) softly pulled toward D via a Gaussian
census-constraint potential. Both are simple enough that the CORRECT
predictive distribution for a held-out area's cells is analytically
recoverable from (a) the model's own fixed prior formula and (b) the refit's
posterior draws of shared/global hyperparameters (sigma_plan, sigma_ben, rho,
sigma_noise, sigma_slab, lambda_weights, alpha_spatial) — no need to keep the
held-out area physically inside the refit at all. See
docs/model-finalization-work-plan.md Task 2 for the fuller design rationale,
and the module docstrings below for each model's specific predictive formula.

Fold design: leave-AREA-out (group_by=area index), not leave-year-out — none
of these three models pool information across years (no AR/temporal-lag
sharing in AZ0a/AZ3; M5's lag kernel is per-area-independent given the shared
lambda_weights), so a year is never "help" for another year the way an area's
own observations help infer its neighbours (M5) or the shared noise/signal
split (AZ3). All three DO share information across areas (global sigma_plan/
sigma_ben/rho/etc., plus M5's explicit spatial term), which is exactly what
leave-area-out tests.

The log-likelihood scored is the JOINT P+E predictive density per held-out
cell (summed, not just P — the same "don't silently drop E" fix already
applied to the PSIS comparison in analysis.py, see az-family-work-plan.md
Phase 6), returned directly by each wrapper's log_likelihood__i — sidestepping
the trace's own separate P_like/E_like log_likelihood variables entirely.

Usage
-----
    pixi run python scripts/kfold_comparison.py --k 10 --draws 600 --tune 500
"""
import argparse
import json
import sys
import time
from pathlib import Path

import arviz as az
import arviz_stats as azs
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from arviz_stats.loo import SamplingWrapper  # noqa: E402

from housing_projections.config import DATA_PATH, INFER_YEARS  # noqa: E402
from housing_projections.data import (  # noqa: E402
    load_data,
    make_data_dict,
    select_spatial_sample,
)
from housing_projections.models.models import AZ0a, AZ3, M5  # noqa: E402
from housing_projections.outliers import apply_outlier_exclusion  # noqa: E402
from housing_projections.spatial import build_spatial_weights  # noqa: E402


# ── Shared numpy building blocks (mirror models.py's pytensor formulas) ────────

def studentt_logpdf(x, mu, sigma, nu):
    return stats.t.logpdf(x, df=nu, loc=mu, scale=sigma)


def studentt_cdf(x, mu, sigma, nu):
    return stats.t.cdf(x, df=nu, loc=mu, scale=sigma)


def zero_sum_z_new_draws(D_new, sigma_delta_new, n_years, n_draws, rng):
    """
    Exact prior draws of z for a held-out area under AZ0a/AZ3's
    _build_zero_sum_z_prior: z = D_new/n_years + ZeroSumNormal(sigma_delta_new).
    Neither term depends on any fitted parameter (floor/k are fixed model
    constants), so this needs no posterior draws at all — one batch of
    n_draws samples covers every posterior draw of the refit.
    """
    mu_area = D_new / n_years
    delta = pm.draw(
        pm.ZeroSumNormal.dist(sigma=sigma_delta_new, n_zerosum_axes=1,
                              shape=(n_draws, n_years)),
        random_seed=rng,
    )
    return mu_area + delta


def gaussian_conjugate_z_new_draws(D_new, sigma_slab_draws, sigma_census_new,
                                    n_years, rng):
    """
    Exact prior draws of z for a held-out area under M5's construction:
    z ~ iid Normal(mu_area, sigma_slab) per year, softly pulled toward D via
    a Gaussian census-constraint potential (_build_census_constraint).
    Because both prior and constraint are Gaussian and linear in z, the
    result is a closed-form multivariate Gaussian (verified numerically
    against brute-force Gaussian conditioning before use here — see
    docs/model-finalization-work-plan.md Task 2). The posterior mean is
    unchanged from the prior mean (mu_area exactly, since mu_area is already
    defined as D_new/n_years — the constraint only shrinks the covariance),
    so only the covariance's shrinkage factor needs computing per draw of
    the fitted sigma_slab.

    Parameters
    ----------
    sigma_slab_draws : (n_draws,) — one posterior draw per refit sample
    """
    n_draws = len(sigma_slab_draws)
    mu_area = D_new / n_years
    c = sigma_slab_draws**2 / (n_years * sigma_slab_draws**2 + sigma_census_new**2)
    d = (1 - np.sqrt(np.clip(1 - n_years * c, 0, None))) / n_years
    eps = rng.standard_normal((n_draws, n_years))
    return mu_area + sigma_slab_draws[:, None] * (
        eps - d[:, None] * eps.sum(axis=1, keepdims=True))


def lag_convolve_numpy(z, pre_inference, lambda_weights, max_lag, n_lags):
    """
    Pure-numpy reimplementation of models.py's _build_lag, for evaluating
    M5's lag-convolved P_mean outside a pm.Model context (needed to combine
    a refit's fitted z for training areas with freshly-drawn z for held-out
    areas within the same spatial-smearing step).

    z              : (n_draws, n_areas, n_years)
    pre_inference   : (n_areas, max_lag) — fixed, not draw-dependent
    lambda_weights : (n_draws, n_lags)
    Returns P_mean_temporal, shape (n_draws, n_areas, n_years).
    """
    n_draws, n_areas, n_years = z.shape
    pre_tiled = np.broadcast_to(pre_inference, (n_draws, n_areas, max_lag))
    z_padded = np.concatenate([pre_tiled, z], axis=2)  # (n_draws, n_areas, max_lag+n_years)

    P_mean = np.zeros((n_draws, n_areas, n_years))
    for k in range(n_lags):
        shifted = z_padded[:, :, (max_lag - k):(max_lag - k + n_years)]
        P_mean += shifted * lambda_weights[:, k][:, None, None]
    return P_mean


# ── Base wrapper: shared sel_observations / sample / get_inference_data ────────

class DwellingModelKFoldWrapper(SamplingWrapper):
    """
    Generic K-fold wrapper for any DwellingModel subclass whose z-prior is
    per-area independent given shared/global hyperparameters (true for
    AZ0a, AZ3, M5 — see module docstring). Handles the model-agnostic parts
    (splitting the 200-area data by held-out area, building+sampling
    ModelClass on the remainder); subclasses implement `_predict_loglik` for
    their own likelihood formula.
    """

    def __init__(self, model_class, data, n_years, sample_kwargs, rng_seed=0):
        super().__init__(model=model_class)
        self.model_class = model_class
        self.full_data = data
        self.n_areas = data['n_areas']
        self.n_years = n_years
        self.sample_kwargs = sample_kwargs
        self.rng_seed = rng_seed
        self.pit_records = []  # (area, year, source, pit_value) accumulated across folds

    def sel_observations(self, idx):
        idx = np.sort(np.asarray(idx))
        held_out_areas = np.unique(idx // self.n_years)
        train_areas = np.setdiff1d(np.arange(self.n_areas), held_out_areas)

        gdf = self.full_data['gdf']
        train_gdf = gdf.iloc[train_areas].reset_index(drop=True)
        train_data = make_data_dict(train_gdf)

        test_data = {
            'held_out_areas': held_out_areas,
            'train_areas': train_areas,
            'idx': idx,
        }
        return train_data, test_data

    def sample(self, modified_observed_data):
        m = self.model_class(modified_observed_data)
        m.build()
        m.sample(use_nutpie=True, **self.sample_kwargs)
        return m.trace

    def get_inference_data(self, fitted_model):
        return fitted_model

    def _record_pit(self, area_orig_idx, source, pit_values):
        for a, p in zip(area_orig_idx, pit_values):
            self.pit_records.append({'area_idx': int(a), 'source': source, 'pit': float(p)})


# ── AZ0a ────────────────────────────────────────────────────────────────────────

class AZ0aKFoldWrapper(DwellingModelKFoldWrapper):
    """AZ0a: plain StudentT(z, sigma_plan/sigma_ben) likelihood, z ~ AZ0a's
    zero-sum prior (fixed floor/k, no fitted dependence)."""

    def log_likelihood__i(self, excluded_obs, idata__i):
        held_out = excluded_obs['held_out_areas']
        D_full, P_full, E_full = (self.full_data['D'], self.full_data['P_obs'],
                                   self.full_data['E_obs'])

        post = idata__i.posterior
        sigma_plan = post['sigma_plan'].values.reshape(-1)
        sigma_ben = post['sigma_ben'].values.reshape(-1)
        n_draws = len(sigma_plan)
        rng = np.random.default_rng(self.rng_seed)

        loglik_rows, test_idx_order = [], []
        for a in held_out:
            D_a = D_full[a]
            sigma_delta_a = AZ0a.sigma_delta_floor + AZ0a.k_sigma_delta * abs(D_a)
            z_new = zero_sum_z_new_draws(D_a, sigma_delta_a, self.n_years, n_draws, rng)

            lp_P = studentt_logpdf(P_full[a][None, :], z_new, sigma_plan[:, None], AZ0a.nu_obs)
            lp_E = studentt_logpdf(E_full[a][None, :], z_new, sigma_ben[:, None], AZ0a.nu_obs)
            loglik_rows.append(lp_P + lp_E)  # (n_draws, n_years) joint P+E

            pit_P = studentt_cdf(P_full[a], z_new, sigma_plan[:, None], AZ0a.nu_obs).mean(axis=0)
            pit_E = studentt_cdf(E_full[a], z_new, sigma_ben[:, None], AZ0a.nu_obs).mean(axis=0)
            self._record_pit([a] * self.n_years, 'P', pit_P)
            self._record_pit([a] * self.n_years, 'E', pit_E)

            for t in range(self.n_years):
                test_idx_order.append(a * self.n_years + t)

        loglik = np.concatenate(loglik_rows, axis=1)  # (n_draws, n_held_out*n_years)
        order = np.argsort(test_idx_order)
        loglik = loglik[:, order]

        return xr.DataArray(loglik[None, :, :], dims=['chain', 'draw', 'test_idx'],
                            coords={'test_idx': np.sort(test_idx_order)})


# ── AZ3 ─────────────────────────────────────────────────────────────────────────

class AZ3KFoldWrapper(DwellingModelKFoldWrapper):
    """AZ3: same zero-sum z prior as AZ0a, but a 2-way noise/signal mixture
    likelihood (_build_noise_mixture_likelihood) instead of plain StudentT."""

    def log_likelihood__i(self, excluded_obs, idata__i):
        held_out = excluded_obs['held_out_areas']
        D_full, P_full, E_full = (self.full_data['D'], self.full_data['P_obs'],
                                   self.full_data['E_obs'])

        post = idata__i.posterior
        sigma_plan = post['sigma_plan'].values.reshape(-1)
        sigma_ben = post['sigma_ben'].values.reshape(-1)
        rho_P = post['rho_P'].values.reshape(-1)
        rho_E = post['rho_E'].values.reshape(-1)
        sigma_noise_P = post['sigma_noise_P'].values.reshape(-1)
        sigma_noise_E = post['sigma_noise_E'].values.reshape(-1)
        n_draws = len(sigma_plan)
        rng = np.random.default_rng(self.rng_seed)

        def mixture_logpdf(obs, z, sigma_obs, rho, sigma_noise):
            lp_signal = np.log(rho)[:, None] + studentt_logpdf(obs, z, sigma_obs[:, None], AZ3.nu_obs)
            lp_noise = np.log(1 - rho)[:, None] + studentt_logpdf(obs, 0.0, sigma_noise[:, None], AZ3.nu_obs)
            return np.logaddexp(lp_signal, lp_noise)

        def mixture_cdf(obs, z, sigma_obs, rho, sigma_noise):
            cdf_signal = studentt_cdf(obs, z, sigma_obs[:, None], AZ3.nu_obs)
            cdf_noise = studentt_cdf(obs, 0.0, sigma_noise[:, None], AZ3.nu_obs)
            return rho[:, None] * cdf_signal + (1 - rho[:, None]) * cdf_noise

        loglik_rows, test_idx_order = [], []
        for a in held_out:
            D_a = D_full[a]
            sigma_delta_a = AZ3.sigma_delta_floor + AZ3.k_sigma_delta * abs(D_a)
            z_new = zero_sum_z_new_draws(D_a, sigma_delta_a, self.n_years, n_draws, rng)

            lp_P = mixture_logpdf(P_full[a][None, :], z_new, sigma_plan, rho_P, sigma_noise_P)
            lp_E = mixture_logpdf(E_full[a][None, :], z_new, sigma_ben, rho_E, sigma_noise_E)
            loglik_rows.append(lp_P + lp_E)

            pit_P = mixture_cdf(P_full[a], z_new, sigma_plan, rho_P, sigma_noise_P).mean(axis=0)
            pit_E = mixture_cdf(E_full[a], z_new, sigma_ben, rho_E, sigma_noise_E).mean(axis=0)
            self._record_pit([a] * self.n_years, 'P', pit_P)
            self._record_pit([a] * self.n_years, 'E', pit_E)

            for t in range(self.n_years):
                test_idx_order.append(a * self.n_years + t)

        loglik = np.concatenate(loglik_rows, axis=1)
        order = np.argsort(test_idx_order)
        loglik = loglik[:, order]

        return xr.DataArray(loglik[None, :, :], dims=['chain', 'draw', 'test_idx'],
                            coords={'test_idx': np.sort(test_idx_order)})


# ── M5 ──────────────────────────────────────────────────────────────────────────

class M5KFoldWrapper(DwellingModelKFoldWrapper):
    """
    M5: Normal(mu_area, sigma_slab) z-prior + soft census constraint (exact
    Gaussian-conjugate predictive, see gaussian_conjugate_z_new_draws), fully-
    pooled lag convolution on P, then spatial misallocation smearing P_mean
    across ALL 200 areas' Queen-contiguity neighbours (not just the training
    subset) — held-out areas' neighbours that happen to be OTHER held-out
    areas in the same fold get their own freshly-drawn z_new too, so the
    smearing step always has a value for every one of a held-out area's
    geometric neighbours, whether trained or held out. BEN has no lag/spatial
    term in M5 (plain StudentT on raw z, same as AZ0a).
    """

    def __init__(self, *args, full_W, full_pre_inference, max_lag, n_lags, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_W = full_W
        self.full_pre_inference = full_pre_inference
        self.max_lag = max_lag
        self.n_lags = n_lags

    def log_likelihood__i(self, excluded_obs, idata__i):
        held_out = excluded_obs['held_out_areas']
        train_areas = excluded_obs['train_areas']
        D_full, P_full, E_full = (self.full_data['D'], self.full_data['P_obs'],
                                   self.full_data['E_obs'])

        post = idata__i.posterior
        sigma_slab = post['sigma_slab'].values.reshape(-1)
        sigma_plan = post['sigma_plan'].values.reshape(-1)
        sigma_ben = post['sigma_ben'].values.reshape(-1)
        alpha = post['alpha_spatial'].values.reshape(-1)
        lambda_weights = post['lambda_weights'].values.reshape(-1, self.n_lags)
        z_train = post['z'].values.reshape(-1, len(train_areas), self.n_years)
        n_draws = len(sigma_slab)
        rng = np.random.default_rng(self.rng_seed)

        sigma_census_full = np.maximum(np.abs(D_full) * M5.census_rel_error,
                                        M5.census_abs_floor)

        z_new = np.stack([
            gaussian_conjugate_z_new_draws(D_full[a], sigma_slab, sigma_census_full[a],
                                            self.n_years, rng)
            for a in held_out
        ], axis=1)  # (n_draws, n_held_out, n_years)

        z_full = np.zeros((n_draws, self.n_areas, self.n_years))
        z_full[:, train_areas, :] = z_train
        z_full[:, held_out, :] = z_new

        P_mean_temporal_full = lag_convolve_numpy(
            z_full, self.full_pre_inference, lambda_weights, self.max_lag, self.n_lags)

        # Spatial smearing, held-out rows only: (1-alpha)*own + alpha*W@all
        W_rows = self.full_W[held_out, :]  # (n_held_out, n_areas)
        spatial_term = np.einsum('ij,djt->dit', W_rows, P_mean_temporal_full)
        own_term = P_mean_temporal_full[:, held_out, :]
        P_mean_smeared = (1 - alpha)[:, None, None] * own_term + alpha[:, None, None] * spatial_term

        loglik_rows, test_idx_order = [], []
        for i, a in enumerate(held_out):
            lp_P = studentt_logpdf(P_full[a][None, :], P_mean_smeared[:, i, :],
                                    sigma_plan[:, None], M5.nu_obs)
            lp_E = studentt_logpdf(E_full[a][None, :], z_new[:, i, :],
                                    sigma_ben[:, None], M5.nu_obs)
            loglik_rows.append(lp_P + lp_E)

            pit_P = studentt_cdf(P_full[a], P_mean_smeared[:, i, :],
                                  sigma_plan[:, None], M5.nu_obs).mean(axis=0)
            pit_E = studentt_cdf(E_full[a], z_new[:, i, :],
                                  sigma_ben[:, None], M5.nu_obs).mean(axis=0)
            self._record_pit([a] * self.n_years, 'P', pit_P)
            self._record_pit([a] * self.n_years, 'E', pit_E)

            for t in range(self.n_years):
                test_idx_order.append(a * self.n_years + t)

        loglik = np.concatenate(loglik_rows, axis=1)
        order = np.argsort(test_idx_order)
        loglik = loglik[:, order]

        return xr.DataArray(loglik[None, :, :], dims=['chain', 'draw', 'test_idx'],
                            coords={'test_idx': np.sort(test_idx_order)})


# ── Driver ────────────────────────────────────────────────────────────────────────

def data_matching_trace(gdf, trace):
    """Subset/reorder gdf to the LSOA order embedded in the trace's 'area' coord —
    same convention as cli.py's _data_matching_traces, reimplemented here to avoid
    importing a private cli.py helper into a standalone script."""
    lsoa_codes = trace.posterior['z'].coords['area'].values.tolist()
    subset = gdf[gdf['LSOA21CD'].isin(lsoa_codes)].copy()
    subset = subset.set_index('LSOA21CD').loc[lsoa_codes].reset_index()
    return make_data_dict(subset)


def subset_data(data, n_subset):
    """
    Restrict the matching data dict to the first n_subset areas — used only
    for a fast smoke test of the K-fold pipeline's mechanics (fold splitting,
    wrapper plumbing, predictive math) on a small problem before committing
    to the full 200-area/K=10 run.
    """
    gdf_sub = data['gdf'].iloc[:n_subset].reset_index(drop=True)
    return make_data_dict(gdf_sub)


def build_shape_reference(data, var_name='P_like'):
    """
    A minimal DataTree containing only a correctly-shaped/coordinated
    log_likelihood[var_name] — all `loo_kfold`'s own bookkeeping
    (_prepare_kfold_inputs) reads from `data` is this group's obs_dims,
    n_data_points, and coords for fold splitting; the actual refitting and
    scoring is entirely delegated to the wrapper (sel_observations/sample/
    log_likelihood__i), so no real posterior or log-likelihood VALUES are
    needed here — only the shape. Building this directly (rather than
    reusing/subsetting a model's real saved trace) sidesteps a DataTree.isel
    quirk found while testing: `.isel(area=...)` silently failed to subset
    the log_likelihood group in this arviz version even though it correctly
    subsetted posterior, which would have made group_by's length check pass
    against inconsistent data silently.
    """
    n_areas, n_years = data['n_areas'], data['n_years']
    # A SINGLE flat 'cell' obs dim, not separate ('area', 'year') dims: the
    # installed arviz_stats' _compute_kfold_results assembles elpd_i/p_kfold_i
    # using only the LAST obs_dim's coord, sized to n_data_points — that
    # silently breaks (a coordinate-length mismatch) whenever obs_dims has
    # more than one dimension, as (area, year) would. A single combined dim
    # sidesteps it entirely and is all group_by/fold-splitting actually needs.
    dummy = np.zeros((1, 1, n_areas * n_years))
    coords = {'cell': np.arange(n_areas * n_years)}
    return az.from_dict({'log_likelihood': {var_name: dummy}},
                        dims={var_name: ['cell']}, coords=coords)


def make_shared_area_folds(n_areas, n_years, k, seed):
    """
    One fixed, seeded area->fold assignment, shared identically across every
    model. `azs.loo_kfold`'s own `group_by` path (arviz_stats.loo.helper_loo_kfold
    ._kfold_split_grouped) calls `np.random.default_rng()` with NO seed — calling
    it once per model would give each model a DIFFERENT random partition of
    areas, silently breaking the comparison (elpd differences would partly
    reflect "which areas each model happened to lose" rather than purely model
    quality) and making the whole run non-reproducible. Building the
    assignment here once and passing it as `folds=` (not `group_by=`) fixes
    both problems.
    """
    rng = np.random.default_rng(seed)
    area_fold = np.empty(n_areas, dtype=int)
    shuffled = rng.permutation(n_areas)
    for fold_id, chunk in enumerate(np.array_split(shuffled, k), start=1):
        area_fold[chunk] = fold_id
    return np.repeat(area_fold, n_years)  # per-cell (area, year) fold assignment


def run_kfold_for_model(model_name, model_class, wrapper_class, data,
                        n_years, k, sample_kwargs, output_dir, folds,
                        extra_wrapper_kwargs=None):
    wrapper = wrapper_class(model_class, data, n_years, sample_kwargs,
                            **(extra_wrapper_kwargs or {}))
    shape_reference = build_shape_reference(data)

    t0 = time.time()
    result = azs.loo_kfold(shape_reference, wrapper, var_name='P_like',
                          folds=folds, pointwise=True)
    elapsed = time.time() - t0
    print(f'  {model_name} k-fold done in {elapsed:.0f}s: '
          f'elpd={result.elpd:.1f}, se={result.se:.1f}')

    fold_dir = output_dir / model_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'model_name': model_name,
        'k': k,
        'elpd_kfold': float(result.elpd),
        'se_elpd_kfold': float(result.se),
        # result.p ("effective parameters") is NOT meaningful here: arviz_stats
        # derives it from `data`'s own log_likelihood values as a baseline,
        # but `data` here is build_shape_reference's placeholder (all zeros) —
        # only elpd/se, computed entirely from the wrapper's real refits, are
        # trustworthy outputs of this pipeline.
        'n_data_points': int(result.n_data_points),
        'n_samples_per_fold': int(result.n_samples),
        'sample_kwargs': sample_kwargs,
        'elapsed_seconds': elapsed,
    }
    (fold_dir / 'kfold_summary.json').write_text(json.dumps(summary, indent=2))

    elpd_i = result.elpd_i.to_dataframe(name='elpd_i').reset_index()
    elpd_i['area_idx'] = elpd_i['cell'] // n_years
    elpd_i['year_idx'] = elpd_i['cell'] % n_years
    elpd_i['area'] = np.asarray(data['gdf']['LSOA21CD'])[elpd_i['area_idx']]
    elpd_i.to_csv(fold_dir / 'elpd_i.csv', index=False)

    pit_df = pd.DataFrame(wrapper.pit_records)
    pit_df.to_csv(fold_dir / 'pit_records.csv', index=False)

    return summary, pit_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--smoke-n-areas', type=int, default=None,
                    help='subset the reference trace+data to this many areas '
                         'for a fast pipeline smoke test (default: use all 200)')
    ap.add_argument('--draws', type=int, default=600)
    ap.add_argument('--tune', type=int, default=500)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--data-path', default=str(DATA_PATH) if DATA_PATH else 'data')
    ap.add_argument('--traces-dir', default='results/traces')
    ap.add_argument('--output-dir', default='results/artifacts/kfold_comparison')
    ap.add_argument('--models', default='AZ0a,AZ3,M5')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fold-seed', type=int, default=42,
                    help='seeds the area->fold assignment. Deterministic given '
                         '(n_areas, k, fold-seed), so running each model as a '
                         'separate process (e.g. for parallelism) still yields '
                         'an IDENTICAL fold partition across models as long as '
                         'this stays fixed — required for a fair comparison.')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('-- Loading reference trace (AZ0a.nc) and matching gdf --')
    reference_trace = az.from_netcdf(str(Path(args.traces_dir) / 'AZ0a.nc'))
    gdf = load_data(args.data_path)
    gdf, _ = apply_outlier_exclusion(gdf, verbose=False)
    data = data_matching_trace(gdf, reference_trace)
    print(f'   {data["n_areas"]} areas matched')

    base_sample_kwargs = dict(draws=args.draws, tune=args.tune, chains=args.chains,
                              random_seed=args.seed)

    requested = [m for m in args.models.split(',')]
    all_summaries = []
    for name in requested:
        model_class = {'AZ0a': AZ0a, 'AZ3': AZ3, 'M5': M5}[name]
        wrapper_class = {'AZ0a': AZ0aKFoldWrapper, 'AZ3': AZ3KFoldWrapper,
                        'M5': M5KFoldWrapper}[name]
        extra_sample_kwargs = {'AZ0a': {'target_accept': 0.9}, 'AZ3': {'target_accept': 0.95},
                               'M5': {'target_accept': 0.9}}[name]

        print(f'\n-- {name}: building model-specific fixtures --')
        model_data = data if args.smoke_n_areas is None else subset_data(data, args.smoke_n_areas)

        extra_wrapper_kwargs = {}
        if name == 'M5':
            from housing_projections.models.models import _build_pre_inference
            max_lag = M5.max_lag
            extra_wrapper_kwargs = {
                'full_W': build_spatial_weights(model_data['gdf']),
                'full_pre_inference': _build_pre_inference(model_data, max_lag, source='P'),
                'max_lag': max_lag,
                'n_lags': max_lag + 1,
            }

        folds = make_shared_area_folds(model_data['n_areas'], model_data['n_years'],
                                       args.k, args.fold_seed)
        summary, _ = run_kfold_for_model(
            name, model_class, wrapper_class, model_data, model_data['n_years'], args.k,
            {**base_sample_kwargs, **extra_sample_kwargs}, output_dir, folds,
            extra_wrapper_kwargs=extra_wrapper_kwargs,
        )
        all_summaries.append(summary)

    comparison = pd.DataFrame(all_summaries).sort_values('elpd_kfold', ascending=False)
    comparison['elpd_diff'] = comparison['elpd_kfold'] - comparison['elpd_kfold'].iloc[0]
    comparison.to_csv(output_dir / 'comparison.csv', index=False)
    print('\n-- K-fold comparison --')
    print(comparison[['model_name', 'elpd_kfold', 'se_elpd_kfold', 'elpd_diff']].to_string(index=False))


if __name__ == '__main__':
    main()
