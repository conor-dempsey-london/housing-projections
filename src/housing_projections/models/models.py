import numpy as np
import pymc as pm
import pytensor.tensor as pt

from housing_projections.config import (
    ALL_COLS_PLAN,
    CENSUS_ABS_FLOOR,
    CENSUS_REL_ERROR,
    INFER_COLS_PLAN,
)
from housing_projections.spatial import build_spatial_weights

from .base import DwellingModel

__all__ = ["M0", "M0h", "M1", "M1h", "M2h", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

# ── Builder functions (private) ───────────────────────────────────────────────

def _build_z_prior(data, n_areas, n_years):
    """
    Build latent z prior with global mean and spread.
    Returns (mu_slab, sigma_slab, z).
    Must be called inside a pm.Model() context.
    """
    mu_slab    = pm.Normal('mu_slab',
                           mu=data['D_full_mean'] / n_years,
                           sigma=5)
    sigma_slab = pm.HalfNormal('sigma_slab', sigma=10)
    z          = pm.Normal('z',
                           mu=mu_slab,
                           sigma=sigma_slab,
                           dims=('area', 'year'))
    return mu_slab, sigma_slab, z


def _build_census_constraint(z, D, sigma_census):
    """
    Add census constraint likelihood.
    Must be called inside a pm.Model() context.
    """
    pm.Normal('census_obs', mu=z.sum(axis=1),
              sigma=sigma_census, observed=D)


def _build_pre_inference(data, max_lag):
    """
    Build fixed array of pre-inference planning observations
    to use as proxies for source years before the inference window.
    Returns numpy array of shape (n_areas, max_lag).
    """
    infer_start = ALL_COLS_PLAN.index(INFER_COLS_PLAN[0])
    P_obs_full  = data['P_obs_full']
    return np.stack([
        P_obs_full[:, infer_start - max_lag + k]
        if (infer_start - max_lag + k) >= 0
        else P_obs_full[:, 0]
        for k in range(max_lag)
    ], axis=1).astype('float64')


def _build_lag(z, pre_inference, n_areas, n_years, n_lags, alpha, max_lag,
              lambda_weights=None):
    """
    Build temporal lag structure for planning data.
    Returns (lambda_weights, P_mean).
    Must be called inside a pm.Model() context.
    """
    if lambda_weights is None:
        lambda_weights = pm.Dirichlet('lambda_weights', a=alpha)
    else:
        lambda_weights = pt.as_tensor_variable(
            lambda_weights.astype('float64'))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    P_mean = pt.sum(shifted * lambda_weights[None, None, :], axis=2)
    return lambda_weights, P_mean


def _build_planning_likelihood_simple(P_mean, P_obs, nu_obs, sigma_obs):
    """
    M1 planning likelihood — StudentT, no missingness.
    Must be called inside a pm.Model() context.
    """
    pm.StudentT('P_like', nu=nu_obs, mu=P_mean,
                sigma=sigma_obs, observed=P_obs)


def _build_symmetric_missingness():
    """
    M2 symmetric missingness — single global pi_miss.
    Prior mean 0.2, informed by observed zero rates.
    Returns pi_miss scalar.
    Must be called inside a pm.Model() context.
    """
    return pm.Beta('pi_miss', alpha=2, beta=8)


def _build_asymmetric_missingness(P_mean, sigma_obs):
    """
    Asymmetric missingness dependent on P_mean rather than z.
    When P_mean < 0 (demolition signal), missingness is high.
    When P_mean > 0 (completion signal), missingness is moderate.
    Returns pi_miss of shape (n_areas, n_years).
    """
    pi_miss_pos = pm.Beta('pi_miss_pos', alpha=7, beta=3)  # mean 0.70
    pi_miss_neg = pm.Beta('pi_miss_neg', alpha=8, beta=2)  # mean 0.80

    sqrt2       = pt.as_tensor_variable(np.float64(np.sqrt(2)))
    sigma_safe  = sigma_obs + 1e-6
    p_pos_local = 0.5 * pt.erfc(-P_mean / (sigma_safe * sqrt2))
    p_neg_local = 1.0 - p_pos_local

    return p_pos_local * pi_miss_pos + p_neg_local * pi_miss_neg


def _build_planning_likelihood_zeroinflated(P_mean, P_obs,
                                            pi_miss, nu_obs, sigma_obs):
    """
    Zero-inflated planning likelihood via pm.Potential.

    Implements the correct zero-inflation formula without a spike approximation:
      P=0:  log p = logaddexp(log(pi_miss), log(1-pi_miss) + logT(0; P_mean, σ))
      P>0:  log p = log(1-pi_miss) + logT(P_obs; P_mean, σ)

    The spike-Normal approach (sigma=1e-6) was wrong for HMC: it made the
    zero component dominate with a huge constant (~26.7), killing the gradient
    from P=0 cells and confusing NUTS step-size adaptation.
    """
    is_zero = (P_obs == 0)   # fixed boolean numpy mask

    log_p_student = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=P_mean, sigma=sigma_obs), P_obs
    )
    log_pi       = pt.log(pi_miss)
    log_1mpi     = pt.log(1 - pi_miss)

    # P=0: mixture of structural zero and StudentT evaluated at 0
    log_lik_zero    = pt.logaddexp(log_pi, log_1mpi + log_p_student)
    # P>0: only the non-zero component can generate the observation
    log_lik_nonzero = log_1mpi + log_p_student

    log_lik = pt.where(is_zero, log_lik_zero, log_lik_nonzero)
    pm.Potential('P_like', log_lik.sum())


def _build_planning_likelihood_zeroinflated_twocomp(P_mean, P_obs,
                                                    pi_miss, nu_obs,
                                                    sigma_obs_tight,
                                                    sigma_obs_loose,
                                                    w_tight):
    """
    Zero-inflated planning likelihood with two-component observation noise.
    Component 1: tight StudentT for typical precise observations.
    Component 2: loose StudentT for rare large-error observations
                 (temporal/spatial misallocation, batch recording etc.)
    """
    n_areas, n_years = P_obs.shape

    pi_miss_broadcast = pt.broadcast_to(pi_miss, (n_areas, n_years))

    w = pt.stack([
        pi_miss_broadcast,
        (1 - pi_miss_broadcast) * w_tight,
        (1 - pi_miss_broadcast) * (1 - w_tight),
    ], axis=-1)

    pm.Mixture('P_like',
               w=w,
               comp_dists=[
                   pm.Normal.dist(mu=0, sigma=1e-6,
                                  shape=(n_areas, n_years)),
                   pm.StudentT.dist(nu=nu_obs, mu=P_mean,
                                    sigma=sigma_obs_tight,
                                    shape=(n_areas, n_years)),
                   pm.StudentT.dist(nu=nu_obs, mu=P_mean,
                                    sigma=sigma_obs_loose,
                                    shape=(n_areas, n_years)),
               ],
               observed=P_obs)


def _build_spatial_misallocation(z, W, n_areas, n_years):
    """
    Apply spatial misallocation to P_mean.
    alpha: global misallocation probability — fraction of completions
           recorded in a neighbouring LSOA rather than the true one.
    W:     row-normalised queen contiguity matrix (n_areas, n_areas).
    Returns spatially smeared P_mean of shape (n_areas, n_years).
    Must be called inside a pm.Model() context.
    """
    alpha      = pm.Beta('alpha_spatial', alpha=1, beta=19)  # prior mean 0.05
    W_tensor   = pt.as_tensor_variable(W.astype('float64'))
    I_tensor   = pt.eye(n_areas)
    W_spatial  = (1 - alpha) * I_tensor + alpha * W_tensor
    return pt.dot(W_spatial, z)


# ── Models ────────────────────────────────────────────────────────────────────

class M0(DwellingModel):
    """
    Baseline: Normal prior on z, learned observation noise per source.

    sigma_plan and sigma_ben are inferred from the data rather than fixed,
    allowing the model to adapt to the actual noise level of each source.
    """

    name             = 'M0'
    description      = 'Baseline: Normal prior on z, learned observation noise'
    var_names        = ['mu_slab', 'sigma_slab', 'sigma_plan', 'sigma_ben']
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z    = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)
            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                             sigma_plan=sigma_plan,
                                             sigma_ben=sigma_ben)

        self.model = model
        return model


class M0h(DwellingModel):
    """
    Hierarchical extension of M0. Each LSOA mean is pinned to D[a]/n_years
    (the census-implied annual rate — empirically the mu_area posterior is
    completely determined by this). z deviates from that fixed mean via
    sigma_slab (non-centred). Observation noise learned per source.
    """

    name             = 'M0h'
    description      = 'M0 + area-level mean pinned to census, sigma_slab hierarchy'
    var_names        = ['sigma_slab', 'sigma_plan', 'sigma_ben']
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            # Area mean fixed at census-implied annual rate — the mu_area
            # posterior is empirically indistinguishable from D[a]/n_years.
            mu_area = D / n_years  # (n_areas,) numpy constant

            # ── Latent true changes (non-centered on sigma_slab) ─────────
            sigma_slab = pm.HalfNormal('sigma_slab', sigma=10)
            z_offset   = pm.Normal('z_offset', mu=0, sigma=1,
                                   dims=('area', 'year'))
            z          = pm.Deterministic('z',
                                          mu_area[:, None] + sigma_slab * z_offset,
                                          dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)
            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                             sigma_plan=sigma_plan,
                                             sigma_ben=sigma_ben)

        self.model = model
        return model


class M1(DwellingModel):
    """
    Adds temporal lag in planning data.
    A true change in year t may be recorded in planning in year t+k,
    with lag weights lambda_k ~ Dirichlet(alpha).
    BEN is assumed to have no lag.
    """

    name        = 'M1'
    description = 'M0 + temporal lag in planning completions'
    var_names   = ['mu_slab', 'sigma_slab', 'sigma_plan', 'sigma_ben', 'lambda_weights']
    max_lag     = 3

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan)
            self.add_ben_likelihood(z, data['E_obs'], sigma_ben=sigma_ben)

        self.model = model
        return model


class M1h(DwellingModel):
    """
    Combines the M0h area-level structure with the M1 temporal lag.

    Area means are pinned to D[a]/n_years (census-implied annual rate).
    z deviates from that fixed mean via sigma_slab (non-centred). The
    planning likelihood uses a lagged z mean (lambda_weights), giving
    sigma_plan a structural explanation for why P_obs != z.

    BEN is assumed lag-free.
    """

    name        = 'M1h'
    description = 'Census-pinned area means + M1 temporal lag in planning'
    var_names   = ['sigma_slab', 'sigma_plan', 'sigma_ben', 'lambda_weights']
    max_lag     = 3

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:

            mu_area = D / n_years  # (n_areas,) numpy constant

            sigma_slab = pm.HalfNormal('sigma_slab', sigma=10)
            z_offset   = pm.Normal('z_offset', mu=0, sigma=1,
                                   dims=('area', 'year'))
            z          = pm.Deterministic('z',
                                          mu_area[:, None] + sigma_slab * z_offset,
                                          dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)

            # ── Planning lag (from M1) ────────────────────────────────────
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                              self.nu_obs, sigma_plan)
            self.add_ben_likelihood(z, data['E_obs'], sigma_ben=sigma_ben)

        self.model = model
        return model


class M2h(DwellingModel):
    """
    Extends M1h with per-area zero-inflation in the planning likelihood.

    pi_miss[a] is fixed at the empirical P=0 rate conditioned on E>0,
    computed from the data rather than inferred. Inferring pi_miss per area
    creates a bimodal posterior (reporter vs non-reporter) for each of ~5k
    areas, causing NUTS non-convergence. With pi_miss fixed, the model is
    identified and tractable.

    sigma_plan is fixed at 5. Recording-rate variation (alpha[a]) is
    deferred to M3h.

    BEN is assumed lag-free and always present.
    """

    name        = 'M2h'
    description = 'M1h + per-area zero-inflation, empirical pi_miss from data'
    var_names   = ['sigma_slab', 'lambda_weights']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:

            mu_area = D / n_years  # (n_areas,) numpy constant

            sigma_slab = pm.HalfNormal('sigma_slab', sigma=10)
            z_offset   = pm.Normal('z_offset', mu=0, sigma=1,
                                   dims=('area', 'year'))
            z          = pm.Deterministic('z',
                                          mu_area[:, None] + sigma_slab * z_offset,
                                          dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)

            # ── Temporal lag (M1) ─────────────────────────────────────────
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            # ── Per-area zero-inflation (fixed from data) ─────────────────
            # pi_miss[a] = empirical P=0 rate conditioned on E>0. Inferring
            # pi_miss per area creates a bimodal posterior (reporter vs
            # non-reporter) for each of ~5k areas, making NUTS non-convergent.
            # Recording-rate modelling (alpha[a]) deferred to M3h.
            pi_miss = data['pi_miss_empirical']   # (n_areas,) numpy constant

            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'],
                pi_miss[:, None],   # broadcasts over years
                self.nu_obs, 5.0,
            )

            # ── BEN: direct, fixed noise ──────────────────────────────────
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M2(DwellingModel):
    """
    Adds symmetric zero-inflation to planning likelihood.
    A planning observation of zero may reflect missing data
    rather than true zero change.
    """

    name        = 'M2'
    description = 'M1 + symmetric zero-inflated planning observations'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights', 'pi_miss']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_symmetric_missingness()
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M3(DwellingModel):
    """
    Replaces symmetric missingness with asymmetric —
    separate missingness rates for completions (z>0) and demolitions (z<0).
    """

    name        = 'M3'
    description = 'M2 + asymmetric missingness in planning'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights',
                   'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M4(DwellingModel):
    """
    M3 with two-component observation noise for planning data.
    Separates typical precise observations (tight StudentT) from
    rare large-error observations due to temporal/spatial misallocation
    (loose StudentT).
    """

    name             = 'M4'
    description      = 'M3 + two-component observation noise in planning'
    var_names        = ['mu_slab', 'sigma_slab', 'lambda_weights',
                        'pi_miss_pos', 'pi_miss_neg', 'w_tight']
    max_lag          = 3
    snap_zeros       = True
    sigma_obs_tight  = 0.5
    sigma_obs_loose  = 20.0

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            w_tight = pm.Beta('w_tight', alpha=9, beta=1)  # prior mean 0.9
            _build_planning_likelihood_zeroinflated_twocomp(
                P_mean, data['P_obs'], pi_miss,
                self.nu_obs,
                self.sigma_obs_tight,
                self.sigma_obs_loose,
                w_tight)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M5(DwellingModel):
    """
    Adds spatial misallocation in planning data to M3.
    A fraction alpha of planning completions are recorded in a
    neighbouring LSOA rather than the true one, modelled via a
    row-stochastic spatial weights matrix derived from queen contiguity.
    BEN is assumed to have no spatial misallocation.

    Set lambda_weights_fixed to a numpy array to fix the lag weights
    rather than sampling them.
    """

    name                 = 'M5'
    description          = 'M3 + spatial misallocation in planning'
    max_lag              = 3
    snap_zeros           = True
    lambda_weights_fixed = None

    @property
    def var_names(self):
        names = ['mu_slab', 'sigma_slab', 'pi_miss_pos',
                 'pi_miss_neg', 'alpha_spatial']
        if self.lambda_weights_fixed is None:
            names.insert(2, 'lambda_weights')
        return names

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)
        W             = build_spatial_weights(data['gdf'])  # M5-specific

        with pm.Model(coords=self._default_coords()) as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)

            _, P_mean_temporal = _build_lag(
                z, pre_inference, n_areas, n_years,
                self.n_lags, self.lag_alpha, self.max_lag,
                lambda_weights=self.lambda_weights_fixed)

            P_mean  = _build_spatial_misallocation(
                P_mean_temporal, W, n_areas, n_years)
            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


# ── M6: Temporal AR(1) prior on z ────────────────────────────────────────────

class M6(DwellingModel):
    """
    Replaces the i.i.d. year prior on z with an AR(1) process per area.

    z[a, 0]   ~ Normal(rho * z_prev[a] + (1-rho) * mu_slab, sigma_innov)
    z[a, t]   ~ Normal(rho * z[a, t-1] + (1-rho) * mu_slab, sigma_innov)

    where rho ~ Beta(8, 2) (prior mean 0.8) and z_prev[a] is the observed
    planning completion in the year immediately before the inference window
    (a fixed constant, not a latent variable).  Using real pre-window data
    as the warm-start eliminates the boundary effect that would arise from
    initialising cold from the global prior, and removes the need for a
    separate sigma_init parameter.

    BEN is assumed lag-free.  Planning uses the same zero-inflated asymmetric
    likelihood as M3 (no spatial misallocation).
    """

    name        = 'M6'
    description = 'M3 + AR(1) temporal prior on z with pre-window warm-start'
    var_names   = ['mu_slab', 'sigma_innov', 'rho',
                   'lambda_weights', 'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        # Fixed planning observation immediately before the inference window —
        # used as a known z_prev to warm-start the AR(1) and avoid boundary effects.
        z_prev_obs = pt.as_tensor_variable(
            pre_inference[:, -1].astype('float64'))

        with pm.Model(coords=self._default_coords()) as model:

            # ── Global prior ──────────────────────────────────────────────
            mu_slab     = pm.Normal('mu_slab',
                                    mu=data['D_full_mean'] / n_years,
                                    sigma=5)
            sigma_innov = pm.HalfNormal('sigma_innov', sigma=10)
            rho         = pm.Beta('rho', alpha=8, beta=2)  # prior mean 0.8

            # ── AR(1) scan over years ─────────────────────────────────────
            # Non-centered parameterisation: z_raw ~ Normal(0,1),
            # z_t = rho * z_{t-1} + (1-rho) * mu_slab + sigma_innov * z_raw_t
            # t=0 is warm-started from z_prev_obs (fixed pre-window data).
            z_init_raw = pm.Normal('z_init_raw', mu=0, sigma=1,
                                   shape=(n_areas,))
            z_init = rho * z_prev_obs + (1 - rho) * mu_slab + sigma_innov * z_init_raw

            z_list = [z_init]
            for t in range(1, n_years):
                z_prev  = z_list[t - 1]
                z_t_raw = pm.Normal(f'z_raw_{t}', mu=0, sigma=1,
                                    shape=(n_areas,))
                z_t = rho * z_prev + (1 - rho) * mu_slab + sigma_innov * z_t_raw
                z_list.append(z_t)

            z = pm.Deterministic('z', pt.stack(z_list, axis=1), dims=('area', 'year'))

            # ── Census constraint ─────────────────────────────────────────
            _build_census_constraint(z, D, sigma_census)

            # ── Planning lag + zero-inflation ─────────────────────────────
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


# ── M7: Borough-level hierarchy ───────────────────────────────────────────────

class M7(DwellingModel):
    """
    Adds a borough-level hierarchical prior on the mean annual change.

    mu_global           ~ Normal(D_full_mean / n_years, sigma=5)
    sigma_borough       ~ HalfNormal(sigma=5)
    mu_borough[b]       ~ Normal(mu_global, sigma_borough)   # per borough
    sigma_slab          ~ HalfNormal(sigma=10)
    z[a, t]             ~ Normal(mu_borough[borough[a]], sigma_slab)

    ``borough_idx`` must be provided in the data dict — a (n_areas,) integer
    array mapping each LSOA to its borough (0-indexed), and ``n_boroughs`` the
    total count. Both can be derived from a LSOA-to-LAD crosswalk joined on gdf.

    If ``borough_idx`` is absent from the data dict, raises ValueError.
    """

    name        = 'M7'
    description = 'M3 + borough-level hierarchical prior on mean annual change'
    var_names   = ['mu_global', 'sigma_borough', 'sigma_slab',
                   'lambda_weights', 'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        if 'borough_idx' not in data:
            raise ValueError(
                "M7 requires 'borough_idx' (int array, shape n_areas) and "
                "'n_boroughs' (int) in the data dict.  "
                "Derive them from a LSOA-to-LAD crosswalk joined on gdf."
            )
        n_boroughs    = data['n_boroughs']
        borough_idx   = data['borough_idx']
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:

            # ── Global hyperprior ─────────────────────────────────────────
            mu_global     = pm.Normal('mu_global',
                                       mu=data['D_full_mean'] / n_years,
                                       sigma=5)
            sigma_borough = pm.HalfNormal('sigma_borough', sigma=5)

            # ── Borough-level means (non-centered) ────────────────────────
            mu_borough_offset = pm.Normal('mu_borough_offset',
                                           mu=0, sigma=1, shape=n_boroughs)
            mu_borough = pm.Deterministic(
                'mu_borough', mu_global + sigma_borough * mu_borough_offset)

            # ── LSOA-level latent z ───────────────────────────────────────
            sigma_slab = pm.HalfNormal('sigma_slab', sigma=10)
            z = pm.Normal('z',
                          mu=mu_borough[borough_idx, None],
                          sigma=sigma_slab,
                          dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)

            # ── Planning lag + zero-inflation ─────────────────────────────
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)

            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)

            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


# ── M8: Time-varying observation noise ───────────────────────────────────────

class M8(DwellingModel):
    """
    Replaces the fixed planning observation noise (sigma_obs) with a
    year-specific noise level drawn from a shared hierarchical prior.

    sigma_base_plan    ~ HalfNormal(sigma=5)
    sigma_year_offset  ~ HalfNormal(sigma=2, shape=n_years)
    sigma_obs_plan[t]  = sigma_base_plan + sigma_year_offset[t]

    This captures the hypothesis that planning data quality varies over the
    intercensal window — e.g. system changes in 2013-2016, COVID in 2020-21.
    Inspecting the posterior of sigma_obs_plan by year is a diagnostic in its
    own right.  BEN noise remains fixed.
    """

    name        = 'M8'
    description = 'M3 + time-varying planning observation noise'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights',
                   'pi_miss_pos', 'pi_miss_neg',
                   'sigma_base_plan', 'sigma_year_offset']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:

            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, D, sigma_census)

            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            # ── Time-varying planning noise ───────────────────────────────
            sigma_base_plan   = pm.HalfNormal('sigma_base_plan',   sigma=5)
            sigma_year_offset = pm.HalfNormal('sigma_year_offset',
                                               sigma=2, shape=n_years)
            sigma_obs_plan    = pm.Deterministic(
                'sigma_obs_plan', sigma_base_plan + sigma_year_offset)

            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            shape   = data['P_obs'].shape
            w = pt.stack([
                pt.ones(shape) * pi_miss,
                pt.ones(shape) * (1 - pi_miss),
            ], axis=-1)
            pm.Mixture(
                'P_like',
                w=w,
                comp_dists=[
                    pm.Normal.dist(mu=0, sigma=1e-6, shape=shape),
                    pm.StudentT.dist(nu=self.nu_obs, mu=P_mean,
                                     sigma=sigma_obs_plan[None, :],
                                     shape=shape),
                ],
                observed=data['P_obs'],
            )
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model
