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

__all__ = ["M0", "M0h", "M1", "M2", "M3", "M4", "M5", "M5b", "M6", "M7", "M8", "M9"]

# ── Builder functions (private) ───────────────────────────────────────────────

def _build_z_prior(data, n_areas, n_years):
    """
    Build latent z prior with global mean and spread.
    Returns (mu_slab, sigma_slab, z).
    Must be called inside a pm.Model() context.
    """
    mu_slab    = pm.Normal('mu_slab',
                           mu=data['D_full_mean'] / n_years / 0.55,
                           sigma=5)
    sigma_slab = pm.HalfNormal('sigma_slab', sigma=30)
    z          = pm.Normal('z',
                           mu=mu_slab,
                           sigma=sigma_slab,
                           shape=(n_areas, n_years))
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
    M3 planning likelihood — StudentT, no missingness.
    Must be called inside a pm.Model() context.
    """
    pm.StudentT('P_like', nu=nu_obs, mu=P_mean,
                sigma=sigma_obs, observed=P_obs)


def _build_symmetric_missingness():
    """
    M4 symmetric missingness — single global pi_miss.
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
    Zero-inflated planning likelihood using pm.Mixture.
    pi_miss can be scalar (M4) or (n_areas, n_years) tensor (M5).
    """
    shape = P_obs.shape

    w = pt.stack([
        pt.ones(shape) * pi_miss,
        pt.ones(shape) * (1 - pi_miss),
    ], axis=-1)

    pm.Mixture('P_like',
               w=w,
               comp_dists=[
                   pm.Normal.dist(mu=0, sigma=1e-6, shape=shape),
                   pm.StudentT.dist(nu=nu_obs, mu=P_mean,
                                    sigma=sigma_obs, shape=shape),
               ],
               observed=P_obs)


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
    """Baseline: Normal prior on z, fixed observation noise."""

    name             = 'M0'
    description      = 'Baseline: Normal prior on z'
    var_names        = ['mu_slab', 'sigma_slab']
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    def build(self):
        data         = self.data
        n_areas      = data['n_areas']
        n_years      = data['n_years']
        sigma_census = self.make_sigma_census(data['D'])

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'])

        self.model = model
        return model


class M0h(DwellingModel):
    """
    Hierarchical extension of M0. Each LSOA has its own mean annual
    change drawn from a global distribution, using non-centered
    parameterisation for better mixing.
    """

    name             = 'M0h'
    description      = 'M0 + hierarchical area-level mean annual change'
    var_names        = ['mu_global', 'sigma_mu', 'sigma_slab']
    census_rel_error = CENSUS_REL_ERROR
    census_abs_floor = CENSUS_ABS_FLOOR

    def build(self):
        data         = self.data
        D            = data['D']
        n_areas      = data['n_areas']
        n_years      = data['n_years']
        sigma_census = self.make_sigma_census(D)

        with pm.Model() as model:

            # ── Global hyperprior ─────────────────────────────────────────
            mu_global = pm.Normal('mu_global',
                                   mu=data['D_full_mean'] / n_years,
                                   sigma=5)
            sigma_mu  = pm.HalfNormal('sigma_mu', sigma=10)

            # ── Area-level means (non-centered) ───────────────────────────
            mu_area_offset = pm.Normal('mu_area_offset',
                                       mu=0, sigma=1,
                                       shape=n_areas)
            pm.Deterministic('mu_area',
                             mu_global + sigma_mu * mu_area_offset)
            mu_area = mu_global + sigma_mu * mu_area_offset

            # ── Latent true changes ───────────────────────────────────────
            sigma_slab = pm.HalfNormal('sigma_slab', sigma=30)
            z = pm.Normal('z',
                          mu=mu_area[:, None],
                          sigma=sigma_slab,
                          shape=(n_areas, n_years))

            _build_census_constraint(z, D, sigma_census)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'])

        self.model = model
        return model


class M1(DwellingModel):
    """
    Adds spike-and-slab prior on z to capture sparsity of dwelling changes.
    Most LSOA-years see little activity (spike near zero); active years
    follow a heavy-tailed StudentT distribution (slab).
    """

    name      = 'M1'
    description = 'M0 + spike-and-slab prior on z'
    var_names = ['pi', 'sigma_slab', 'nu', 'sigma_obs']

    def build(self):
        data             = self.data
        D                = data['D']
        n_areas          = data['n_areas']
        n_years          = data['n_years']
        sigma_census     = self.make_sigma_census(D)
        empirical_obs_sd = float(np.abs(data['P_obs'] - data['E_obs']).mean())

        with pm.Model() as model:

            pi         = pm.Beta('pi',        alpha=4.5, beta=5.5)
            mu_slab    = pm.TruncatedNormal('mu_slab',
                             mu=data['D_full_mean'] / n_years / 0.55,
                             sigma=5, lower=0)
            sigma_slab = pm.HalfNormal('sigma_slab', sigma=30)
            nu         = pm.Gamma('nu',       alpha=2,   beta=0.1)
            sigma_obs  = pm.HalfNormal('sigma_obs', sigma=empirical_obs_sd)

            w = pt.stack([
                pt.ones((n_areas, n_years)) * pi,
                pt.ones((n_areas, n_years)) * (1 - pi),
            ], axis=-1)

            z = pm.Mixture('z',
                           w=w,
                           comp_dists=[
                               pm.Normal.dist(mu=0, sigma=0.3),
                               pm.StudentT.dist(nu=nu, mu=mu_slab,
                                                sigma=sigma_slab),
                           ],
                           shape=(n_areas, n_years))

            _build_census_constraint(z, D, sigma_census)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                             sigma_plan=sigma_obs,
                                             sigma_ben=sigma_obs)

        self.model = model
        return model


class M2(DwellingModel):
    """M0 with separate fixed observation noise per source."""

    name           = 'M2'
    description    = 'M0 + separate fixed observation noise per source'
    var_names      = ['mu_slab', 'sigma_slab']
    sigma_obs_plan = 2.0
    sigma_obs_ben  = 2.0

    def build(self):
        data         = self.data
        sigma_census = self.make_sigma_census(data['D'])

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, data['n_areas'], data['n_years'])
            _build_census_constraint(z, data['D'], sigma_census)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                             sigma_plan=self.sigma_obs_plan,
                                             sigma_ben=self.sigma_obs_ben)

        self.model = model
        return model


class M3(DwellingModel):
    """
    Adds temporal lag in planning data.
    A true change in year t may be recorded in planning in year t+k,
    with lag weights lambda_k ~ Dirichlet(alpha).
    BEN is assumed to have no lag.
    """

    name        = 'M3'
    description = 'M0 + temporal lag in planning completions'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights']
    max_lag     = 3

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        sigma_census  = self.make_sigma_census(data['D'])
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M4(DwellingModel):
    """
    Adds symmetric zero-inflation to planning likelihood.
    A planning observation of zero may reflect missing data
    rather than true zero change.
    """

    name        = 'M4'
    description = 'M3 + symmetric zero-inflated planning observations'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights', 'pi_miss']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        sigma_census  = self.make_sigma_census(data['D'])
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_symmetric_missingness()
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M5(DwellingModel):
    """
    Replaces symmetric missingness with asymmetric —
    separate missingness rates for completions (z>0) and demolitions (z<0).
    """

    name        = 'M5'
    description = 'M4 + asymmetric missingness in planning'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights',
                   'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        sigma_census  = self.make_sigma_census(data['D'])
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag)
            pi_miss = _build_asymmetric_missingness(P_mean, self.sigma_obs)
            _build_planning_likelihood_zeroinflated(
                P_mean, data['P_obs'], pi_miss, self.nu_obs, self.sigma_obs)
            self.add_ben_likelihood(z, data['E_obs'])

        self.model = model
        return model


class M5b(DwellingModel):
    """
    M5 with two-component observation noise for planning data.
    Separates typical precise observations (tight StudentT) from
    rare large-error observations due to temporal/spatial misallocation
    (loose StudentT).
    """

    name             = 'M5b'
    description      = 'M5 + two-component observation noise in planning'
    var_names        = ['mu_slab', 'sigma_slab', 'lambda_weights',
                        'pi_miss_pos', 'pi_miss_neg', 'w_tight']
    max_lag          = 3
    snap_zeros       = True
    sigma_obs_tight  = 0.5
    sigma_obs_loose  = 20.0

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        sigma_census  = self.make_sigma_census(data['D'])
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)
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


class M6(DwellingModel):
    """
    Adds spatial misallocation in planning data to M5.
    A fraction alpha of planning completions are recorded in a
    neighbouring LSOA rather than the true one, modelled via a
    row-stochastic spatial weights matrix derived from queen contiguity.
    BEN is assumed to have no spatial misallocation.

    Set lambda_weights_fixed to a numpy array to fix the lag weights
    rather than sampling them.
    """

    name                 = 'M6'
    description          = 'M5 + spatial misallocation in planning'
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
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        sigma_census  = self.make_sigma_census(data['D'])
        pre_inference = _build_pre_inference(data, self.max_lag)
        W             = build_spatial_weights(data['gdf'])

        with pm.Model() as model:
            _, _, z = _build_z_prior(data, n_areas, n_years)
            _build_census_constraint(z, data['D'], sigma_census)

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


# ── M7: Temporal AR(1) prior on z ────────────────────────────────────────────

class M7(DwellingModel):
    """
    Replaces the i.i.d. year prior on z with an AR(1) process per area.

    z[a, 0]   ~ Normal(mu_slab, sigma_init)
    z[a, t]   ~ Normal(rho * z[a, t-1] + (1-rho) * mu_slab, sigma_innov)

    where rho ~ Beta(8, 2) (prior mean 0.8).  This captures the empirical
    observation that dwelling change is autocorrelated year-on-year — an
    active development site remains active for several years.

    BEN is assumed lag-free.  Planning uses the same zero-inflated asymmetric
    likelihood as M5 (no spatial misallocation).
    """

    name        = 'M7'
    description = 'M5 + AR(1) temporal prior on z (replaces i.i.d. year prior)'
    var_names   = ['mu_slab', 'sigma_init', 'sigma_innov', 'rho',
                   'lambda_weights', 'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        D             = data['D']
        sigma_census  = self.make_sigma_census(D)
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:

            # ── Global prior ──────────────────────────────────────────────
            mu_slab     = pm.Normal('mu_slab',
                                    mu=data['D_full_mean'] / n_years / 0.55,
                                    sigma=5)
            sigma_init  = pm.HalfNormal('sigma_init',  sigma=30)
            sigma_innov = pm.HalfNormal('sigma_innov', sigma=15)
            rho         = pm.Beta('rho', alpha=8, beta=2)  # prior mean 0.8

            # ── AR(1) scan over years ─────────────────────────────────────
            # Non-centered: z_raw ~ Normal(0, 1), z = mu + sigma * z_raw
            z_init_raw = pm.Normal('z_init_raw', mu=0, sigma=1,
                                   shape=(n_areas,))
            z_init = mu_slab + sigma_init * z_init_raw

            z_list = [z_init]
            for t in range(1, n_years):
                z_prev  = z_list[t - 1]
                z_t_raw = pm.Normal(f'z_raw_{t}', mu=0, sigma=1,
                                    shape=(n_areas,))
                z_t = rho * z_prev + (1 - rho) * mu_slab + sigma_innov * z_t_raw
                z_list.append(z_t)

            z = pm.Deterministic('z', pt.stack(z_list, axis=1))

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


# ── M8: Borough-level hierarchy ───────────────────────────────────────────────

class M8(DwellingModel):
    """
    Adds a borough-level hierarchical prior on the mean annual change.

    mu_global           ~ Normal(D_full_mean / n_years, sigma=5)
    sigma_borough       ~ HalfNormal(sigma=5)
    mu_borough[b]       ~ Normal(mu_global, sigma_borough)   # per borough
    sigma_slab          ~ HalfNormal(sigma=20)
    z[a, t]             ~ Normal(mu_borough[borough[a]], sigma_slab)

    ``borough_idx`` must be provided in the data dict — a (n_areas,) integer
    array mapping each LSOA to its borough (0-indexed), and ``n_boroughs`` the
    total count. Both can be derived from a LSOA-to-LAD crosswalk joined on gdf.

    If ``borough_idx`` is absent from the data dict, raises ValueError.
    """

    name        = 'M8'
    description = 'M5 + borough-level hierarchical prior on mean annual change'
    var_names   = ['mu_global', 'sigma_borough', 'sigma_slab',
                   'lambda_weights', 'pi_miss_pos', 'pi_miss_neg']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data = self.data
        if 'borough_idx' not in data:
            raise ValueError(
                "M8 requires 'borough_idx' (int array, shape n_areas) and "
                "'n_boroughs' (int) in the data dict.  "
                "Derive them from a LSOA-to-LAD crosswalk joined on gdf."
            )
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        n_boroughs    = data['n_boroughs']
        borough_idx   = data['borough_idx']
        D             = data['D']
        sigma_census  = self.make_sigma_census(D)
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:

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
            sigma_slab = pm.HalfNormal('sigma_slab', sigma=20)
            z = pm.Normal('z',
                          mu=mu_borough[borough_idx, None],
                          sigma=sigma_slab,
                          shape=(n_areas, n_years))

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


# ── M9: Time-varying observation noise ───────────────────────────────────────

class M9(DwellingModel):
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

    name        = 'M9'
    description = 'M5 + time-varying planning observation noise'
    var_names   = ['mu_slab', 'sigma_slab', 'lambda_weights',
                   'pi_miss_pos', 'pi_miss_neg',
                   'sigma_base_plan', 'sigma_year_offset']
    max_lag     = 3
    snap_zeros  = True

    def build(self):
        data          = self.data
        n_areas       = data['n_areas']
        n_years       = data['n_years']
        D             = data['D']
        sigma_census  = self.make_sigma_census(D)
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model() as model:

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
