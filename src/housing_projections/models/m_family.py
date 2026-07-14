"""
M-family model classes (M0-M16).

The original M0->M8 progression plus later M9-M16 variants — see the module docstring in
CLAUDE.md for the full per-model narrative. Split out of the single `models.py` file (which
had grown to ~5,000 lines) purely for navigability; no behavioural change. See `az_family.py`
for the separate AZ0-AZ4b progression and `builders.py` for the shared construction pieces
both families use.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from housing_projections.config import (
    CENSUS_ABS_FLOOR,
    CENSUS_REL_ERROR,
    DEFAULT_SAMPLE_KWARGS,
    INFER_YEARS,
)
from housing_projections.spatial import build_spatial_weights

from .base import DwellingModel
from .builders import (
    _build_agreement_gated_likelihood,
    _build_asymmetric_missingness,
    _build_capture_rate,
    _build_census_constraint,
    _build_independent_agreement_gated_likelihood,
    _build_lag,
    _build_planning_likelihood_simple,
    _build_planning_likelihood_zeroinflated,
    _build_pre_inference,
    _build_spatial_misallocation,
    _build_temporal_reallocation_likelihood,
    _build_temporal_reallocation_likelihood_marginalizable,
    _build_z_prior,
    _build_z_prior_hierarchical,
    _build_z_prior_hierarchical_borough,
    _build_z_prior_profile_library,
    _build_z_prior_profile_library_horseshoe,
    _build_zero_sum_profile_library,
)

__all__ = [
    "M0", "M0h", "M1", "M1h", "M5", "M6", "M7", "M8",
    "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16",
]

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
    sigma_slab. Observation noise learned per source.

    z uses a CENTRED parameterisation (mu=mu_area, sigma=sigma_slab), not
    non-centred. A non-centred z = mu_area + sigma_slab * z_offset lets
    chains collapse into sigma_slab -> 0 (a funnel: the likelihood gradient
    w.r.t. z_offset vanishes there) and get trapped — see M2h's docstring,
    where this was diagnosed and fixed. That fix applies here identically:
    mu_area is a fixed constant, so centring carries no correlation-ridge
    risk (the ridge that originally motivated non-centring required mu_area
    to be a free parameter, which it no longer is).
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

            # ── Latent true changes (centred — see class docstring) ───────
            sigma_slab = pm.Gamma('sigma_slab', alpha=2, beta=1 / 12)
            z          = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab,
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
    z deviates from that fixed mean via sigma_slab (centred — see M0h/M2h
    docstrings for why). The planning likelihood uses a lagged z mean
    (lambda_weights), giving sigma_plan a structural explanation for why
    P_obs != z.

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

            sigma_slab = pm.Gamma('sigma_slab', alpha=2, beta=1 / 12)
            z          = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab,
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


class M5(DwellingModel):
    """
    Adds spatial misallocation to M1h's planning likelihood.
    A fraction alpha of planning completions are recorded in a
    neighbouring LSOA rather than the true one, modelled via a
    row-stochastic spatial weights matrix derived from queen contiguity.
    BEN is assumed to have no spatial misallocation.

    Rebuilt on M1h's plain StudentT planning likelihood (fixed mu_area,
    centred z) rather than an earlier zero-inflated version. The
    zero-inflated planning likelihood, combined with the tight per-area
    census-sum constraint, produced a weakly-identified "which year
    absorbed the change" ambiguity across the whole zero-inflated model
    family (formerly M2h/M2/M3/M4/M5/M6/M8) that got much worse at full
    ~5,000-area scale (min_ess ~5, <4% posterior predictive coverage) —
    see git history. M1h converges cleanly at full scale (min_ess in the
    thousands), so the remaining structural hypotheses — spatial
    misallocation here, AR(1) in M6, time-varying noise in M8 — are now
    built on that foundation instead. M2/M3/M4 (built entirely around
    zero-inflation/missingness) and the zero-inflated version of M2h were
    retired.

    Set lambda_weights_fixed to a numpy array to fix the lag weights
    rather than sampling them.
    """

    name                 = 'M5'
    description          = 'M1h + spatial misallocation in planning'
    max_lag              = 3
    lambda_weights_fixed = None

    @property
    def var_names(self):
        names = ['sigma_slab', 'sigma_plan', 'sigma_ben', 'alpha_spatial']
        if self.lambda_weights_fixed is None:
            names.insert(1, 'lambda_weights')
        return names

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)
        W             = build_spatial_weights(data['gdf'])  # M5-specific

        with pm.Model(coords=self._default_coords()) as model:

            mu_area = D / n_years  # (n_areas,) numpy constant

            sigma_slab = pm.Gamma('sigma_slab', alpha=2, beta=1 / 12)
            z          = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab,
                                   dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)

            _, P_mean_temporal = _build_lag(
                z, pre_inference, n_areas, n_years,
                self.n_lags, self.lag_alpha, self.max_lag,
                lambda_weights=self.lambda_weights_fixed)

            P_mean = _build_spatial_misallocation(
                P_mean_temporal, W, n_areas, n_years)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                              self.nu_obs, sigma_plan)
            self.add_ben_likelihood(z, data['E_obs'], sigma_ben=sigma_ben)

        self.model = model
        return model


# ── M6: Temporal AR(1) prior on z ────────────────────────────────────────────

class M6(DwellingModel):
    """
    Replaces M1h's i.i.d.-per-year z prior with an AR(1) process per area,
    mean-reverting to the fixed mu_area (D[a]/n_years).

    z[a, 0]   ~ Normal(rho * z_prev_obs[a] + (1-rho) * mu_area[a], sigma_innov)
    z[a, t]   ~ Normal(rho * z[a, t-1]     + (1-rho) * mu_area[a], sigma_innov)

    where rho ~ Beta(8, 2) (prior mean 0.8) and z_prev_obs[a] is the observed
    planning completion in the year immediately before the inference window
    (a fixed constant, not a latent variable).  Using real pre-window data
    as the warm-start eliminates the boundary effect that would arise from
    initialising cold from the prior.

    Rebuilt on M1h's plain StudentT planning likelihood — see M5's
    docstring for why the zero-inflated version was dropped.

    BEN is assumed lag-free.
    """

    name        = 'M6'
    description = 'M1h + AR(1) temporal prior with pre-window warm-start'
    var_names   = ['sigma_innov', 'rho', 'sigma_plan', 'sigma_ben', 'lambda_weights']
    max_lag     = 3

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        # Fixed planning observation immediately before the inference window —
        # used as a known z_prev to warm-start the AR(1) and avoid boundary effects.
        z_prev_obs = pt.as_tensor_variable(
            pre_inference[:, -1].astype('float64'))

        with pm.Model(coords=self._default_coords()) as model:

            mu_area = D / n_years  # (n_areas,) numpy constant, AR(1) target

            sigma_innov = pm.Gamma('sigma_innov', alpha=2, beta=1 / 10)
            rho         = pm.Beta('rho', alpha=8, beta=2)  # prior mean 0.8

            # ── AR(1) scan over years ─────────────────────────────────────
            # Non-centered parameterisation: z_raw ~ Normal(0,1),
            # z_t = rho * z_{t-1} + (1-rho) * mu_area + sigma_innov * z_raw_t
            # t=0 is warm-started from z_prev_obs (fixed pre-window data).
            z_init_raw = pm.Normal('z_init_raw', mu=0, sigma=1,
                                   shape=(n_areas,))
            z_init = rho * z_prev_obs + (1 - rho) * mu_area + sigma_innov * z_init_raw

            z_list = [z_init]
            for t in range(1, n_years):
                z_prev  = z_list[t - 1]
                z_t_raw = pm.Normal(f'z_raw_{t}', mu=0, sigma=1,
                                    shape=(n_areas,))
                z_t = rho * z_prev + (1 - rho) * mu_area + sigma_innov * z_t_raw
                z_list.append(z_t)

            z = pm.Deterministic('z', pt.stack(z_list, axis=1), dims=('area', 'year'))

            # ── Census constraint ─────────────────────────────────────────
            _build_census_constraint(z, D, sigma_census)

            # ── Planning lag ────────────────────────────────────────────────
            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                              self.nu_obs, sigma_plan)
            self.add_ben_likelihood(z, data['E_obs'], sigma_ben=sigma_ben)

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
    Replaces M1h's fixed planning observation noise (sigma_plan) with a
    year-specific noise level drawn from a shared hierarchical prior.

    sigma_base_plan    ~ HalfNormal(sigma=5)
    sigma_year_offset  ~ HalfNormal(sigma=2, shape=n_years)
    sigma_obs_plan[t]  = sigma_base_plan + sigma_year_offset[t]

    This captures the hypothesis that planning data quality varies over the
    intercensal window — e.g. system changes in 2013-2016, COVID in 2020-21.
    Inspecting the posterior of sigma_obs_plan by year is a diagnostic in its
    own right.  BEN noise remains fixed.

    Rebuilt on M1h's plain StudentT planning likelihood — see M5's
    docstring for why the zero-inflated version was dropped.
    """

    name        = 'M8'
    description = 'M1h + time-varying planning observation noise'
    var_names   = ['sigma_slab', 'sigma_ben', 'lambda_weights',
                   'sigma_base_plan', 'sigma_year_offset']
    max_lag     = 3

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference = _build_pre_inference(data, self.max_lag)

        with pm.Model(coords=self._default_coords()) as model:

            mu_area = D / n_years  # (n_areas,) numpy constant

            sigma_slab = pm.Gamma('sigma_slab', alpha=2, beta=1 / 12)
            z          = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab,
                                   dims=('area', 'year'))

            _build_census_constraint(z, D, sigma_census)

            _, P_mean = _build_lag(z, pre_inference, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag)

            # ── Time-varying planning noise ───────────────────────────────
            sigma_base_plan   = pm.HalfNormal('sigma_base_plan',   sigma=5)
            sigma_year_offset = pm.HalfNormal('sigma_year_offset',
                                               sigma=2, shape=n_years)
            sigma_obs_plan    = pm.Deterministic(
                'sigma_obs_plan', sigma_base_plan + sigma_year_offset)

            pm.StudentT('P_like', nu=self.nu_obs, mu=P_mean,
                       sigma=sigma_obs_plan[None, :], observed=data['P_obs'])

            sigma_ben = pm.HalfNormal('sigma_ben', sigma=2)
            self.add_ben_likelihood(z, data['E_obs'], sigma_ben=sigma_ben)

        self.model = model
        return model


# ── M9: Symmetric temporal reconciliation ────────────────────────────────────

class M9(DwellingModel):
    """
    Per-area hierarchical sigma_slab + independently-lagged temporal
    misallocation for BOTH planning and BEN.

    M0h/M1h/M5/M6/M8 all pin z's prior mean to a fixed mu_area = D[a]/n_years
    and share a SINGLE GLOBAL sigma_slab scalar across every area and year.
    Empirically, sigma_slab collapses to ~0.04-0.11 (vs a Gamma(2,1/12)
    prior mean of 24) in every one of those models, while sigma_plan/
    sigma_ben inflate to ~8.6-12.6 to compensate — z ends up flat and
    overconfident. The reason: de-meaning each area's P_obs/E_obs series
    and correlating their year-to-year deviations gives a mean correlation
    of ~+0.01 across areas (near zero) even though each source individually
    has large within-area year-to-year variance. Since z is the one shared
    mean feeding both P_like and E_like, only the part P and E agree on
    rewards growing sigma_slab — which is ~0 on average — so a single
    global scalar shrinks to ~0 and the disagreement gets dumped into
    sigma_plan/sigma_ben instead, which pay no penalty for cross-source
    disagreement. (M0/M1/M7, which don't pin a fixed per-area mean, don't
    show this collapse — their sigma_slab is forced wide to cover
    cross-area heterogeneity in census rates, and that width incidentally
    also permits real cross-year shape.)

    Domain framing: P and E signal should both be trusted as real change
    (except rare true-error cases); disagreement is mainly explained by
    incompleteness and/or misallocation in time and/or space, not generic
    noise. z should show uncertainty only where disagreement survives
    after accounting for those mechanisms.

    Two changes, bundled deliberately (they compete for the same residual
    variance, so must land in the same model — without E-lag, per-area
    sigma_slab would absorb variance that's really a timing offset;
    without per-area sigma_slab, E-lag alone reproduces the original
    collapse on better-aligned residuals):

    1. sigma_slab becomes per-area (non-centred log-normal hierarchy via
       _build_z_prior_hierarchical) instead of one global scalar — lets
       each area's z-temporal-freedom be data-driven rather than crushed
       by a population-wide shared scalar.
    2. BEN gets its own learned lag distribution (lambda_weights_E),
       mirroring planning's (lambda_weights_P), via _build_lag called
       twice with independent pre-inference arrays. Previously BEN was
       assumed perfectly synchronous with z (add_ben_likelihood, mu=z
       directly) — notebook 4.0 section 4's cross-correlation analysis
       found a real, structured, non-zero peak lag between P and E,
       contradicting that assumption.

    sigma_ben's prior widens to HalfNormal(5) (matching sigma_plan, not
    the HalfNormal(2) used in M1h/M5/M6/M8) — that tighter prior predates
    both giving E its own lag mechanism and the finding that E's raw
    year-to-year std is empirically larger than P's (~26.8 vs ~15.6
    median per area), so it was never well-justified to begin with.

    Deliberately excludes spatial misallocation (candidate M10) and a
    per-area/source capture-rate term (candidate M11, motivated by
    notebook 4.0 section 10's log(P/E) area-effect decomposition) so each
    mechanism's marginal contribution can be diagnosed independently
    before stacking further complexity — see M5's docstring for why this
    repo treats combined-complexity big-bang models as high risk.

    `sample_kwargs` overrides target_accept to 0.95 — the first override
    in this repo. z remains centred on a now-stochastic per-area
    sigma_slab[a] (not just a fixed mu_area), a weaker justification for
    centring than M0h's, so this is expected to be the first model here
    needing target_accept above the 0.9 default. If divergences persist
    and cluster in specific low-signal areas (informative, not noise),
    escalate: raise tune to ~1000, then try nutpie's adaptation='flow'
    (plumbed via DwellingModel._NUTPIE_PASSTHROUGH_KEYS).
    """

    name        = 'M9'
    description = 'Per-area hierarchical sigma_slab + independent P/E temporal lag'
    var_names   = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
                   'sigma_plan', 'sigma_ben',
                   'lambda_weights_P', 'lambda_weights_E']
    max_lag       = 3
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        with pm.Model(coords=self._default_coords()) as model:

            _, sigma_slab, z, _, _ = _build_z_prior_hierarchical(
                D, n_areas, n_years)

            _build_census_constraint(z, D, sigma_census)

            _, P_mean = _build_lag(z, pre_inference_P, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_P')
            _, E_mean = _build_lag(z, pre_inference_E, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_E')

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=5)

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                              self.nu_obs, sigma_plan)
            _build_planning_likelihood_simple(E_mean, data['E_obs'],
                                              self.nu_obs, sigma_ben,
                                              name='E_like')

        self.model = model
        return model


# ── M10: Per-borough sigma_slab + per-area capture-rate ──────────────────────

class M10(DwellingModel):
    """
    Per-borough hierarchical sigma_slab (replacing M9's per-area version,
    which failed to converge: mu_log_sigma r-hat=1.17, ESS=19 across 200
    areas — a classic hierarchical funnel) + M9's independent P/E temporal
    lag (unchanged) + new per-area capture-rate scaling of both P and E
    likelihoods.

    Motivation: M9's own diagnostics showed sigma_plan/sigma_ben
    essentially UNCHANGED from M1h (8.6/12.9 vs 8.6/12.65) despite the
    per-area sigma_slab hierarchy — cross-source disagreement was still
    being dumped into observation noise rather than explained, and
    sigma_slab itself collapsed to a near-constant ~0.16 across all 200
    areas (no real cross-area differentiation). A pre-check re-running
    notebook 4.0's log(P/E) two-way ANOVA (area + year + residual) on
    P/E series shifted by M9's own learned mean lags found the area-effect
    share of variance did NOT shrink after lag correction — total
    disagreement variance actually grew (2.10 -> 3.13, var_area
    0.51 -> 0.94). This is consistent with a real per-area RECORDING-RATE
    difference between P and E that is distinct from a timing problem —
    kappa_P[a]/kappa_E[a] targets exactly that.

    kappa multiplies the LAGGED mean before each source's likelihood:
    P_mean = kappa_P[:,None] * P_mean_lagged, same for E. It is a
    first-moment (mean-scale) parameter, not a second-moment (spread)
    parameter like sigma_slab — better identified per-area from the same
    ~10-20 observations/area that broke M9's per-area sigma_slab, because
    z's overall level is already pinned per-area by both the tight census
    constraint and the mu_area = D[a]/n_years prior mean, so kappa fits a
    ratio against an externally anchored target rather than a
    free-floating one. Residual risk: D[a] ~= 0 areas have a vanishing
    likelihood gradient w.r.t. kappa there (harmless — the shrinkage
    prior reverts it to ~=1, a no-op — but check for this rather than
    assume uniform identification across areas).

    kappa_P and kappa_E share a single sigma_kappa hyperprior — halves the
    new hyperparameter count vs. independent scales (mitigating the kind
    of funnel that broke M9), and there's no way to separately identify
    each source's heterogeneity from notebook 4.0 section 10's *combined*
    log(P/E) area effect alone. If diagnostics later show one source
    shrinking much harder toward 1 than the other despite real signal
    there, that's evidence for splitting the prior in a follow-up.

    sigma_kappa_prior=0.68 is CALIBRATED, not a placeholder: assuming
    log(P/E)[a] ~= log(kappa_P[a]) - log(kappa_E[a]) up to residual/year
    effects, and Var(log kappa_P) ~= Var(log kappa_E) = sigma_kappa^2,
    then Var(area_effect) ~= 2*sigma_kappa^2. Using the lag-corrected
    ANOVA's var_area=0.9367 (std(alpha_hat)~=0.968, the "residual
    disagreement after timing is accounted for" number):
    sigma_kappa_prior ~= 0.968/sqrt(2) ~= 0.68.

    Deliberately excludes per-area/source heterogeneous lag (candidate
    M11, deferred pending this model's results) so capture-rate's
    marginal contribution can be diagnosed independently — see M5's
    docstring for why this repo avoids combined-complexity big-bang
    models.

    Requires 'borough_idx' (int array, shape n_areas) and 'n_boroughs'
    (int) in the data dict — derive with
    housing_projections.data.make_borough_idx(data['gdf']). Raises
    ValueError if absent (mirrors M7's pattern).

    `sample_kwargs` overrides target_accept to 0.95, matching M9 — this
    model still has two non-centred hierarchies (sigma_slab_borough,
    kappa), even though the sigma_slab one now has far fewer groups
    (n_boroughs vs 200 areas). If divergences persist, escalate: raise
    tune, then nutpie adaptation='flow' (plumbed via
    DwellingModel._NUTPIE_PASSTHROUGH_KEYS), then tighten
    sigma_kappa_prior itself.
    """

    name              = 'M10'
    description       = ('Per-borough sigma_slab + independent P/E lag '
                          '+ per-area capture-rate scaling')
    var_names         = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab_borough',
                          'sigma_kappa', 'sigma_plan', 'sigma_ben',
                          'lambda_weights_P', 'lambda_weights_E']
    max_lag           = 3
    sigma_kappa_prior = 0.68  # calibrated — see docstring
    sample_kwargs     = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        if 'borough_idx' not in data:
            raise ValueError(
                "M10 requires 'borough_idx' (int array, shape n_areas) and "
                "'n_boroughs' (int) in the data dict. "
                "Derive them with housing_projections.data.make_borough_idx(data['gdf'])."
            )
        n_boroughs      = data['n_boroughs']
        borough_idx     = data['borough_idx']
        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        with pm.Model(coords=self._default_coords()) as model:

            _, sigma_slab, z, _, _, sigma_slab_borough = (
                _build_z_prior_hierarchical_borough(
                    D, n_areas, n_years, borough_idx, n_boroughs))

            _build_census_constraint(z, D, sigma_census)

            _, P_mean_lagged = _build_lag(z, pre_inference_P, n_areas, n_years,
                                          self.n_lags, self.lag_alpha, self.max_lag,
                                          name='lambda_weights_P')
            _, E_mean_lagged = _build_lag(z, pre_inference_E, n_areas, n_years,
                                          self.n_lags, self.lag_alpha, self.max_lag,
                                          name='lambda_weights_E')

            sigma_kappa = pm.HalfNormal('sigma_kappa', sigma=self.sigma_kappa_prior)
            kappa_P = _build_capture_rate(n_areas, 'P', sigma_kappa)
            kappa_E = _build_capture_rate(n_areas, 'E', sigma_kappa)

            P_mean = kappa_P[:, None] * P_mean_lagged
            E_mean = kappa_E[:, None] * E_mean_lagged

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=5)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=5)

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                              self.nu_obs, sigma_plan)
            _build_planning_likelihood_simple(E_mean, data['E_obs'],
                                              self.nu_obs, sigma_ben,
                                              name='E_like')

        self.model = model
        return model


# ── M11: Joint agreement-gated P/E likelihood ─────────────────────────────────

class M11(DwellingModel):
    """
    M9 (per-area hierarchical sigma_slab + independent P/E temporal lag)
    with its two independent per-source StudentT likelihoods replaced by
    _build_agreement_gated_likelihood — a single joint mixture likelihood
    sharing one marginalised discrete "agree/disagree" state per
    area-year across BOTH sources.

    Motivating gap: M9's own diagnostics (and M10's docstring) found that
    cross-source disagreement gets dumped whole-cloth into sigma_plan/
    sigma_ben regardless of per-area sigma_slab — nothing in M0h..M10
    reacts specifically to P and E disagreeing with EACH OTHER in a given
    area-year, only to either source individually sitting far from z
    (already handled, weakly, by StudentT's tails). This model gives each
    area-year a mixture over:

      agree    (prob rho):    P_obs ~ St(P_mean, sigma_agree_plan)
                               E_obs ~ St(E_mean, sigma_agree_ben)
      disagree (prob 1-rho):  P_obs ~ St(mu_area, sigma_disagree_plan)
                               E_obs ~ St(mu_area, sigma_disagree_ben)

    In the disagree branch both sources are explained by the area's fixed
    long-run rate rather than z's current lag-weighted mean, so a
    disagreeing area-year contributes ~zero gradient to z — z reverts
    toward mu_area for that year instead of being pulled toward a
    compromise between two conflicting sources. rho is a single global
    Beta-distributed mixture weight, deliberately not per-area (see
    _build_z_prior_hierarchical's docstring on the funnel risk M9 hit
    going per-area) — revisit only if diagnostics show a single global
    agreement rate is too coarse across boroughs/areas.

    sigma_agree_plan/ben get a tighter HalfNormal(3) prior than M9's
    sigma_plan/sigma_ben HalfNormal(5) — the agree branch only has to
    explain observation noise in already-agreeing years, not the whole
    mixed population M9's single likelihood had to absorb.
    sigma_disagree_plan/ben get a wide HalfNormal(20) prior — this branch
    must explain a full idiosyncratic, z-independent realisation, so it
    should be free to land well above M9's fitted 8.6/12.65.

    Deliberately excludes M10's capture-rate term so the two competing
    explanations for disagreement (per-area recording-rate bias vs.
    genuine cross-source conflict) can be diagnosed independently before
    stacking — see M5's docstring for why this repo treats
    combined-complexity models as high risk.

    New diagnostic surface: 'agreement_prob' (dims=('area', 'year')) is
    the posterior probability mass on the agree branch per area-year —
    check it shows real structure (doesn't collapse to ~0 or ~1
    everywhere) and correlates with -|P_obs - E_obs| as a sanity check.

    Convergence risk: rho trades off against sigma_disagree_plan/ben —
    a small rho with tight disagree-sigma can mimic a large rho with wide
    disagree-sigma over most of the data, so check rho's posterior isn't
    glued to a prior-driven boundary before trusting agreement_prob.
    `sample_kwargs` overrides target_accept to 0.95, matching M9 (same
    per-area sigma_slab hierarchy).

    UPDATE (post first run): the first sample (centred z, per M9's
    default) failed to converge badly — max r-hat 2.80, 1575 divergences,
    chains splitting into two (rho_agree, sigma_agree_plan) basins. Area
    E01002694 (D=9; P/E both silent except a corroborating 7/8 spike in
    one year) showed WHY: sigma_slab[a] collapsed to ~0.2-0.5 (population
    10th percentile) and z stayed flat at ~0.9/year in every chain,
    because reaching the correct concentrated-burst solution requires
    moving sigma_slab[a] and z[a, burst_year] together — a classic
    hierarchical funnel, worse under z's CENTRED parameterisation. Now
    built with _build_z_prior_hierarchical(..., non_centered=True) to
    test whether decoupling z from sigma_slab's scale fixes the sampling
    geometry. A separate, likely unrelated contributor: r-hat severity
    also scales with each area's |D - (sum(P_obs)+sum(E_obs))/2| (the
    census/data coverage gap) — areas where P and E jointly capture only
    a small fraction of the census-implied change (e.g. E01002802: D=206,
    captured~25) have a genuinely under-identified year-allocation
    problem that reparameterisation alone won't resolve.
    """

    name        = 'M11'
    description = ('M9 + joint agreement-gated mixture likelihood '
                   '(explicit P/E disagreement gating)')
    var_names   = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_agree', 'lambda_weights_P', 'lambda_weights_E']
    max_lag       = 3
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, _, z, _, _ = _build_z_prior_hierarchical(
                D, n_areas, n_years, non_centered=True)

            _build_census_constraint(z, D, sigma_census)

            _, P_mean = _build_lag(z, pre_inference_P, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_P')
            _, E_mean = _build_lag(z, pre_inference_E, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_E')

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_agree           = pm.Beta('rho_agree', alpha=2, beta=2)

            _build_agreement_gated_likelihood(
                P_mean, E_mean, data['P_obs'], data['E_obs'], mu_area,
                sigma_agree_plan, sigma_agree_ben,
                sigma_disagree_plan, sigma_disagree_ben,
                rho_agree, self.nu_obs)

        self.model = model
        return model


# ── M12: Independent per-source agreement gating ──────────────────────────────

class M12(DwellingModel):
    """
    M11 with its JOINT (P, E) agreement-gated mixture replaced by
    _build_independent_agreement_gated_likelihood — two independent
    per-source mixtures, each deciding whether ITS OWN source is
    informative this area-year without requiring the other to agree.

    Motivation: an empirical area-year taxonomy (see the builder's own
    docstring) found that same-year cross-source CONFLICT — the case
    M11's joint mixture was built to target — is rare (2% of area-years),
    while the dominant non-trivial pattern (38%) is one source being
    genuinely informative while the other is simply silent that year,
    not contradicting it. M11's joint mixture has no way to represent
    that: an "agree" branch that also has to explain a silent second
    source, or a "disagree" branch that also has to explain an already
    z-shifted first source, both fit poorly. Two independent gates let
    each source's own state (informative / not) be assessed on its own
    evidence.

    rho_P and rho_E are independent global Beta(2,2) mixture weights
    (not shared, unlike a single rho_agree) — the empirical P-only vs
    E-only active rates differ by >2x, so forcing one shared rate would
    misrepresent whichever source it doesn't match.

    Everything else — non-centred per-area sigma_slab hierarchy,
    independent P/E temporal lag, sigma_agree/disagree priors — is
    unchanged from M11, so any difference in convergence or z-path
    behaviour between M11 and M12 isolates the marginal effect of
    per-source vs. joint gating specifically.

    New diagnostic surface: 'agreement_prob_P', 'agreement_prob_E'
    (each dims=('area', 'year')) — cross-check these against the raw
    area-year taxonomy: empirically "disjoint" area-years should show
    high responsibility on the active source and low on the silent one,
    not a compromise on both as M11 forces.

    `sample_kwargs` overrides target_accept to 0.95, matching M9/M11.
    """

    name        = 'M12'
    description = ('M11 + independent per-source agreement gating '
                   '(replaces joint P/E mixture with two independent ones)')
    var_names   = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_P', 'rho_E', 'lambda_weights_P', 'lambda_weights_E']
    max_lag       = 3
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()
        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, _, z, _, _ = _build_z_prior_hierarchical(
                D, n_areas, n_years, non_centered=True)

            _build_census_constraint(z, D, sigma_census)

            _, P_mean = _build_lag(z, pre_inference_P, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_P')
            _, E_mean = _build_lag(z, pre_inference_E, n_areas, n_years,
                                   self.n_lags, self.lag_alpha, self.max_lag,
                                   name='lambda_weights_E')

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_P               = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E               = pm.Beta('rho_E', alpha=2, beta=2)

            _build_independent_agreement_gated_likelihood(
                P_mean, E_mean, data['P_obs'], data['E_obs'], mu_area,
                sigma_agree_plan, sigma_agree_ben,
                sigma_disagree_plan, sigma_disagree_ben,
                rho_P, rho_E, self.nu_obs)

        self.model = model
        return model


# ── M13: Per-record temporal reallocation ─────────────────────────────────────

class M13(DwellingModel):
    """
    M12 (independent per-source agreement gating) with _build_lag's
    population-level lag CONVOLUTION removed entirely and replaced by
    _build_temporal_reallocation_likelihood — a per-ACTIVE-RECORD
    marginalised temporal offset, called once per source.

    Motivation: M11/M12 both compare P_obs/E_obs against a lag-weighted
    z (one shared Dirichlet-distributed lag distribution per source,
    convolved into every year alike). An empirical offset analysis (raw
    P_obs/E_obs, 200-area sample) found each area's actual P/E timing
    match is idiosyncratic — 54.8% of matched pairs land on the exact
    same year, with an asymmetric spread out to +/-2 years for most of
    the rest — so a single population-average kernel is right in
    aggregate but wrong for any specific record, which is exactly what
    left M11's E01002694 unable to connect an unambiguous same-year P/E
    spike to z at all (see that model's docstring and
    _build_temporal_reallocation_likelihood's for the concrete numbers).

    This model removes lambda_weights_P/E and the pre_inference lag
    machinery entirely — there is no smoothing kernel left to fit.
    Instead, each source gets its own small pi_offset_P / pi_offset_E
    (Dirichlet over 2*max_offset+1 = 5 candidate shifts, default
    +/-2 years), applied only to records where |P_obs| or |E_obs|
    exceeds active_threshold (3 dwellings/year) — roughly 1,200 of the
    2,000 area-year cells in the 200-area sample; the rest compare
    directly against z[a,t], unchanged from a simple per-cell
    comparison.

    Deliberately excludes spatial reallocation (candidate M14) — an
    empirical Moran's I / best-1-hop-neighbour-cancellation check found
    no signal beyond a random-relabelling null for either P/E
    disagreement or the census/(P+E) coverage gap, so spatial
    misallocation between queen neighbours doesn't look like a
    productive mechanism to add on top of this.

    Validation to check once sampled: pi_offset_P / pi_offset_E's fitted
    posterior should roughly reproduce the empirical offset histogram
    above (peaked at 0, biased toward negative -- E leading P -- with a
    long-ish tail); if it doesn't, that's a sign the marginalisation
    isn't doing what's intended. Also check E01002694 specifically:
    z[2014] should now be free to track the P=7/E=8 spike directly,
    since the same-year (offset=0) candidate is compared against z[2014]
    exactly rather than a lag-smoothed blend.

    `sample_kwargs` overrides target_accept to 0.95, matching M9/M11/M12.
    """

    name        = 'M13'
    description = ('M12 + per-record temporal reallocation '
                   '(replaces the lag convolution with a per-active-record offset)')
    var_names   = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E']
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}
    active_threshold = 3.0
    max_offset       = 2

    def build(self):
        data, n_areas, n_years, D, sigma_census = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, _, z, _, _ = _build_z_prior_hierarchical(
                D, n_areas, n_years, non_centered=True)

            _build_census_constraint(z, D, sigma_census)

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_P               = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E               = pm.Beta('rho_E', alpha=2, beta=2)

            _build_temporal_reallocation_likelihood(
                z, data['P_obs'], mu_area, sigma_agree_plan, sigma_disagree_plan,
                rho_P, self.nu_obs, name='P',
                active_threshold=self.active_threshold, max_offset=self.max_offset)
            _build_temporal_reallocation_likelihood(
                z, data['E_obs'], mu_area, sigma_agree_ben, sigma_disagree_ben,
                rho_E, self.nu_obs, name='E',
                active_threshold=self.active_threshold, max_offset=self.max_offset)

        self.model = model
        return model


class M14(DwellingModel):
    """
    M13 with its z prior replaced by _build_z_prior_profile_library — the
    "flat unless something happened" generative story discussed in chat,
    instead of a free per-cell hierarchical Normal.

    Motivation: every model so far (M9-M13) lets z[a, :] be a free
    per-cell continuous vector regardless of how much is actually going
    on in an area that year; "quiet vs active" only ever showed up
    implicitly, through the observation-likelihood gating (rho_P/rho_E,
    sigma_agree vs sigma_disagree). This model instead makes the z PRIOR
    itself default to flat (D[a]/n_years every year) and requires an
    explicit, discrete "this area has one concentrated active year"
    choice (profile_k) before z is allowed to deviate — see
    _build_z_prior_profile_library and _build_zero_sum_profile_library
    for the exact construction. A naive version of this idea (free
    per-year deltas constrained to sum to zero) would tightly couple the
    discrete year-choice to the continuous magnitudes of every OTHER
    active year in the same area, which is a bad combination for any
    sampler; building the zero-sum property into a small fixed library of
    candidate shapes instead removes that coupling entirely, at the cost
    of only supporting ONE concentrated active year per area (not
    combinations of several) in this first version.

    The observation-likelihood side (independent rho_P/rho_E gating,
    per-record temporal-offset marginalisation) is unchanged from M13 —
    deliberately NOT adding a second reallocation mechanism on top of the
    z-prior's own single-active-year choice, since both would be trying
    to explain the same "which year did this really happen in" ambiguity
    from two different directions (z's own timing vs. recording lag)
    and would likely fight each other rather than combine cleanly.

    z sums to D EXACTLY by construction (see
    _build_z_prior_profile_library), so — unlike every other model in
    this module — there is no _build_census_constraint call and no
    sigma_census: the census total is an identity here, not a soft
    likelihood term. This also means it is not comparable via
    census-residual diagnostics that assume some slack.

    profile_k is a discrete per-area Categorical, which nutpie cannot
    compile (it requires a fully differentiable model) — sample() is
    overridden below to always fall back to PyMC's own sampler, which
    auto-assigns CategoricalGibbsMetropolis to profile_k and NUTS to
    everything else as a CompoundStep, regardless of what use_nutpie is
    passed as.

    Things to check once sampled: (1) whether CategoricalGibbsMetropolis
    mixes acceptably for 200 largely-independent per-area categoricals —
    this is the empirical test of whether removing the coupling actually
    helped, per the design discussion; (2) whether pi_profile's fitted
    posterior puts most mass on the null (flat) row, consistent with the
    area/year taxonomy's ~50% quiet cells; (3) whether amplitude/profile_k
    for E01002694, E01002802, E01002719 recovers something sensible
    against what we already know about those areas.
    """

    name        = 'M14'
    description = ('M13 + flat-baseline z prior with a marginalised '
                   'single-active-year profile library (replaces the free '
                   'per-cell hierarchical z prior)')
    var_names   = ['mu_log_sigma', 'tau_log_sigma', 'sigma_slab', 'amplitude',
                   'pi_profile',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E']
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}
    active_threshold = 3.0
    max_offset       = 2

    def sample(self, use_nutpie=True, **kwargs):
        """
        profile_k is discrete -- nutpie cannot compile this model
        (requires full differentiability), so this always samples with
        PyMC's own sampler regardless of the use_nutpie argument.
        """
        return super().sample(use_nutpie=False, **kwargs)

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, _, z, _, _, _, _, _, _ = _build_z_prior_profile_library(
                D, n_areas, n_years)

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_P               = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E               = pm.Beta('rho_E', alpha=2, beta=2)

            _build_temporal_reallocation_likelihood(
                z, data['P_obs'], mu_area, sigma_agree_plan, sigma_disagree_plan,
                rho_P, self.nu_obs, name='P',
                active_threshold=self.active_threshold, max_offset=self.max_offset)
            _build_temporal_reallocation_likelihood(
                z, data['E_obs'], mu_area, sigma_agree_ben, sigma_disagree_ben,
                rho_E, self.nu_obs, name='E',
                active_threshold=self.active_threshold, max_offset=self.max_offset)

        self.model = model
        return model


class M15(DwellingModel):
    """
    M14 with the null-row-redundancy problem fixed: replaces the flat
    Normal(0, sigma_slab) amplitude prior + explicit null library row with
    a regularised (Finnish) horseshoe prior directly on amplitude, via
    _build_z_prior_profile_library_horseshoe — see that builder's
    docstring for the full motivation and construction.

    In short: M14 diagnostics showed pi_profile did NOT concentrate on the
    null row as intended (8.1%, the LEAST likely of 11 rows) — nothing in
    the likelihood structurally favoured "no activity" over "activity
    with amplitude~0", since a plain Normal(0, sigma_slab) is merely free
    to shrink, not incentivised to. The horseshoe prior fixes this at its
    source (genuine sparsity-inducing shrinkage), so the null row is
    dropped entirely — profile_k ranges over the n_years spike rows only.

    Also checked and found NOT viable: pymc_extras.marginalize() on
    profile_k (to avoid CategoricalGibbsMetropolis and its role in M14's
    1553 divergences). It raises "No RVs depend on marginalized RV
    profile_k" — it only traces dependencies through observed
    Distributions, not through a Deterministic (z) feeding a pm.Potential
    (the pattern this whole module uses). Using it would require
    rebuilding _build_temporal_reallocation_likelihood as explicit
    per-candidate Distributions rather than a hand-rolled Potential — the
    same expensive rewrite already scoped out in
    docs/model-progression-notes.md, not a shortcut around it. profile_k
    is therefore STILL sampled via literal CategoricalGibbsMetropolis
    here: this model isolates whether M14's divergences came from the
    redundant-null geometry (which this fixes) or from discrete-Gibbs
    sampling itself being rough regardless (which this does not fix).

    The observation-likelihood side is unchanged from M13/M14
    (independent rho_P/rho_E gating, per-record temporal-offset
    marginalisation). z sums to D exactly by construction, as in M14 — no
    _build_census_constraint call.

    Still non-differentiable (profile_k discrete) — sample() is
    overridden to always fall back to PyMC's own sampler, same as M14.
    """

    name        = 'M15'
    description = ('M14 + regularised horseshoe prior on amplitude, no null '
                   'library row (fixes the null-row redundancy found in M14)')
    var_names   = ['tau_amplitude', 'c2_amplitude', 'amplitude', 'pi_profile',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E']
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}
    active_threshold = 3.0
    max_offset       = 2
    p0          = None   # None -> n_areas // 4 at build time
    slab_scale  = 12.0

    def sample(self, use_nutpie=True, **kwargs):
        """
        profile_k is discrete -- nutpie cannot compile this model
        (requires full differentiability), so this always samples with
        PyMC's own sampler regardless of the use_nutpie argument.
        """
        return super().sample(use_nutpie=False, **kwargs)

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, z, _, _, _, _, _, _, _ = _build_z_prior_profile_library_horseshoe(
                D, n_areas, n_years,
                p0=self.p0 or n_areas // 4, slab_scale=self.slab_scale)

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_P               = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E               = pm.Beta('rho_E', alpha=2, beta=2)

            _build_temporal_reallocation_likelihood(
                z, data['P_obs'], mu_area, sigma_agree_plan, sigma_disagree_plan,
                rho_P, self.nu_obs, name='P',
                active_threshold=self.active_threshold, max_offset=self.max_offset)
            _build_temporal_reallocation_likelihood(
                z, data['E_obs'], mu_area, sigma_agree_ben, sigma_disagree_ben,
                rho_E, self.nu_obs, name='E',
                active_threshold=self.active_threshold, max_offset=self.max_offset)

        self.model = model
        return model


class M16(DwellingModel):
    """
    M15 with profile_k fully MARGINALISED via pymc_extras.marginalize(),
    instead of sampled via literal CategoricalGibbsMetropolis — the
    rewrite scoped out (as "expensive") in docs/model-progression-notes.md,
    made tractable by a discovery made mid-session: pymc_extras.marginalize()
    doesn't require hand-vectorising the likelihood over a profile_k axis
    ourselves; it only requires the likelihood to be built as a genuine
    Distribution/RV (pm.CustomDist) rather than a pm.Potential, and it does
    the substitute-each-candidate-and-logsumexp step automatically. See
    _build_temporal_reallocation_likelihood_marginalizable for that
    rewrite, applied identically to M15's likelihood (independent
    rho_P/rho_E gating, per-record temporal-offset marginalisation) —
    unchanged except for the Potential -> CustomDist swap.

    Three things had to be verified empirically before this was usable,
    each a genuine gotcha, not a guess:

    1. z cannot be a pm.Deterministic when profile_k is marginalised —
       marginalize() raises "Cannot marginalize profile_k due to
       dependent Deterministic z" (a Deterministic needs one concrete
       value per draw; profile_k no longer has one once marginalised
       out). So z is built as a bare pytensor expression here
       (_build_z_prior_profile_library_horseshoe(..., wrap_z_as_deterministic
       =False)) and never appears directly in the sampled trace.

    2. Losing z as a live Deterministic means every diagnostic/plot this
       session has been built around (z-timeseries, agreement_prob,
       pointwise log-likelihood) has nothing to read post-sampling.
       pymc_extras.recover_marginals() gives profile_k's posterior back
       AFTER sampling (conditioned on the sampled continuous parameters);
       sample() below reconstructs z, agreement_prob_P/E, and P_like/
       E_like's pointwise log-likelihood in numpy from the recovered
       profile_k + amplitude, using the exact same formulas as the
       pytensor builders. This is extra bookkeeping load-bearing on
       correctness — the numpy reimplementation must track the pytensor
       one exactly (offsets, active/inactive split, agree/disagree gate).

    3. When P_like and E_like (two separate CustomDists) share the SAME
       marginalised profile_k, pymc_extras cannot cleanly separate their
       pointwise log-likelihoods afterward even via
       pm.compute_log_likelihood() — verified directly: P_like's log_
       likelihood came back shaped (chain, draw) instead of (chain, draw,
       area), a degenerate scalar-per-draw total, with an explicit
       NonSeparableLogpWarning ("joint logp terms will be assigned to the
       first value"). The JOINT marginal logp is still correct (verified
       by perturbing E_obs and confirming the marginal model's logp
       changes accordingly) — only the per-node ATTRIBUTION breaks. This
       is why sample() reconstructs P_like/E_like pointwise log-
       likelihood manually too, rather than calling
       pm.compute_log_likelihood() on the marginalised model.

    Because z sums to D by construction (same as M14/M15) and no
    Deterministic depends on profile_k anywhere in build(), the resulting
    marginal_model is FULLY DIFFERENTIABLE — nutpie-compatible again,
    unlike M14/M15 (which always forced use_nutpie=False). This is the
    actual point of the whole exercise: if M14/M15's divergences were
    caused by CategoricalGibbsMetropolis-vs-NUTS compound-step
    instability rather than the redundant-null geometry alone, this
    should fix it; M15's horseshoe fix is still in effect for the
    null-row-redundancy problem specifically.

    LOO NOTE: _build_temporal_reallocation_likelihood_marginalizable's
    CustomDist returns one log-density value PER AREA (all 10 years
    together), not per (area, year) cell like M11-M13 — required for
    profile_k's marginalisation to attribute each area's contribution
    correctly. This means the natural "held out" unit for LOO here is one
    area, not one area-year cell — not directly apples-to-apples with
    M11-M13's granularity, a real difference worth remembering when
    comparing ELPD across the two families, not just an implementation
    detail to gloss over.

    var_names/sample_kwargs mirror M15 (same continuous parameters, same
    target_accept). p0/slab_scale defaults also mirror M15.
    """

    name        = 'M16'
    description = ('M15 with profile_k marginalised via pymc_extras.marginalize() '
                   'instead of sampled via CategoricalGibbsMetropolis')
    var_names   = ['tau_amplitude', 'c2_amplitude', 'amplitude', 'pi_profile',
                   'sigma_agree_plan', 'sigma_agree_ben',
                   'sigma_disagree_plan', 'sigma_disagree_ben',
                   'rho_P', 'rho_E', 'pi_offset_P', 'pi_offset_E']
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}
    active_threshold = 3.0
    max_offset       = 2
    p0          = None   # None -> n_areas // 4 at build time
    slab_scale  = 12.0

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            mu_area, z, _, _, _, _, _, _, _ = _build_z_prior_profile_library_horseshoe(
                D, n_areas, n_years,
                p0=self.p0 or n_areas // 4, slab_scale=self.slab_scale,
                wrap_z_as_deterministic=False)

            sigma_agree_plan    = pm.HalfNormal('sigma_agree_plan', sigma=3)
            sigma_agree_ben     = pm.HalfNormal('sigma_agree_ben', sigma=3)
            sigma_disagree_plan = pm.HalfNormal('sigma_disagree_plan', sigma=20)
            sigma_disagree_ben  = pm.HalfNormal('sigma_disagree_ben', sigma=20)
            rho_P               = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E               = pm.Beta('rho_E', alpha=2, beta=2)

            _build_temporal_reallocation_likelihood_marginalizable(
                z, data['P_obs'], mu_area, sigma_agree_plan, sigma_disagree_plan,
                rho_P, self.nu_obs, name='P',
                active_threshold=self.active_threshold, max_offset=self.max_offset)
            _build_temporal_reallocation_likelihood_marginalizable(
                z, data['E_obs'], mu_area, sigma_agree_ben, sigma_disagree_ben,
                rho_E, self.nu_obs, name='E',
                active_threshold=self.active_threshold, max_offset=self.max_offset)

        self.model = model
        return model

    def sample(self, use_nutpie=True, **kwargs):
        """
        Marginalises profile_k out of self.model, samples the resulting
        fully-differentiable model (nutpie-compatible, unlike M14/M15),
        then recovers profile_k's posterior and reconstructs z,
        agreement_prob_P/E, and P_like/E_like's pointwise log-likelihood
        in numpy — see the class docstring for why each of these steps
        is necessary.
        """
        import pymc_extras as pmx

        if self.model is None:
            self.build()

        merged = {**self.sample_kwargs, **kwargs}
        marginal_model = pmx.marginalize(model=self.model, rvs_to_marginalize=['profile_k'])

        with marginal_model:
            if use_nutpie:
                try:
                    import nutpie
                    compiled = nutpie.compile_pymc_model(marginal_model)
                    nutpie_extra = {
                        k: merged[k] for k in self._NUTPIE_PASSTHROUGH_KEYS
                        if k in merged and merged[k] is not None
                    }
                    trace = nutpie.sample(
                        compiled,
                        draws         = merged.get('draws',         500),
                        tune          = merged.get('tune',          500),
                        chains        = merged.get('chains',        2),
                        target_accept = merged.get('target_accept', 0.9),
                        seed          = merged.get('random_seed',   42),
                        **nutpie_extra,
                    )
                except ImportError:
                    print("nutpie not installed, falling back to PyMC sampler")
                    trace = pm.sample(**merged)
            else:
                trace = pm.sample(**merged)

        trace = pmx.recover_marginals(trace, model=marginal_model, var_names=['profile_k'])
        self.trace = self._reconstruct_posterior(trace)
        return self.trace

    def _reconstruct_posterior(self, trace):
        """
        Rebuild z, agreement_prob_P/E, and P_like/E_like's pointwise
        log-likelihood in numpy from the recovered profile_k + sampled
        amplitude/sigma_agree/sigma_disagree/rho/pi_offset — none of
        these exist natively in the trace because z can't be a
        Deterministic under marginalisation (see class docstring points
        1-3). Reimplements _build_temporal_reallocation_likelihood_
        marginalizable's logp formula exactly, in scipy/numpy instead of
        pytensor, vectorised over (chain, draw).
        """
        import scipy.stats as stats
        from scipy.special import logsumexp as sp_logsumexp

        data = self.data
        n_areas, n_years = data['n_areas'], data['n_years']
        mu_area = data['D'] / n_years
        profile_library = _build_zero_sum_profile_library(n_years, include_null=False)

        profile_k = trace.posterior['profile_k'].values   # (chain, draw, area)
        amplitude = trace.posterior['amplitude'].values    # (chain, draw, area)
        n_chains, n_draws = profile_k.shape[:2]

        z = mu_area[None, None, :, None] + amplitude[..., None] * profile_library[profile_k]

        area_coord = (trace.posterior.coords['area'].values
                      if 'area' in trace.posterior.coords
                      else data['gdf']['LSOA21CD'].tolist())
        base_coords = {
            'chain': trace.posterior.coords['chain'],
            'draw':  trace.posterior.coords['draw'],
            'area':  area_coord,
            'year':  INFER_YEARS,
        }

        new_posterior_vars = {
            'z': xr.DataArray(z, dims=('chain', 'draw', 'area', 'year'), coords=base_coords),
        }
        loglik_vars = {}

        for name, obs, sigma_agree_name, sigma_disagree_name, rho_name, pi_offset_name in [
            ('P', data['P_obs'], 'sigma_agree_plan', 'sigma_disagree_plan', 'rho_P', 'pi_offset_P'),
            ('E', data['E_obs'], 'sigma_agree_ben',  'sigma_disagree_ben',  'rho_E', 'pi_offset_E'),
        ]:
            sigma_agree    = trace.posterior[sigma_agree_name].values     # (chain, draw)
            sigma_disagree = trace.posterior[sigma_disagree_name].values  # (chain, draw)
            rho            = trace.posterior[rho_name].values             # (chain, draw)
            pi_offset      = trace.posterior[pi_offset_name].values       # (chain, draw, K_offset)

            active = np.abs(obs) > self.active_threshold
            active_area, active_year     = np.where(active)
            inactive_area, inactive_year = np.where(~active)

            offsets = np.arange(-self.max_offset, self.max_offset + 1)
            shifted_year = active_year[:, None] + offsets[None, :]
            valid        = (shifted_year >= 0) & (shifted_year < n_years)
            shifted_year_clipped = np.clip(shifted_year, 0, n_years - 1)

            obs_active   = obs[active_area, active_year]
            obs_inactive = obs[inactive_area, inactive_year]

            z_shifted = z[:, :, active_area[:, None], shifted_year_clipped]  # (C,D,n_active,K_off)
            log_p_k = stats.t.logpdf(
                obs_active[None, None, :, None], df=self.nu_obs,
                loc=z_shifted, scale=sigma_agree[:, :, None, None])
            log_pi = np.log(pi_offset)[:, :, None, :]
            log_terms_active = np.where(valid[None, None, :, :], log_pi + log_p_k, -1e10)
            log_p_agree_active = sp_logsumexp(log_terms_active, axis=-1)  # (C,D,n_active)

            log_p_disagree_active = stats.t.logpdf(
                obs_active[None, None, :], df=self.nu_obs,
                loc=mu_area[active_area][None, None, :], scale=sigma_disagree[:, :, None])

            log_lik_agree_active    = np.log(rho)[:, :, None]      + log_p_agree_active
            log_lik_disagree_active = np.log(1 - rho)[:, :, None] + log_p_disagree_active
            log_lik_active = np.logaddexp(log_lik_agree_active, log_lik_disagree_active)
            resp_active    = np.exp(log_lik_agree_active - log_lik_active)

            z_direct = z[:, :, inactive_area, inactive_year]  # (C, D, n_inactive)
            log_p_agree_inactive = stats.t.logpdf(
                obs_inactive[None, None, :], df=self.nu_obs,
                loc=z_direct, scale=sigma_agree[:, :, None])
            log_p_disagree_inactive = stats.t.logpdf(
                obs_inactive[None, None, :], df=self.nu_obs,
                loc=mu_area[inactive_area][None, None, :], scale=sigma_disagree[:, :, None])

            log_lik_agree_inactive    = np.log(rho)[:, :, None]      + log_p_agree_inactive
            log_lik_disagree_inactive = np.log(1 - rho)[:, :, None] + log_p_disagree_inactive
            log_lik_inactive = np.logaddexp(log_lik_agree_inactive, log_lik_disagree_inactive)
            resp_inactive    = np.exp(log_lik_agree_inactive - log_lik_inactive)

            loglik_grid = np.zeros((n_chains, n_draws, n_areas, n_years))
            loglik_grid[:, :, active_area, active_year]     = log_lik_active
            loglik_grid[:, :, inactive_area, inactive_year] = log_lik_inactive

            resp_grid = np.zeros((n_chains, n_draws, n_areas, n_years))
            resp_grid[:, :, active_area, active_year]     = resp_active
            resp_grid[:, :, inactive_area, inactive_year] = resp_inactive

            new_posterior_vars[f'agreement_prob_{name}'] = xr.DataArray(
                resp_grid, dims=('chain', 'draw', 'area', 'year'), coords=base_coords)
            loglik_vars[f'{name}_like'] = xr.DataArray(
                loglik_grid, dims=('chain', 'draw', 'area', 'year'), coords=base_coords)

        trace.posterior = xr.merge([trace.posterior.to_dataset(), xr.Dataset(new_posterior_vars)])
        trace.log_likelihood = xr.Dataset(loglik_vars)
        return trace

