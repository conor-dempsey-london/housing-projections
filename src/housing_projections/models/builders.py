"""
Private builder functions shared across model classes (M0-M16, AZ0-AZ4b).

Each function assembles one reusable piece of a pm.Model — a z prior variant, a lag
convolution, a likelihood term, etc. — composed together by the model classes in
`m_family.py`/`az_family.py`. Split out of the single `models.py` file (which had grown to
~5,000 lines) purely for navigability; `models.py` re-exports every name here so existing
`from housing_projections.models.models import _build_...` imports keep working.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from housing_projections.config import (
    ALL_COLS_BEN,
    ALL_COLS_PLAN,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
)

__all__ = [
    "_build_z_prior",
    "_build_z_prior_hierarchical",
    "_build_z_prior_hierarchical_borough",
    "_build_zero_sum_profile_library",
    "_build_z_prior_profile_library",
    "_build_z_prior_profile_library_horseshoe",
    "_build_zero_sum_z_prior",
    "_build_zero_sum_z_prior_top_boost",
    "_build_zero_sum_z_prior_top_boost_smooth",
    "_build_zero_sum_z_prior_banded",
    "_build_capture_rate",
    "_build_census_constraint",
    "_build_pre_inference",
    "_build_lag",
    "_build_hierarchical_lag",
    "_build_hierarchical_lag_capped",
    "_build_hierarchical_lag_pinned",
    "_build_fixed_lag",
    "_build_hierarchical_lag_horseshoe",
    "_build_hierarchical_lag_regularized_horseshoe",
    "_build_hierarchical_lag_regularized_horseshoe_v2",
    "_build_planning_likelihood_simple",
    "_build_asymmetric_missingness",
    "_build_hierarchical_lag_marginalized",
    "_build_planning_likelihood_marginalized_lag",
    "_build_planning_likelihood_zeroinflated",
    "_build_agreement_gated_likelihood",
    "_build_independent_agreement_gated_likelihood",
    "_build_temporal_reallocation_likelihood",
    "_build_temporal_reallocation_likelihood_marginalizable",
    "_build_backward_reallocation_likelihood",
    "_build_backward_reallocation_likelihood_2way",
    "_build_noise_mixture_likelihood",
    "_build_spatial_misallocation",
]


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


def _build_z_prior_hierarchical(D, n_areas, n_years,
                                mu_log_sigma_prior=(np.log(12), 0.7),
                                tau_log_sigma_prior=0.7,
                                non_centered=False):
    """
    Per-area hierarchical z prior. mu_area pinned to D[a]/n_years (fixed
    constant — same justification as M0h; unaffected by sigma_slab
    becoming per-area, since that argument was about the mean).

    sigma_slab[a] is drawn from a shared log-normal hierarchy, non-centred
    at the hyperparameter level, replacing the single global scalar that
    collapses in M0h/M1h/M5/M6/M8 (a global scalar is only rewarded for
    cross-source-*agreed* temporal signal, which is ~0 on average across
    real P/E data — see M9's docstring).

    non_centered: if False (default — unchanged behaviour for M9/M10), z
    is centred on sigma_slab[a] directly (mu=mu_area fixed, sigma=
    sigma_slab[a] stochastic). If True, z is built non-centred (z_raw ~
    Normal(0,1); z = mu_area + sigma_slab * z_raw), matching the same
    reparameterisation already used one level up for sigma_slab's own
    hyperparameters. Centred z couples z[a,t] and sigma_slab[a] into a
    classic hierarchical funnel: reaching a genuine local burst requires
    moving sigma_slab[a] and z[a,t] together in a correlated way that's
    hard for HMC to traverse when sigma_slab[a] currently sits small — see
    M11's docstring for a concrete area where this trapped the sampler at
    a flat, worse-fitting z despite unambiguous P/E agreement. Try
    non_centered=True first on any model that shows this symptom; it
    targets the sampling geometry only; it does not change the prior.

    Returns (mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled

    mu_log_sigma  = pm.Normal('mu_log_sigma', mu=mu_log_sigma_prior[0],
                              sigma=mu_log_sigma_prior[1])
    tau_log_sigma = pm.HalfNormal('tau_log_sigma', sigma=tau_log_sigma_prior)
    log_sigma_offset = pm.Normal('log_sigma_offset', mu=0, sigma=1,
                                 dims='area')

    sigma_slab = pm.Deterministic(
        'sigma_slab',
        pt.exp(mu_log_sigma + tau_log_sigma * log_sigma_offset),
        dims='area')

    if non_centered:
        z_raw = pm.Normal('z_raw', mu=0, sigma=1, dims=('area', 'year'))
        z = pm.Deterministic(
            'z', mu_area[:, None] + sigma_slab[:, None] * z_raw,
            dims=('area', 'year'))
    else:
        z = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab[:, None],
                      dims=('area', 'year'))

    return mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma


def _build_z_prior_hierarchical_borough(D, n_areas, n_years, borough_idx, n_boroughs,
                                        mu_log_sigma_prior=(np.log(12), 0.7),
                                        tau_log_sigma_prior=0.7):
    """
    Per-borough hierarchical z prior — replaces
    _build_z_prior_hierarchical's per-area version to fix M9's
    mu_log_sigma non-convergence (r-hat=1.17, ESS=19 across 200 areas).
    Same non-centred log-normal hyperparameter structure, but
    log_sigma_offset has shape=n_boroughs (no named PyMC dim — follows
    M7's mu_borough_offset convention; there's no 'borough' coord in
    _default_coords()) instead of dims='area', then broadcasts to a
    per-area sigma_slab via borough_idx so
    diagnostics._check_sigma_slab_vs_disagreement (which requires
    trace.posterior['sigma_slab'] shaped (chain,draw,area)) keeps working
    unmodified.

    Returns (mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma, sigma_slab_borough).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled

    mu_log_sigma  = pm.Normal('mu_log_sigma', mu=mu_log_sigma_prior[0],
                              sigma=mu_log_sigma_prior[1])
    tau_log_sigma = pm.HalfNormal('tau_log_sigma', sigma=tau_log_sigma_prior)
    log_sigma_offset = pm.Normal('log_sigma_offset', mu=0, sigma=1, shape=n_boroughs)

    sigma_slab_borough = pm.Deterministic(
        'sigma_slab_borough',
        pt.exp(mu_log_sigma + tau_log_sigma * log_sigma_offset))  # (n_boroughs,)

    sigma_slab = pm.Deterministic(
        'sigma_slab', sigma_slab_borough[borough_idx], dims='area')  # (n_areas,)

    z = pm.Normal('z', mu=mu_area[:, None], sigma=sigma_slab[:, None],
                  dims=('area', 'year'))

    return mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma, sigma_slab_borough


def _build_zero_sum_profile_library(n_years, include_null=True):
    """
    Fixed (K, n_years) numpy library of deviation shapes from a flat
    baseline, every row summing to EXACTLY zero: row t concentrates a unit
    deviation in year t, offset by a uniform compensating deviation of
    -1/(n_years-1) in every other year. If include_null, row 0 is also a
    null/flat shape (all zeros — "no concentrated activity this decade"),
    giving K = n_years + 1; if not, K = n_years (see
    _build_z_prior_profile_library_horseshoe's docstring for why a model
    might drop the null row deliberately).

    Used by _build_z_prior_profile_library[_horseshoe] so that z =
    flat_baseline + amplitude * profile_library[k] sums to the census
    total D exactly for ANY amplitude or k — the zero-sum property is
    baked into the library itself rather than enforced as a constraint
    across free parameters, which is what avoids the discrete/continuous
    coupling a naive "per-year deltas summing to zero" version would have.
    """
    n_spike_rows = n_years
    offset = 1 if include_null else 0
    K = n_spike_rows + offset
    profiles = np.zeros((K, n_years))
    for t in range(n_years):
        profiles[t + offset, :] = -1.0 / (n_years - 1)
        profiles[t + offset, t] = 1.0
    return profiles


def _build_z_prior_profile_library(D, n_areas, n_years,
                                   mu_log_sigma_prior=(np.log(12), 0.7),
                                   tau_log_sigma_prior=0.7):
    """
    Flat-baseline z prior with a per-area discrete choice of AT MOST ONE
    concentrated "active year", picked from the small fixed library in
    _build_zero_sum_profile_library, instead of _build_z_prior_hierarchical's
    free per-cell Normal.

        z[a, :] = D[a]/n_years + amplitude[a] * profile_library[profile_k[a], :]

    Every library row sums to zero, so z[a, :] sums to D[a] EXACTLY on
    every posterior draw regardless of amplitude[a] or profile_k[a] — the
    census constraint holds by construction. Callers should NOT also call
    _build_census_constraint alongside this prior (no sigma_census slack
    term is needed or meaningful here).

    profile_k[a] ~ Categorical(pi_profile) is discrete. Because every
    library row already sums to zero, amplitude[a] (continuous, real, can
    be negative) is NOT coupled to anything else through a sum constraint
    once profile_k[a] is fixed — unlike a naive "free per-year deltas
    constrained to sum to zero" version, where every proposed continuous
    value has to stay jointly consistent with the other active years'
    values. That decoupling is what makes an ordinary discrete Gibbs step
    viable here: PyMC auto-assigns CategoricalGibbsMetropolis to
    profile_k and NUTS to everything else (a CompoundStep), rather than
    needing an expensive exact marginalisation over profile_k through the
    whole downstream observation likelihood.

    IMPORTANT: this makes the model non-differentiable end-to-end, so it
    cannot be compiled by nutpie — callers must sample with
    use_nutpie=False.

    pi_profile (population-level Dirichlet prior over which library row
    areas tend to pick) is deliberately uninformative, same convention as
    pi_offset (M13) / lambda_weights (M1-M8).

    amplitude[a] ~ Normal(0, sigma_slab[a]), sigma_slab[a] drawn from the
    same non-centred log-normal hierarchy as _build_z_prior_hierarchical
    (kept as-is here; not redesigned by this change).

    Returns (mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma,
    profile_k, amplitude, pi_profile, profile_library).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled

    profile_library   = _build_zero_sum_profile_library(n_years)  # (K, n_years) numpy const
    K                  = profile_library.shape[0]
    profile_library_pt = pt.as_tensor_variable(profile_library)

    mu_log_sigma  = pm.Normal('mu_log_sigma', mu=mu_log_sigma_prior[0],
                              sigma=mu_log_sigma_prior[1])
    tau_log_sigma = pm.HalfNormal('tau_log_sigma', sigma=tau_log_sigma_prior)
    log_sigma_offset = pm.Normal('log_sigma_offset', mu=0, sigma=1, dims='area')
    sigma_slab = pm.Deterministic(
        'sigma_slab',
        pt.exp(mu_log_sigma + tau_log_sigma * log_sigma_offset),
        dims='area')

    pi_profile = pm.Dirichlet('pi_profile', a=np.ones(K))
    profile_k  = pm.Categorical('profile_k', p=pi_profile, dims='area')
    amplitude  = pm.Normal('amplitude', mu=0, sigma=sigma_slab, dims='area')

    z = pm.Deterministic(
        'z',
        mu_area[:, None] + amplitude[:, None] * profile_library_pt[profile_k],
        dims=('area', 'year'))

    return (mu_area, sigma_slab, z, mu_log_sigma, tau_log_sigma,
            profile_k, amplitude, pi_profile, profile_library)


def _build_z_prior_profile_library_horseshoe(D, n_areas, n_years, p0=None, slab_scale=12.0,
                                              wrap_z_as_deterministic=True):
    """
    _build_z_prior_profile_library with the null-row-redundancy problem
    found in that version's first use (M14) fixed at its source.

    M14 diagnostics showed pi_profile did NOT concentrate on the null row
    as intended (8.1% posterior probability — the LEAST likely of the 11
    rows, not the most). On reflection this is because amplitude ~
    Normal(0, sigma_slab) is merely FREE to shrink toward 0, not
    incentivised to: any library row with amplitude~0 is functionally
    identical to the null row, so nothing in the likelihood structurally
    favours "no activity" over "activity with a tiny amplitude."

    Fix: replace the flat Normal(0, sigma_slab) amplitude prior with a
    regularised ("Finnish") horseshoe prior — a genuinely sparsity-
    inducing prior (Carvalho/Polson/Scott 2010; regularisation per
    Piironen & Vehtari 2017) that pulls MOST areas' amplitude hard toward
    zero (behaving like "null" without a dedicated category) while a few
    areas with real signal escape shrinkage entirely, via a per-area local
    scale (lam) combined with a shared global scale (tau), with an
    InverseGamma-distributed slab component (c2) capping how large an
    escaped amplitude can get. Because genuine inactivity is now the
    PRIOR's job, the null library row is dropped entirely — profile_k
    ranges over the n_years spike rows only (see
    _build_zero_sum_profile_library(..., include_null=False)); which row
    an amplitude~0 area's profile_k lands on is irrelevant, since it
    contributes ~nothing to z either way.

    p0: prior guess for how many of the n_areas areas are "genuinely
    active" (sets tau's scale via the standard regularised-horseshoe
    tau0 = p0/(n_areas-p0) formula) — NOT fit to the area/year taxonomy,
    just a reasonable default (n_areas // 4); revisit if
    pi_profile/amplitude diagnostics suggest it's off.
    slab_scale: the scale (in dwellings/year) an escaped amplitude is
    capped around — matches mu_log_sigma_prior's implied ~12
    dwellings/year used throughout this module's sigma_slab hierarchy, so
    this isn't introducing a new unrelated magnitude convention.

    wrap_z_as_deterministic: True (default, used by M15) registers z as a
    named pm.Deterministic, as every other builder in this module does.
    Set False (used by M16) to return z as a bare pytensor expression
    instead — REQUIRED for pymc_extras.marginalize() to be able to
    marginalise profile_k: marginalize() raises "Cannot marginalize
    profile_k due to dependent Deterministic z" if z is registered as a
    Deterministic (a marginalised RV can't have a downstream Deterministic
    depending on it, since a Deterministic needs one concrete value per
    draw and profile_k no longer has one once marginalised out). With
    wrap_z_as_deterministic=False, z never appears in the trace directly
    — callers sampling this way must reconstruct it afterwards from
    recovered profile_k + amplitude (see M16's sample() override).

    Also NOTE: whether pymc_extras.marginalize() can marginalise
    profile_k at all depends on THIS flag and on how the downstream
    likelihood is built. With wrap_z_as_deterministic=True (M15) and a
    pm.Potential-based likelihood, it cannot — see M15's docstring for
    the "No RVs depend on marginalized RV profile_k" finding. With
    wrap_z_as_deterministic=False AND a pm.CustomDist-based likelihood
    (_build_temporal_reallocation_likelihood_marginalizable, M16), it
    works, verified directly against this codebase's actual pattern
    (checked further: a *shared* z feeding two CustomDists, one per
    source, does NOT silently drop either one's contribution to the
    joint marginal logp — but pm.compute_log_likelihood() can't cleanly
    separate their POINTWISE breakdown afterward, a pymc_extras
    "non-separable logp" limitation when multiple observed nodes share
    one marginalised RV — see M16's docstring for how this is handled).

    Returns (mu_area, z, profile_k, amplitude, pi_profile, tau, lam, c2,
    profile_library).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled

    profile_library = _build_zero_sum_profile_library(n_years, include_null=False)
    K = profile_library.shape[0]  # = n_years, no null row
    profile_library_pt = pt.as_tensor_variable(profile_library)

    if p0 is None:
        p0 = max(1, n_areas // 4)
    tau0 = p0 / (n_areas - p0)

    tau = pm.HalfCauchy('tau_amplitude', beta=tau0)
    lam = pm.HalfCauchy('lam_amplitude', beta=1, dims='area')
    c2  = pm.InverseGamma('c2_amplitude', alpha=2, beta=slab_scale ** 2)
    lam_tilde = lam * pt.sqrt(c2 / (c2 + tau ** 2 * lam ** 2))
    amplitude = pm.Normal('amplitude', mu=0, sigma=tau * lam_tilde, dims='area')

    pi_profile = pm.Dirichlet('pi_profile', a=np.ones(K))
    profile_k  = pm.Categorical('profile_k', p=pi_profile, dims='area')

    z_expr = mu_area[:, None] + amplitude[:, None] * profile_library_pt[profile_k]
    if wrap_z_as_deterministic:
        z = pm.Deterministic('z', z_expr, dims=('area', 'year'))
    else:
        z = z_expr

    return (mu_area, z, profile_k, amplitude, pi_profile, tau, lam, c2, profile_library)


def _build_zero_sum_z_prior(D, n_areas, n_years, floor, k):
    """
    Anchored zero-sum z prior — the AZ family's replacement for both the
    soft census likelihood (_build_census_constraint, M9-M13) and the
    discrete profile-library mechanism (_build_z_prior_profile_library[
    _horseshoe], M14-M16).

        mu_area[a]     = D[a] / n_years                         (fixed)
        sigma_delta[a] = floor + k * |D[a]|                      (fixed)
        delta[a, :]   ~ ZeroSumNormal(sigma_delta[a])
        z[a, :]        = mu_area[a] + delta[a, :]

    z[a, :] sums to D[a] EXACTLY on every draw, by construction of
    pm.ZeroSumNormal (verified directly: residual from D is ~1e-14) —
    same exactness guarantee M14-M16 got from a hand-rolled profile
    library, but from a genuinely continuous distribution (no discrete
    profile_k latent, no CategoricalGibbsMetropolis/marginalize()
    machinery, fully nutpie-compatible).

    Concern raised before adopting this: does concentrating almost all of
    D[a] in a single year require some OTHER year to swing sharply
    negative to pay for it (which would bias the prior against exactly
    the concentrated-activity areas we most need to allow for)? Checked
    directly by simulation (n=10, sigma=5, 300k prior draws): every pair
    of years has Corr(delta_i, delta_j) = -1/(n_years-1) = -0.111 — weak.
    Conditional on one year taking value x, each of the other 9 shifts by
    only -x/9 on average (confirmed exactly against the -x/(n-1) theory
    across every tested bin), and keeps almost its full marginal spread
    doing so (std 4.59 vs marginal 4.74, for the other 9 years given a
    >2.5-sigma spike in the tenth). So no — a spike is paid for
    collectively, in small increments spread across the rest of the
    profile, not by manufacturing a single offsetting trough. A large
    concentrated deviation is well within the prior's mass.

    k controls how much per-year volatility is allowed PER UNIT of the
    area's own decade-long census change, on top of a shared floor.
    Checked against real data two ways: (1) empirically, only ~7-9% of
    areas (|D|>10, face-value (P+E)/2) need a single year to capture
    80-95%+ of D; (2) an unsigned prior-predictive check
    (max|single-year deviation|/|D|) shows k=0.08 already assigns MORE
    probability (26-33%) to that level of concentration than the
    fraction of real areas that actually need it — i.e. k=0.08 is
    comfortably generous, not too tight, despite being the smallest value
    tested (0.08/0.15/0.3). A synthetic recovery test (D=206, a
    genuinely concentrated true pattern, two independently-noisy-but-
    agreeing simulated sources) recovered well (RMSE 1.1-1.6) across
    that whole k range — the likelihood dominates once it's genuinely
    informative, so k mostly needs to be plausible, not precisely tuned.
    Treated here as a fixed hyperparameter (not sampled) — first-pass
    calibrated from the checks above, not yet validated against a real
    posterior fit; revisit if diagnostics show areas that need more (or
    less) room than floor + 0.08*|D| provides.

    floor sets the minimum per-year volatility even for D~0 areas (an
    area with near-zero net change can still have gains in some years
    offset by losses in others) — a first-pass value, same status as k.

    sigma_delta is NOT wrapped in pm.Deterministic (same convention as
    mu_area in _build_z_prior_hierarchical): it's a fixed function of
    data, not of any sampled RV, so recording it per-draw in the trace
    would only waste space.

    Returns (mu_area, sigma_delta, delta, z).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area     = D / n_years            # (n_areas,) numpy constant — not sampled
    sigma_delta = floor + k * np.abs(D)  # (n_areas,) numpy constant — not sampled

    delta = pm.ZeroSumNormal('delta', sigma=sigma_delta[:, None],
                             n_zerosum_axes=1, dims=('area', 'year'))
    z = pm.Deterministic('z', mu_area[:, None] + delta, dims=('area', 'year'))

    return mu_area, sigma_delta, delta, z


def _build_zero_sum_z_prior_top_boost(D, n_areas, n_years, floor, k, is_top,
                                      top_boost_sigma=40.0):
    """
    "Global formula + one extra top-tier boost" variant of
    _build_zero_sum_z_prior — AZ2's Option 2 (see
    docs/az-family-work-plan.md Phase 2), tried after the full 4-band
    hierarchy (_build_zero_sum_z_prior_banded) was shown to be the wrong
    level of complexity for what the data actually needed.

        sigma_delta[a] = floor + k*|D[a]| + is_top[a] * top_boost

    Keeps AZ0a's original, calibrated floor + k*|D| formula UNCHANGED and
    fully fixed (not sampled) for every area -- the bottom 75% of areas
    were never the problem, so they get the exact same well-behaved prior
    AZ0a already validated (max r-hat 1.006, min ESS 3180). One extra
    sampled scalar, top_boost, ADDS flexibility only for areas flagged
    is_top, rather than trying to fit a separate hierarchical scale for
    every magnitude tier.

    Directly evidenced, not a guess: the 4-band hierarchy's own posterior
    showed only the top quartile's band picked up any real "excess" (~62,
    vs ~0 for the bottom 3 bands, which collapsed to an undifferentiated
    floor-only fit and actually made those areas WORSE than AZ0a's
    baseline -- see docs/az-family-work-plan.md for the full investigation,
    including the per-band frac_flat_despite_active breakdown that pinned
    this down: 45.7%/43.4%/27.5%/0.0% for bands 0-3). This construction
    is the direct, minimal-risk consequence of that finding: give ONLY the
    tier that needed it the extra flexibility, leave everything else at
    the already-validated baseline.

    top_boost_sigma=40 is a weakly-informative HalfNormal scale chosen to
    comfortably cover the ~62 magnitude the 4-band version's top-tier
    excess converged to (E[HalfNormal(40)]~=32, with plenty of prior mass
    out past 100) without being so wide it's effectively flat.

    is_top: fixed (n_areas,) boolean/0-1 array flagging the top D-magnitude
    tier (e.g. top quartile of |D| -- see AZ2.build()).

    Returns (mu_area, sigma_delta, top_boost, delta, z).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled
    is_top_f = is_top.astype('float64')

    top_boost = pm.HalfNormal('sigma_delta_top_boost', sigma=top_boost_sigma)

    sigma_delta = pm.Deterministic(
        'sigma_delta', floor + k * np.abs(D) + is_top_f * top_boost, dims='area')

    delta = pm.ZeroSumNormal('delta', sigma=sigma_delta[:, None],
                             n_zerosum_axes=1, dims=('area', 'year'))
    z = pm.Deterministic('z', mu_area[:, None] + delta, dims=('area', 'year'))

    return mu_area, sigma_delta, top_boost, delta, z


def _build_zero_sum_z_prior_top_boost_smooth(D, n_areas, n_years, floor, k,
                                             top_quantile=0.75,
                                             transition_width=0.08,
                                             top_boost_sigma=40.0):
    """
    _build_zero_sum_z_prior_top_boost with the binary is_top cutoff
    replaced by a smooth sigmoid ramp over |D|'s rank percentile — AZ2b's
    fix attempt for AZ2's residual low bulk ESS on `sigma_delta_top_boost`
    (docs/az-ess-diagnosis.md).

    AZ2's diagnose-table min ESS (bulk=47) was traced to a small,
    persistent BETWEEN-chain disagreement on sigma_delta_top_boost/
    sigma_plan/sigma_ben (chain means stable but not converging toward
    each other, e.g. top_boost's 4 chain means sitting at 28.5-30.3 for
    the whole run) — ruled out ordinary slow mixing (autocorrelation near
    zero at lag>20) and ruled out a simple 2-3-variable ridge among the
    three named scalars (all pairwise |r| < 0.12, even at 4000 draws).
    Untested candidate: `is_top` is a hard 0/1 step exactly at the 75th
    percentile of |D| — areas straddling that boundary see a discontinuous
    jump in how sigma_delta responds to top_boost, which is exactly the
    kind of sharp threshold that can produce a shallow, hard-to-mix ridge
    in the joint posterior (a few areas near the cutoff effectively vote
    on a knife-edge). This replaces the step with a smooth logistic ramp
    over the RANK percentile of |D| (not |D| itself, to stay scale-free
    across the wide 10-600+ range this dataset spans):

        rank_pct[a]     = (rank of |D[a]| among all areas + 0.5) / n_areas
        smooth_weight[a] = sigmoid((rank_pct[a] - top_quantile) / transition_width)
        sigma_delta[a]   = floor + k*|D[a]| + smooth_weight[a] * top_boost

    transition_width=0.08 (about 16 areas' worth of rank, for n_areas=200)
    keeps the ramp fairly sharp — areas well inside/outside the top
    quartile still get ~0/~1 weight, only the boundary ~15-20 areas see a
    meaningfully intermediate weight — deliberately a SMOOTHING fix for
    the discontinuity, not a redesign of which areas get the boost.
    smooth_weight[a] is a fixed numpy array (a function of the fixed,
    unsampled |D| ranks), not itself a sampled quantity, so this doesn't
    add any new degrees of freedom versus the original — same one extra
    scalar (top_boost) as before.

    Returns (mu_area, sigma_delta, top_boost, delta, z, smooth_weight) —
    smooth_weight included (unlike the binary version's is_top, which the
    caller already has) since it's newly computed here and useful for
    diagnostics (e.g. plotting sigma_delta vs rank to confirm the ramp
    looks as intended).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled
    abs_D = np.abs(D)
    order = np.argsort(np.argsort(abs_D))
    rank_pct = (order + 0.5) / n_areas
    smooth_weight = 1.0 / (1.0 + np.exp(-(rank_pct - top_quantile) / transition_width))

    top_boost = pm.HalfNormal('sigma_delta_top_boost', sigma=top_boost_sigma)

    sigma_delta = pm.Deterministic(
        'sigma_delta', floor + k * abs_D + smooth_weight * top_boost, dims='area')

    delta = pm.ZeroSumNormal('delta', sigma=sigma_delta[:, None],
                             n_zerosum_axes=1, dims=('area', 'year'))
    z = pm.Deterministic('z', mu_area[:, None] + delta, dims=('area', 'year'))

    return mu_area, sigma_delta, top_boost, delta, z, smooth_weight


def _build_zero_sum_z_prior_banded(D, n_areas, n_years, band_idx, n_bands,
                                   floor=3.0,
                                   mu_log_sigma_prior=(np.log(6), 1.0),
                                   tau_log_sigma_prior=1.0):
    """
    D-magnitude-band hierarchical variant of _build_zero_sum_z_prior:
    replaces the fixed floor + k*|D| formula's k*|D| term with a per-band
    hierarchical scale (floor stays fixed and shared), reusing the SAME
    non-centred log-normal construction _build_z_prior_hierarchical_borough
    already validated for per-borough sigma_slab (M9/M10).

    Motivation (see docs/az-family-work-plan.md Phase 2): the fixed
    floor + k*|D| formula forces ONE global k to describe every area's
    deviation-scale-vs-magnitude relationship. A weak but real shrinkage
    correlation (spike-size-relative-to-sigma_delta vs. shrinkage amount,
    r=-0.21, found investigating AZ0a's under-tracked spikes) suggested
    some genuinely concentrated large-D areas may be prior-starved under
    the single global relationship. Banding by |D| magnitude lets each
    tier's typical scale be fit from its own data instead of one formula
    extrapolated across the whole 10-600+ range.

    floor is DELIBERATELY still a fixed, shared additive term -- NOT also
    banded/hierarchical. First version of this dropped it entirely (a pure
    per-band multiplier, sigma_delta = sigma_delta_band[band_idx]) and hit
    a severe, real regression on real data: sigma_delta_band collapsed to
    ~0.2-0.8 for the bottom 3 of 4 bands (150/200 areas), flattening z
    almost everywhere regardless of real P/E signal (frac_flat_despite_
    active jumped from AZ0a's 11.5% to 70%). Root cause: bands are defined
    by |D| (NET census change), which is a poor proxy for how much
    year-to-year VOLATILITY an area needs -- an area with +100 one year
    and -95 the next has tiny net D but needs large sigma_delta, and vice
    versa, so binning purely on |D| puts some genuinely volatile areas in
    low-D bands and lets the hierarchy shrink that band's scale toward
    whatever fits the *median* (usually quiet) area there. This closely
    mirrors an already-documented failure in this codebase (M9's per-area
    sigma_slab hierarchy collapsing to a near-constant ~0.16 across all
    200 areas) -- same shrinkage-toward-the-majority pathology, different
    parameter. Reinstating floor as a shared, non-collapsible minimum
    (matching the original formula's role) is the direct, minimal fix:
    every area retains baseline flexibility regardless of which band's
    hierarchical component collapses, while the per-band term still adds
    band-specific EXCESS flexibility on top where the data supports it.

    tau_log_sigma_prior is wider than M9/M10's borough version (1.0 vs
    0.7) deliberately: D-magnitude bands are constructed specifically to
    differ (that's the whole point of banding), unlike boroughs, which
    aren't expected to differ as strongly a priori.

    band_idx: fixed (n_areas,) int array assigning each area to a
    D-magnitude band (e.g. quartiles of |D| -- see AZ2.build()).

    Returns (mu_area, sigma_delta, sigma_delta_band, delta, z).
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    mu_area = D / n_years  # (n_areas,) numpy constant — not sampled

    mu_log_sigma  = pm.Normal('mu_log_sigma_delta', mu=mu_log_sigma_prior[0],
                              sigma=mu_log_sigma_prior[1])
    tau_log_sigma = pm.HalfNormal('tau_log_sigma_delta', sigma=tau_log_sigma_prior)
    log_sigma_offset = pm.Normal('log_sigma_delta_offset', mu=0, sigma=1, shape=n_bands)

    sigma_delta_band = pm.Deterministic(
        'sigma_delta_band',
        pt.exp(mu_log_sigma + tau_log_sigma * log_sigma_offset))  # (n_bands,) — excess over floor

    sigma_delta = pm.Deterministic(
        'sigma_delta', floor + sigma_delta_band[band_idx], dims='area')  # (n_areas,)

    delta = pm.ZeroSumNormal('delta', sigma=sigma_delta[:, None],
                             n_zerosum_axes=1, dims=('area', 'year'))
    z = pm.Deterministic('z', mu_area[:, None] + delta, dims=('area', 'year'))

    return mu_area, sigma_delta, sigma_delta_band, delta, z


def _build_capture_rate(n_areas, name, sigma_kappa):
    """
    Per-area multiplicative capture-rate (recording-completeness) scale.
    kappa[a] = exp(sigma_kappa * log_kappa_offset[a]) — non-centred
    log-normal; kappa[a] = 1 (no bias) at offset 0. sigma_kappa is a
    SHARED hyperparameter built once by the caller and passed in (see
    M10's docstring for why P/E share one scale rather than independent
    ones).

    name: 'P' or 'E' — used to build f'log_kappa_offset_{name}' /
    f'kappa_{name}' so this can be called twice in the same model without
    a PyMC variable-name collision.

    Returns kappa, shape (n_areas,), dims='area'.
    Must be called inside a pm.Model() context with an 'area' coord, after
    sigma_kappa has been constructed by the caller.
    """
    log_kappa_offset = pm.Normal(f'log_kappa_offset_{name}', mu=0, sigma=1, dims='area')
    kappa = pm.Deterministic(
        f'kappa_{name}', pt.exp(sigma_kappa * log_kappa_offset), dims='area')
    return kappa


def _build_census_constraint(z, D, sigma_census):
    """
    Add census constraint likelihood.
    Must be called inside a pm.Model() context.
    """
    pm.Normal('census_obs', mu=z.sum(axis=1),
              sigma=sigma_census, observed=D)


def _build_pre_inference(data, max_lag, source='P'):
    """
    Build fixed array of pre-inference observations to use as proxies for
    source years before the inference window.

    source: 'P' (planning, default — preserves existing behaviour for all
            callers that don't pass source) or 'E' (BEN). Both column
            families (ALL_COLS_PLAN/ALL_COLS_BEN) span 2009-2024 with the
            inference window starting 2011, so both sources have only 2
            real pre-window years before this pads with the earliest
            available column — a real ceiling on how much lag correction
            is achievable for either source.

    Returns numpy array of shape (n_areas, max_lag).
    """
    if source == 'P':
        obs_full, all_cols, infer_cols = (
            data['P_obs_full'], ALL_COLS_PLAN, INFER_COLS_PLAN)
    elif source == 'E':
        obs_full, all_cols, infer_cols = (
            data['E_obs_full'], ALL_COLS_BEN, INFER_COLS_BEN)
    else:
        raise ValueError(f"source must be 'P' or 'E', got {source!r}")

    infer_start = all_cols.index(infer_cols[0])
    return np.stack([
        obs_full[:, infer_start - max_lag + k]
        if (infer_start - max_lag + k) >= 0
        else obs_full[:, 0]
        for k in range(max_lag)
    ], axis=1).astype('float64')


def _build_lag(z, pre_inference, n_areas, n_years, n_lags, alpha, max_lag,
              lambda_weights=None, name='lambda_weights'):
    """
    Build temporal lag structure for planning data.

    name: PyMC variable name for the sampled Dirichlet — override to e.g.
    'lambda_weights_P'/'lambda_weights_E' when calling this twice in the
    same model (P and E each need their own lag distribution) to avoid a
    PyMC variable-name collision. Ignored when lambda_weights is fixed.

    Returns (lambda_weights, P_mean).
    Must be called inside a pm.Model() context.
    """
    if lambda_weights is None:
        lambda_weights = pm.Dirichlet(name, a=alpha)
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


def _build_hierarchical_lag(z, pre_inference, n_areas, n_years, n_lags,
                            max_lag, prior_logit, tau_sigma=1.5, name='lag'):
    """
    Per-area hierarchically-pooled temporal lag structure -- the area-level
    alternative to _build_lag's single shared Dirichlet kernel.

    _build_lag forces EVERY area to share one lag-weight vector. AZ1a
    (which used two calls to _build_lag, one per source) showed this is a
    real problem, not a theoretical one: lambda_weights_E's posterior was
    genuinely bimodal across chains (two chains found "E is same-year",
    two found "E lags by 1 year") -- a single shared kernel has no one
    right answer when different areas' true lag patterns genuinely differ.
    Borough-level pooling was considered and rejected before building it:
    no strong mechanistic reason a site/development-level completion-to
    -registration lag would align with administrative borough boundaries.

    This builds a genuine per-area kernel instead, but NOT independently
    per area (that would repeat M12's failure -- ~10 obs/area/source can't
    identify 200 independent 3-parameter kernels). Areas are hierarchically
    shrunk toward a shared population kernel via a softmax-over-logits
    construction, non-centered for the usual funnel-avoidance reasons:

        mu_logit[k]      ~ Normal(prior_logit[k], 1)      k = 1..n_lags-1
        tau[k]           ~ HalfNormal(tau_sigma)          k = 1..n_lags-1
        raw_offset[a,k]  ~ Normal(0, 1)
        area_logit[a,k]  = mu_logit[k] + raw_offset[a,k] * tau[k]
        full_logit[a,:]  = concat([0, area_logit[a,:]])   # k=0 (same-year)
                                                            # fixed as the
                                                            # softmax reference
                                                            # category
        lambda_weights[a,:] = softmax(full_logit[a,:])

    tau -> 0 recovers _build_lag's fully-pooled answer exactly (every area
    converges to the same softmax(mu_logit)); tau large lets an area's own
    data pull its kernel away from the population one. This is regularised
    partial pooling, not literal per-area independence -- weakly-informed
    (quiet) areas shrink hard toward the shared kernel, only areas with
    real signal can justify diverging from it.

    prior_logit: array of length n_lags-1, the prior mean for mu_logit in
    logit space -- pass log(lag_alpha[1:] / lag_alpha[0]) to centre this on
    the same "prefer short lags" bias _build_lag's Dirichlet(lag_alpha)
    prior encoded, so switching from AZ1a to this isn't also silently
    changing the prior's central tendency.

    name: prefix for PyMC variable names (e.g. 'lag_P'/'lag_E' to avoid
    collisions when called twice in the same model for P and E).

    Returns (lambda_weights, mean) where lambda_weights has shape
    (n_areas, n_lags) and mean has shape (n_areas, n_years).
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    tau = pm.HalfNormal(f'{name}_tau', sigma=tau_sigma, shape=n_free)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau[None, :]

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_hierarchical_lag_capped(z, pre_inference, n_areas, n_years, n_lags,
                                   max_lag, prior_logit, tau_cap=1.5, name='lag'):
    """
    _build_hierarchical_lag with a HARD ceiling on tau instead of a free
    HalfNormal(tau_sigma) — AZ1b's untried fix option 1 (docs/az-family-
    work-plan.md Phase 1b "Status: open" list), tried properly instead of
    left untested. AZ1c uses this.

    AZ1b's residual max r-hat/min ESS were diagnosed as GENUINE hard
    multimodality for ~10-15% of areas (individual chains spending all
    1500 draws in one of two disconnected modes and never crossing over) —
    not a geometry/tuning problem. Tightening tau's PRIOR (tau_sigma
    1.5 -> 0.5) was tried and made things WORSE (max r-hat 1.12 -> 1.24,
    min ESS 23 -> 12), because the posterior tau barely moved under the
    tighter prior (e.g. lag_E_tau ~2.7-3.2 vs ~5.1-5.9) — the likelihood's
    pull toward per-area divergence is strong enough to mostly override a
    3x tighter prior. A softer prior nudge doesn't reach the mechanism.

    This is mechanically different, not just a smaller sigma on the same
    free HalfNormal: tau is reparameterised as a FIXED ceiling times a
    Beta(2,2)-distributed fraction, `tau = tau_cap * tau_frac`, so the
    actual per-area divergence effect (not just its prior) is hard-capped
    at tau_cap regardless of what the likelihood wants — the thing that
    made the prior-tightening attempt fail (posterior tau resisting the
    prior) cannot happen here, since tau_frac is bounded to [0, 1] by
    construction (Beta support), not merely encouraged toward 0.

    tau_cap=1.5 (AZ1c's default) is a genuine test, not a guess free of
    consequence: AZ1b's converged tau was 2.6-5.9 (P) / 5.1-5.9 (E)
    unconstrained, and even the failed tighter-PRIOR attempt still let
    tau reach ~2.7-3.2 in practice — 1.5 is meaningfully below both,
    intended to test whether a hard (not just encouraged) ceiling well
    below the areas' apparent preference suppresses the multimodality,
    at a documented, expected cost to per-area flexibility (specifically
    AZ1b's E01033711 multi-spike win, which depended on large per-area
    divergence from the population kernel).

    Returns (lambda_weights, mean), same shapes as _build_hierarchical_lag.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    tau_frac = pm.Beta(f'{name}_tau_frac', alpha=2, beta=2, shape=n_free)
    tau = pm.Deterministic(f'{name}_tau', tau_cap * tau_frac)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau[None, :]

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_hierarchical_lag_pinned(z, pre_inference, n_areas, n_years, n_lags,
                                   max_lag, prior_logit, pinned_mask, pinned_logit,
                                   tau_sigma=1.5, name='lag'):
    """
    _build_hierarchical_lag with a caller-supplied SUBSET of areas' area_logit
    fixed at a constant instead of sampled — a diagnostic ablation, not a new
    modelling choice, built to test a specific causal claim from
    docs/az-ess-diagnosis.md: that AZ1d's/AZ4's worst-converged POPULATION-level
    scalars (`lag_*_mu_logit`, `lag_*_tau`, and in AZ4's case `sigma_noise_E`/
    `sigma_delta_top_boost` downstream of them) are a diffuse consequence of the
    hierarchy having to reconcile the well-behaved majority against a collective
    pull from a small lag-ambiguous minority of areas — evidenced so far only by
    a weak per-chain correlation check (r=-0.01/+0.26), not a direct intervention.

    Unlike dropping the flagged areas from the dataset entirely (which also
    shrinks n_areas and changes what mu_logit/tau are estimated FROM), this
    keeps every area's P/E observations in the likelihood and in `z`'s zero-sum
    prior — only the flagged areas' OWN lag-category posterior is prevented
    from feeding back into mu_logit/tau. If population-scalar r-hat/ESS
    recovers under this SHARPER intervention, that is real support for the
    "these areas' own unresolved posterior is the channel" story; if it
    doesn't, the mechanism is something else (e.g. ordinary small-tau funnel
    geometry, unrelated to which specific areas are ambiguous).

    `raw_offset` is still sampled for every area (including pinned ones) to
    keep the model's shape simple — for pinned rows it only ever appears in
    its own Normal(0,1) prior term, since `pinned_mask` zeroes it out of
    `area_logit` before that reaches mu_logit/tau or the likelihood, so it
    costs nothing beyond a few inert extra dimensions.

    pinned_mask  : bool array, shape (n_areas,) — True where area_logit is
                   fixed at `pinned_logit` instead of `mu_logit + raw_offset*tau`.
    pinned_logit : float array, shape (n_areas, n_lags-1) — value used for
                   pinned areas (ignored elsewhere); typically each pinned
                   area's own posterior-mean area_logit from an existing trace,
                   so this ablation asks "what if that area's ALREADY-FOUND
                   answer were known in advance" rather than "what if that
                   area had no lag at all."

    Returns (lambda_weights, mean), same shapes as _build_hierarchical_lag.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    tau = pm.HalfNormal(f'{name}_tau', sigma=tau_sigma, shape=n_free)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    sampled_logit = mu_logit[None, :] + raw_offset * tau[None, :]

    mask = pt.as_tensor_variable(pinned_mask.astype(float))[:, None]
    pinned = pt.as_tensor_variable(pinned_logit.astype(float))
    area_logit = mask * pinned + (1 - mask) * sampled_logit

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_fixed_lag(z, pre_inference, n_areas, n_years, n_lags, max_lag,
                     fixed_logit, name='lag'):
    """
    A lag convolution with NO sampled parameters at all — `fixed_logit`
    (length n_lags-1) sets one constant lambda_weights vector, shared
    across every area, via the same softmax(concat([0, logit])) convention
    _build_hierarchical_lag uses. Not a modelling choice in its own right;
    built for the AZ4 ablation in docs/az-ess-diagnosis.md that tests
    whether it's specifically E's lag-category SAMPLING uncertainty (mu_logit/
    tau/raw_offset all still free RVs, degrees of freedom that must be
    reconciled against 200 areas' worth of pull) — as opposed to any one
    area's own data — that leaks into sigma_noise_E/sigma_delta_top_boost
    when combined with AZ3's noise mixture. Passing `fixed_logit` as the
    POSTERIOR MEAN from an already-sampled AZ4/AZ1b-family trace keeps this
    a like-for-like comparison (same central lag behaviour) with all
    per-area/per-draw uncertainty in that mechanism removed, rather than
    resetting to an arbitrary or uninformative lag.

    Returns (lambda_weights, mean): lambda_weights has shape (n_lags,)
    (not (n_areas, n_lags) — genuinely one shared constant, not a
    per-area Deterministic), mean has shape (n_areas, n_years).
    Must be called inside a pm.Model() context.
    """
    full_logit = np.concatenate([[0.0], np.asarray(fixed_logit)])
    lambda_weights = pt.special.softmax(pt.as_tensor_variable(full_logit))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[None, None, :], axis=2)
    return lambda_weights, mean


def _build_hierarchical_lag_horseshoe(z, pre_inference, n_areas, n_years, n_lags,
                                      max_lag, prior_logit, global_tau_sigma=1.5,
                                      local_scale_beta=1.0, name='lag'):
    """
    Local/global (horseshoe-style) hierarchical lag — replaces
    _build_hierarchical_lag's ONE shared `tau` (identical divergence
    allowance for all 200 areas) with a per-area `tau[a,k]`, decomposed
    as a shared global scale times a heavy-tailed per-area local
    multiplier — AZ1e's fix attempt for AZ1d's residual P-lag ambiguity
    (docs/ess-rhat-diagnostic-guide.md Pattern 3 / the "outside the box"
    proposal 2 in that round's discussion), tried instead of assumed to
    help, given every UNIFORM tau constraint tried so far (AZ1c's hard
    cap, AZ4b's hard cap in the combined model) made things WORSE, not
    better, by fighting genuinely-needed per-area divergence for a
    minority while giving nothing back to the majority.

    Mechanistically different from AZ1c's cap in exactly the way that
    matters: AZ1c/AZ4b constrained EVERY area's tau identically (a global
    ceiling), which cannot help since the areas needing large divergence
    and the areas that don't are lumped under one shared value regardless.
    This lets tau vary per area:

        mu_logit[k]      ~ Normal(prior_logit[k], 1)
        global_tau[k]    ~ HalfNormal(global_tau_sigma)     -- population scale
        local_scale[a,k] ~ HalfCauchy(local_scale_beta)     -- per-area multiplier,
                                                                heavy-tailed so MOST
                                                                areas' local_scale
                                                                stays small (shrinking
                                                                tighter than AZ1d's
                                                                uniform tau ever did)
                                                                while a few areas can
                                                                have local_scale >> 1
                                                                without moving
                                                                global_tau at all
        tau[a,k]         = global_tau[k] * local_scale[a,k]
        raw_offset[a,k]  ~ Normal(0, 1)
        area_logit[a,k]  = mu_logit[k] + raw_offset[a,k] * tau[a,k]

    Documented, known risk carried in from the general horseshoe
    literature (see the pymc-extras skill and priors.md): HalfCauchy's
    heavy tails create their own funnel-like geometry between
    global_tau and local_scale, distinct from (and potentially just as
    bad as) the problem this is meant to fix — this is NOT assumed to be
    a clean win, it's a genuinely different mechanism worth testing on
    its own terms. Unregularized (no slab scale) on this first pass,
    deliberately — see docs/ess-rhat-diagnostic-guide.md before adding
    that complexity if plain HalfCauchy's tails prove to be the actual
    problem rather than the fix.

    Returns (lambda_weights, mean), same shapes as _build_hierarchical_lag.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    global_tau = pm.HalfNormal(f'{name}_global_tau', sigma=global_tau_sigma,
                               shape=n_free)
    local_scale = pm.HalfCauchy(f'{name}_local_scale', beta=local_scale_beta,
                                shape=(n_areas, n_free))
    tau = pm.Deterministic(f'{name}_tau', global_tau[None, :] * local_scale)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_hierarchical_lag_regularized_horseshoe(z, pre_inference, n_areas, n_years,
                                                   n_lags, max_lag, prior_logit,
                                                   global_tau_sigma=1.5, local_scale_beta=1.0,
                                                   slab_scale=10.0, name='lag'):
    """
    _build_hierarchical_lag_horseshoe + a SLAB that caps the local multiplier's effective
    reach -- the "regularized horseshoe" (Piironen & Vehtari 2017), tried because AZ1e (the
    plain/unregularized version) failed for a specific, diagnosed reason: `local_scale`
    (HalfCauchy(1), unbounded) explored values up to 3.09 MILLION, which is a geometry
    problem for NUTS (scattered divergences, 17 total) regardless of whether the underlying
    idea -- letting a few areas diverge without moving the shared tau -- is sound. AZ1e's own
    docstring flagged this as untested but expected-necessary before concluding the local/
    global approach is dead; this is that follow-through, not a new idea.

    This session's own direct check (results/scratch/az1d_leakage_mechanism.py, both the
    original Islington 200-area sample AND a fresh Croydon-centred resample) found the
    mechanism this is meant to fix is real and strong, not just plausible: a chain's own
    lag_P_tau draw correlates r=0.85-0.98 (Islington) / 0.42-0.87 (Croydon) with how far that
    SAME chain's flagged-area logits sit from their own mean -- i.e. tau's between-chain
    disagreement IS mechanically the disagreement about which mode the ~6% lag-ambiguous
    minority landed in, on both samples tested. Those flagged areas are not a random subset:
    median |D| is 7.6-8.6x the dataset median and median max|P_obs| is ~12x -- exactly the
    large-spike, stakeholder-critical areas this model family exists to serve, so simply
    dropping or dampening their flexibility (AZ1c's/AZ4b's uniform tau cap, both already
    rejected) costs real, wanted behaviour.

    Reparameterises AZ1e's local_scale as a smooth interpolation between "shrunk toward 0"
    (ordinary areas) and "capped at slab_scale, not unbounded" (the minority that needs room):

        mu_logit[k]        ~ Normal(prior_logit[k], 1)
        global_tau[k]      ~ HalfNormal(global_tau_sigma)
        local_lambda[a,k]  ~ HalfCauchy(local_scale_beta)      -- same heavy tail as AZ1e,
                                                                    deliberately UNCHANGED so
                                                                    any ESS difference isolates
                                                                    the slab's effect specifically
        local_lambda_tilde[a,k] = sqrt(
            slab_scale^2 * local_lambda[a,k]^2
            / (slab_scale^2 + global_tau[k]^2 * local_lambda[a,k]^2)
        )
        tau[a,k]           = global_tau[k] * local_lambda_tilde[a,k]
        raw_offset[a,k]    ~ Normal(0, 1)
        area_logit[a,k]    = mu_logit[k] + raw_offset[a,k] * tau[a,k]

    As local_lambda[a,k] -> 0, local_lambda_tilde -> 0 (same shrinkage-toward-mu_logit
    behaviour as AZ1e for the well-behaved majority). As local_lambda[a,k] -> infinity,
    local_lambda_tilde -> slab_scale / global_tau[k], so the offset saturates at
    `raw_offset[a,k] * slab_scale` -- bounded by construction REGARDLESS of how large
    local_lambda's own draw gets, unlike AZ1e where the offset scaled linearly with an
    unbounded local_scale all the way to the 3-million-fold blowup. This directly targets
    the diagnosed failure (the geometry of exploring local_lambda's own extreme tail) without
    changing the intended asymmetric-shrinkage behaviour local_lambda was chosen for.

    slab_scale=10.0 (default): calibrated against AZ1d's own baseline trace, where the most
    extreme observed area_logit draws (the flagged minority's competing-mode commitments)
    reach magnitude ~30 (99.9th percentile ~15) -- since the offset saturates at
    `raw_offset * slab_scale` and raw_offset routinely reaches 3+ in its own tail, 10.0 leaves
    enough room for a flagged area to still commit strongly to one lag category (matching
    what AZ1d's actually-observed range required) while remaining a fixed, finite ceiling, not
    an encouragement-only prior scale (the exact class of fix -- softer priors, tighter
    tau_sigma -- that already failed for AZ1b/AZ1c by not reaching the mechanism).

    Returns (lambda_weights, mean), same shapes as _build_hierarchical_lag.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    global_tau = pm.HalfNormal(f'{name}_global_tau', sigma=global_tau_sigma,
                               shape=n_free)
    local_lambda = pm.HalfCauchy(f'{name}_local_lambda', beta=local_scale_beta,
                                 shape=(n_areas, n_free))
    local_lambda_tilde = (slab_scale * local_lambda) / pt.sqrt(
        slab_scale**2 + (global_tau[None, :] * local_lambda)**2)
    tau = pm.Deterministic(f'{name}_tau', global_tau[None, :] * local_lambda_tilde)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_hierarchical_lag_regularized_horseshoe_v2(z, pre_inference, n_areas, n_years,
                                                      n_lags, max_lag, prior_logit,
                                                      p0, local_scale_beta=1.0,
                                                      slab_c2_beta=100.0, name='lag'):
    """
    `_build_hierarchical_lag_regularized_horseshoe` moved onto the CANONICAL Piironen &
    Vehtari (2017) regularized-horseshoe recipe exactly as given in this codebase's own
    `pymc-extras` skill reference (`references/r2d2_horseshoe.md`), rather than the earlier
    version's two hand-picked deviations from it: a FIXED `slab_scale` constant and a
    `global_tau` prior reused from AZ1d's unrelated `tau_sigma` rather than derived from this
    model's own expected sparsity. AZ1g's own check-multimodality follow-up found its residual
    bad-r-hat scalars (`global_tau`, `mu_logit`, and via them `sigma_ben`) are `not_multimodal`
    -- a mild "shallow basin" (clean autocorrelation, smooth per-chain rank-histogram tilt, no
    discrete cluster split) -- on BOTH the Islington and Croydon 200-area samples tested. Per
    `docs/ess-rhat-diagnostic-guide.md`'s own decision procedure (step 4: for a shared/
    population-level parameter, check whether a smooth function could still be doing more
    principled work than the current one), this targets that residual with the standard
    fix for exactly this geometry, not a novel one.

    Two concrete corrections vs `_build_hierarchical_lag_regularized_horseshoe`:

    1. **The slab scale is SAMPLED, not fixed.** The reference recipe puts a weakly-
       informative prior on the slab variance, `c2 ~ InverseGamma(2, 1)`, rather than treating
       it as a hand-tuned ceiling -- AZ1g's `slab_scale=10.0` was calibrated by eyeballing
       AZ1d's own observed area_logit range, the same class of fixed, hand-picked constraint
       that already failed twice elsewhere in this family (AZ1c's/AZ4b's tau cap). Here,
       `c2 ~ InverseGamma(alpha=2, beta=slab_c2_beta)` lets the data determine how much room
       the genuinely-divergent minority needs. `slab_c2_beta` is NOT the textbook sparse-
       regression default (`beta=1`, implying a slab std ~1, appropriate for standardized
       regression coefficients) -- it's set so the InverseGamma's prior MEAN
       (`beta/(alpha-1) = slab_c2_beta`) matches AZ1g's own empirically-useful slab_scale^2
       (10^2 = 100), since this model's logit space isn't unit-scaled the way a
       standardized-predictor regression is. Kept as a genuine PRIOR (the posterior can move
       away from 100 if the data wants to), not silently smuggling the old fixed value back in
       as a disguised default.

    2. **`global_tau`'s prior is derived from this model's own expected sparsity, via the
       reference recipe's own `tau0 = p0 / (p - p0) / sqrt(n)` formula**, rather than reusing
       AZ1d's `tau_sigma=1.5` (a value calibrated for a DIFFERENT model's un-pooled, not
       horseshoe-shrunk, per-area tau). Here `p = n_areas` (200, the total group count,
       playing the same role as "number of predictors" in the reference formula), `p0`
       (caller-supplied) is the expected count of areas needing genuine divergence -- this
       session's own check-multimodality runs flagged 12/200 (Islington) and 13-15/200
       (Croydon), so `p0=15` (AZ1h's default) is a documented, slightly-conservative estimate
       spanning both, not a guess -- and `n = n_years` (this model's closest analogue of "the
       number of observations informing each group's own effect," since each area's own P
       lag-category signal comes from its own ~n_years planning observations, not the
       dataset's total 200*n_years cell count). `global_tau ~ HalfCauchy(tau0)` (matching the
       reference's own Cauchy-tailed global scale, not AZ1e's/AZ1g's `HalfNormal`) — now that
       the slab is sampled rather than fixed, the classic double-Cauchy horseshoe geometry the
       slab exists to regularize is exactly the geometry actually present, so using
       `HalfNormal` for the global scale (as AZ1e/AZ1g both did, presumably to avoid stacking
       two heavy tails) no longer has a clear justification.

    Returns (lambda_weights, mean), same shapes as _build_hierarchical_lag.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1
    # p0 is calibrated for the real 200-area dataset (12-15 flagged areas there) -- clamp so
    # the formula stays well-defined (denominator positive, tau0 finite) on much smaller
    # datasets too, e.g. this codebase's 9-area synthetic test fixture. Never a live concern
    # for the real data (p0=15 << n_areas=200), only a safety guard for tests/toy runs.
    p0_eff = min(p0, max(1, n_areas // 2))
    tau0 = p0_eff / (n_areas - p0_eff) / np.sqrt(n_years)

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    global_tau = pm.HalfCauchy(f'{name}_global_tau', beta=tau0, shape=n_free)
    local_lambda = pm.HalfCauchy(f'{name}_local_lambda', beta=local_scale_beta,
                                 shape=(n_areas, n_free))
    c2 = pm.InverseGamma(f'{name}_c2', alpha=2, beta=slab_c2_beta, shape=n_free)
    local_lambda_tilde = local_lambda * pt.sqrt(
        c2[None, :] / (c2[None, :] + (global_tau[None, :] * local_lambda)**2))
    tau = pm.Deterministic(f'{name}_tau', global_tau[None, :] * local_lambda_tilde)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    mean = pt.sum(shifted * lambda_weights[:, None, :], axis=2)
    return lambda_weights, mean


def _build_planning_likelihood_simple(P_mean, P_obs, nu_obs, sigma_obs,
                                      name='P_like'):
    """
    M1 planning likelihood — StudentT, no missingness.

    name: PyMC variable name — override to 'E_like' to reuse this for a
    lagged BEN likelihood (M9+) without duplicating the function.

    Must be called inside a pm.Model() context.
    """
    pm.StudentT(name, nu=nu_obs, mu=P_mean,
                sigma=sigma_obs, observed=P_obs)


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


def _build_hierarchical_lag_marginalized(z, pre_inference, n_areas, n_years, n_lags,
                                         max_lag, prior_logit, tau_sigma=1.5, name='lag'):
    """
    Identical hierarchical lag-category structure to _build_hierarchical_lag
    (same mu_logit/tau/raw_offset/lambda_weights construction, same prior),
    but returns the per-category SHIFTED z tensor instead of pre-mixing it
    into a single blended mean -- for use with
    _build_planning_likelihood_marginalized_lag instead of
    _build_planning_likelihood_simple.

    AZ1d's own deep-dive (docs/az-ess-diagnosis.md) found genuine,
    irreducible multimodality in lambda_weights for a handful of areas
    (e.g. E01002702, E01035649, E01035646, E01035708) where the data ties
    two or three lag categories. _build_hierarchical_lag's mean-mixing
    construction (P_mean = sum_k lambda_k * shifted_k, one StudentT around
    that blend) forces every observation to be scored against a single
    blended value that no individual lag category actually predicts --
    a known anti-pattern for representing genuine discrete-category
    uncertainty, and a plausible (not yet confirmed on this model) source
    of the disconnected-mode geometry that degrades lag_P_mu_logit/tau's
    r-hat even under more chains (checked directly: AZ1d's 16-chain
    informed-init run still left lag_P_mu_logit at r-hat 1.02-1.03).

    Returns (lambda_weights, shifted) where lambda_weights has shape
    (n_areas, n_lags) and shifted has shape (n_areas, n_years, n_lags) --
    same lambda_weights as _build_hierarchical_lag, but the mean is left
    unmixed for the caller's likelihood to marginalize over.
    Must be called inside a pm.Model() context.
    """
    n_free = n_lags - 1

    mu_logit = pm.Normal(f'{name}_mu_logit', mu=prior_logit, sigma=1.0,
                         shape=n_free)
    tau = pm.HalfNormal(f'{name}_tau', sigma=tau_sigma, shape=n_free)
    raw_offset = pm.Normal(f'{name}_raw_offset', mu=0, sigma=1,
                           shape=(n_areas, n_free))
    area_logit = mu_logit[None, :] + raw_offset * tau[None, :]

    full_logit = pt.concatenate([
        pt.zeros((n_areas, 1)), area_logit
    ], axis=1)
    lambda_weights = pm.Deterministic(
        f'{name}_lambda_weights', pt.special.softmax(full_logit, axis=1))

    z_padded = pt.concatenate([
        pt.as_tensor_variable(pre_inference), z
    ], axis=1)

    shifted = pt.stack([
        z_padded[:, (max_lag - k):(max_lag - k + n_years)]
        for k in range(n_lags)
    ], axis=2)

    return lambda_weights, shifted


def _build_planning_likelihood_marginalized_lag(shifted, lambda_weights, P_obs,
                                                nu_obs, sigma_obs, name='P_like'):
    """
    Marginalized-mixture planning likelihood over lag categories --
    genuine mixture-of-densities, replacing _build_hierarchical_lag's
    mean-mixing (a single StudentT centered on a blended mean):

        log p(P_obs[a,t]) = logsumexp_k( log(lambda_weights[a,k])
                                        + logT(P_obs[a,t]; shifted[a,t,k], sigma) )

    i.e. "this observation came from exactly one lag category, we just
    don't know which" (a genuine categorical mixture), instead of "this
    observation is a literal blend of several lags' z" (what
    _build_hierarchical_lag's mixed-mean construction implicitly assumes).
    Mean-mixing lets every observation be scored against a value no real
    lag category predicts, a known anti-pattern for discrete-category
    uncertainty; whether that's actually the source of AZ1d's residual
    lag_P_mu_logit/tau r-hat problems is the empirical question this
    likelihood (used by AZ1f) was built to test, not an assumed fix --
    see the az1d-multimodal-deepdive artifact discussion for the reasoning
    and its limits.

    shifted        : (n_areas, n_years, n_lags) -- z shifted by each lag k,
                     from _build_hierarchical_lag_marginalized
    lambda_weights : (n_areas, n_lags) -- per-area lag-category simplex
    Adds P_like via pm.Potential and exposes 'P_like_pointwise' (dims
    area/year) so DwellingModel._attach_pointwise_log_likelihood can
    populate the log_likelihood group for LOO/az.compare, the same
    convention _build_planning_likelihood_zeroinflated uses.
    """
    log_p_per_cat = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=shifted, sigma=sigma_obs),
        P_obs[:, :, None]
    )  # (n_areas, n_years, n_lags)
    log_lambda = pt.log(lambda_weights)[:, None, :]  # (n_areas, 1, n_lags)
    log_lik = pt.logsumexp(log_lambda + log_p_per_cat, axis=2)  # (n_areas, n_years)

    pm.Potential(name, log_lik.sum())
    pm.Deterministic(f'{name}_pointwise', log_lik, dims=('area', 'year'))


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
    pm.Deterministic('P_like_pointwise', log_lik, dims=('area', 'year'))


def _build_agreement_gated_likelihood(P_mean, E_mean, P_obs, E_obs, mu_area,
                                      sigma_agree_plan, sigma_agree_ben,
                                      sigma_disagree_plan, sigma_disagree_ben,
                                      rho, nu_obs):
    """
    Joint agreement-gated mixture likelihood for P_obs and E_obs, sharing
    ONE marginalised discrete state per area-year across both sources
    (rather than two independent per-source likelihoods, as in
    _build_planning_likelihood_simple called twice).

      agree    (prob rho):    P_obs ~ StudentT(P_mean, sigma_agree_plan)
                               E_obs ~ StudentT(E_mean, sigma_agree_ben)
      disagree (prob 1-rho):  P_obs ~ StudentT(mu_area, sigma_disagree_plan)
                               E_obs ~ StudentT(mu_area, sigma_disagree_ben)

    Targets a failure mode per-source robust likelihoods can't reach: P
    and E individually plausible but jointly inconsistent. StudentT tails
    (already used throughout this module) down-weight a single source's
    own outliers against a fixed mean; they have no mechanism that reacts
    to the two sources disagreeing with EACH OTHER, which is what left
    sigma_slab collapsed near 0 in M9 (cross-source disagreement dumped
    into sigma_plan/sigma_ben instead of being explained). In the
    disagree branch, both sources are explained by the area's fixed
    long-run rate (mu_area — the same target used for z's prior mean)
    rather than by z's current lag-weighted mean, so a disagreeing
    area-year contributes ~zero gradient to z: z reverts toward mu_area
    for that year instead of being pulled toward some compromise between
    two conflicting sources.

    rho is a single global mixture weight, deliberately not per-area —
    see _build_z_prior_hierarchical's docstring on the funnel risk of
    under-informed per-area hierarchies; revisit only if diagnostics show
    a single global agreement rate is too coarse.

    Implemented via pm.Potential + logaddexp (same pattern as
    _build_planning_likelihood_zeroinflated) rather than pm.Mixture, so
    the discrete state is marginalised analytically instead of sampled —
    avoids the divergence/label-switching issues of sampling a discrete
    latent with NUTS.

    Adds Deterministic 'agreement_prob' (dims=('area', 'year')) — the
    posterior responsibility of the agree branch per area-year, i.e. how
    much this area-year's data actually informed z. Inspect this for
    diagnostics: it should show real per-area/per-year structure (not
    collapse to ~0 or ~1 everywhere), and should correlate with
    -|P_obs - E_obs|.

    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    log_p_agree = (
        pm.logp(pm.StudentT.dist(nu=nu_obs, mu=P_mean, sigma=sigma_agree_plan), P_obs) +
        pm.logp(pm.StudentT.dist(nu=nu_obs, mu=E_mean, sigma=sigma_agree_ben), E_obs)
    )
    log_p_disagree = (
        pm.logp(pm.StudentT.dist(nu=nu_obs, mu=mu_area[:, None],
                                 sigma=sigma_disagree_plan), P_obs) +
        pm.logp(pm.StudentT.dist(nu=nu_obs, mu=mu_area[:, None],
                                 sigma=sigma_disagree_ben), E_obs)
    )

    log_lik_agree    = pt.log(rho)       + log_p_agree
    log_lik_disagree = pt.log(1 - rho)   + log_p_disagree
    log_lik          = pt.logaddexp(log_lik_agree, log_lik_disagree)

    pm.Potential('PE_like', log_lik.sum())
    pm.Deterministic('PE_like_pointwise', log_lik, dims=('area', 'year'))
    agreement_prob = pm.Deterministic(
        'agreement_prob', pt.exp(log_lik_agree - log_lik), dims=('area', 'year'))

    return agreement_prob


def _build_independent_agreement_gated_likelihood(P_mean, E_mean, P_obs, E_obs, mu_area,
                                                   sigma_agree_plan, sigma_agree_ben,
                                                   sigma_disagree_plan, sigma_disagree_ben,
                                                   rho_P, rho_E, nu_obs):
    """
    Per-source independent agreement-gated mixture likelihood — replaces
    _build_agreement_gated_likelihood's single JOINT (P, E) mixture with
    two independent per-source mixtures. Each source gets its own
    marginalised "informative this year" state, gated against z's
    lag-weighted mean, versus "uninformative", gated against the area's
    fixed long-run rate mu_area — without requiring the OTHER source to
    agree first.

    Motivation: an empirical area-year taxonomy over the 200-area sample
    (raw P_obs/E_obs, active-threshold=3 dwellings/year) found:
      quiet (both silent):                50.0%
      agree (same-sign overlap):          10.1%
      conflict (opposite-sign overlap):    2.0%
      disjoint (one source active only):  38.0%  (P-only 10.6%, E-only 27.4%)
    _build_agreement_gated_likelihood's joint mixture forces ONE shared
    state across both sources per area-year, so the dominant non-trivial
    pattern here — one source genuinely informative, the other simply
    not tracking anything that year, not actively contradicting it — gets
    penalised on whichever source is silent: it fits neither branch well,
    since the "agree" branch expects it to match a now-elevated z and the
    "disagree" branch's fit is degraded by the other source having
    already pulled z off mu_area. True same-year conflict (opposite-sign
    overlap) is rare (2%) by comparison, so this targets the common case,
    not the rare one.

    rho_P, rho_E are independent global Beta-distributed mixture
    weights (each analogous to M11's single rho_agree) — kept
    independent rather than shared because the empirical P-only vs
    E-only activity rates differ by more than 2x (10.6% vs 27.4%); a
    single shared rate would misrepresent whichever source it doesn't
    match.

    Adds Deterministic 'agreement_prob_P', 'agreement_prob_E' (each
    dims=('area', 'year')) — per-source posterior responsibility of the
    informative branch, for diagnostics.

    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    log_p_agree_P    = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=P_mean, sigma=sigma_agree_plan), P_obs)
    log_p_disagree_P = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=mu_area[:, None],
                         sigma=sigma_disagree_plan), P_obs)
    log_lik_agree_P    = pt.log(rho_P)     + log_p_agree_P
    log_lik_disagree_P = pt.log(1 - rho_P) + log_p_disagree_P
    log_lik_P = pt.logaddexp(log_lik_agree_P, log_lik_disagree_P)

    log_p_agree_E    = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=E_mean, sigma=sigma_agree_ben), E_obs)
    log_p_disagree_E = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=mu_area[:, None],
                         sigma=sigma_disagree_ben), E_obs)
    log_lik_agree_E    = pt.log(rho_E)     + log_p_agree_E
    log_lik_disagree_E = pt.log(1 - rho_E) + log_p_disagree_E
    log_lik_E = pt.logaddexp(log_lik_agree_E, log_lik_disagree_E)

    pm.Potential('P_like', log_lik_P.sum())
    pm.Potential('E_like', log_lik_E.sum())
    pm.Deterministic('P_like_pointwise', log_lik_P, dims=('area', 'year'))
    pm.Deterministic('E_like_pointwise', log_lik_E, dims=('area', 'year'))

    agreement_prob_P = pm.Deterministic(
        'agreement_prob_P', pt.exp(log_lik_agree_P - log_lik_P), dims=('area', 'year'))
    agreement_prob_E = pm.Deterministic(
        'agreement_prob_E', pt.exp(log_lik_agree_E - log_lik_E), dims=('area', 'year'))

    return agreement_prob_P, agreement_prob_E


def _build_temporal_reallocation_likelihood(z, obs, mu_area,
                                            sigma_agree, sigma_disagree,
                                            rho, nu_obs, name,
                                            active_threshold=3.0, max_offset=2):
    """
    Per-source agreement-gated likelihood where the AGREE branch's mean
    is a per-ACTIVE-RECORD marginalised temporal offset instead of
    _build_lag's population-level convolution kernel.

    Motivation: an empirical check (raw P_obs/E_obs, 200-area sample)
    found that for records where P and E have a same-sign match nearby
    in time, the offset between them (E_year - P_year) is idiosyncratic
    per record: 54.8% land exactly on the same year, but with a real,
    asymmetric spread (-1: 21.3%, +1: 7.6%, -2: 9.3%, +2: 1.6%; 94.6%
    within +/-2 years overall). _build_lag fits ONE population-wide lag
    distribution per source and convolves it into EVERY year, active or
    not — right on average, wrong for any specific record, including the
    single most common case (same-year, offset=0): see M11's
    E01002694 (a same-year P/E match, 54.8% of matches are this pattern)
    where the fitted lambda_weights_P happened to favour 2-3-year-old
    signal, so the model couldn't connect the 2014 spike to z[2014] at
    all, regardless of sampler/parameterisation — the observation simply
    isn't compared against the right z value under a smoothed kernel.

    This builder only applies offset marginalisation to ACTIVE records
    (|obs[a,t]| > active_threshold — a fixed, data-derived boolean mask,
    same convention as _build_planning_likelihood_zeroinflated's
    is_zero). Inactive (near-threshold) cells compare directly against
    z[a,t] with no shift: there's no real signal there to reallocate,
    and marginalising an offset for ~50% of all cells that carry no
    signal would add cost for no benefit — only ~1,200 of 2,000
    area-year cells in the 200-area sample are active for either source.

    pi_offset ~ Dirichlet over 2*max_offset+1 candidate shifts
    (default +/-2, covering ~95% of empirically matched pairs) is
    SHARED across areas, like lambda_weights, but applied PER RECORD via
    marginalisation rather than as a convolution smoothing every cell —
    the shared part only encodes "how far do real records typically
    drift", not "blend together every year's signal by this fixed
    amount". Deliberately uninformative (Dirichlet(ones)) rather than
    seeded from the empirical histogram above, so the fitted posterior
    can be checked against that histogram as a validation.

    Records within max_offset of the window boundary (year 0 or
    n_years-1) have their out-of-window candidate shifts masked out of
    the marginalisation — an approximate, unnormalised truncation, same
    spirit as _build_pre_inference's boundary padding (documented there
    as a real ceiling on achievable correction, not a bug).

    disagree branch is unchanged from
    _build_independent_agreement_gated_likelihood — obs[a,t] compared to
    the area's flat mu_area, no offset, for BOTH active and inactive
    cells.

    Deliberately excludes spatial reallocation: an empirical Moran's I /
    best-1-hop-neighbour-cancellation check (see chat) found no signal
    for either P/E disagreement or the census/(P+E) gap beyond a
    random-relabelling null — spatial misallocation between queen
    neighbours does not look like a productive mechanism to add here.

    Returns (agreement_prob, pi_offset). agreement_prob has
    dims=('area', 'year') for ALL cells (offset-marginalised where
    active, direct where not), matching M11/M12's diagnostic surface.
    name: 'P' or 'E' — used to build f'pi_offset_{name}' /
    f'{name}_like' so this can be called twice in the same model.

    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    n_areas, n_years = obs.shape
    active = np.abs(obs) > active_threshold
    active_area, active_year   = np.where(active)
    inactive_area, inactive_year = np.where(~active)

    offsets = np.arange(-max_offset, max_offset + 1)
    K = len(offsets)
    pi_offset = pm.Dirichlet(f'pi_offset_{name}', a=np.ones(K))

    # ── Active records: marginalise over candidate offsets ────────────────────
    shifted_year = active_year[:, None] + offsets[None, :]              # (n_active, K)
    valid        = (shifted_year >= 0) & (shifted_year < n_years)
    shifted_year_clipped = np.clip(shifted_year, 0, n_years - 1)

    z_shifted   = z[active_area[:, None], shifted_year_clipped]         # (n_active, K)
    obs_active  = obs[active_area, active_year]                          # (n_active,) fixed

    log_p_k  = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=z_shifted, sigma=sigma_agree),
                       obs_active[:, None])
    log_pi   = pt.log(pi_offset)[None, :]
    log_terms_active = pt.switch(valid, log_pi + log_p_k, -1e10)
    log_p_agree_active = pt.logsumexp(log_terms_active, axis=1)         # (n_active,)

    log_p_disagree_active = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=mu_area[active_area], sigma=sigma_disagree),
        obs_active)

    log_lik_agree_active    = pt.log(rho)     + log_p_agree_active
    log_lik_disagree_active = pt.log(1 - rho) + log_p_disagree_active
    log_lik_active = pt.logaddexp(log_lik_agree_active, log_lik_disagree_active)
    resp_active    = pt.exp(log_lik_agree_active - log_lik_active)

    # ── Inactive records: direct comparison, no offset ─────────────────────────
    z_direct      = z[inactive_area, inactive_year]                     # (n_inactive,)
    obs_inactive  = obs[inactive_area, inactive_year]                    # fixed

    log_p_agree_inactive    = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=z_direct, sigma=sigma_agree), obs_inactive)
    log_p_disagree_inactive = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=mu_area[inactive_area], sigma=sigma_disagree),
        obs_inactive)

    log_lik_agree_inactive    = pt.log(rho)     + log_p_agree_inactive
    log_lik_disagree_inactive = pt.log(1 - rho) + log_p_disagree_inactive
    log_lik_inactive = pt.logaddexp(log_lik_agree_inactive, log_lik_disagree_inactive)
    resp_inactive    = pt.exp(log_lik_agree_inactive - log_lik_inactive)

    pm.Potential(f'{name}_like', log_lik_active.sum() + log_lik_inactive.sum())

    loglik_grid = pt.zeros((n_areas, n_years))
    loglik_grid = pt.set_subtensor(loglik_grid[active_area, active_year], log_lik_active)
    loglik_grid = pt.set_subtensor(loglik_grid[inactive_area, inactive_year], log_lik_inactive)
    pm.Deterministic(f'{name}_like_pointwise', loglik_grid, dims=('area', 'year'))

    resp_grid = pt.zeros((n_areas, n_years))
    resp_grid = pt.set_subtensor(resp_grid[active_area, active_year], resp_active)
    resp_grid = pt.set_subtensor(resp_grid[inactive_area, inactive_year], resp_inactive)
    agreement_prob = pm.Deterministic(
        f'agreement_prob_{name}', resp_grid, dims=('area', 'year'))

    return agreement_prob, pi_offset


def _build_temporal_reallocation_likelihood_marginalizable(z, obs, mu_area,
                                                            sigma_agree, sigma_disagree,
                                                            rho, nu_obs, name,
                                                            active_threshold=3.0, max_offset=2):
    """
    _build_temporal_reallocation_likelihood, rebuilt so profile_k CAN be
    marginalised by pymc_extras.marginalize() (used by M16) instead of
    sampled via literal CategoricalGibbsMetropolis (M14/M15).

    Identical math to _build_temporal_reallocation_likelihood (the same
    active/inactive split, per-record offset marginalisation, and
    agree/disagree gate) — the only difference is HOW the total
    log-density is attached to the model: as a pm.CustomDist (a genuine
    Distribution/RV node) instead of a pm.Potential (a bare scalar term
    with no RV identity). pymc_extras.marginalize() can only trace a
    discrete RV's downstream effect through actual Distribution nodes,
    not through a Deterministic feeding a Potential — verified directly:
    swapping Potential for CustomDist here is sufficient for
    marginalize() to successfully marginalise profile_k out of a model
    using this builder (see M16's docstring for the validation, including
    the two-CustomDists-sharing-one-marginalised-RV case, which does NOT
    silently drop either source's contribution to the joint marginal
    logp).

    IMPORTANT: unlike the Potential version, this does NOT build
    'agreement_prob_{name}' or '{name}_like_pointwise' Deterministics —
    doing so would ALSO block marginalisation (the same "dependent
    Deterministic" restriction that applies to z itself — see
    _build_z_prior_profile_library_horseshoe's docstring). Callers that
    need those must reconstruct them in numpy after sampling and
    recovering profile_k via pymc_extras.recover_marginals() — see M16's
    sample() override, which reimplements this exact formula in
    numpy/scipy for that purpose. pm.compute_log_likelihood() ALSO can't
    be relied on here even after sampling: when two CustomDists (P_like,
    E_like) share the same marginalised RV, pymc_extras cannot cleanly
    separate their pointwise log-likelihoods afterward (a
    "NonSeparableLogpWarning" — one of the two collapses to a degenerate
    per-draw scalar instead of a proper per-area breakdown) — this is
    also handled by M16's numpy reconstruction, not by
    pm.compute_log_likelihood().

    IMPLEMENTATION NOTE — this does NOT gather across a global flattened
    list of active (area, year) cells the way
    _build_temporal_reallocation_likelihood does. First attempt did
    exactly that and pymc_extras.marginalize() rejected it: "Use of known
    dimensions as core dimensions ... not supported" / "The graph between
    the marginalized and dependent RVs cannot be marginalized
    efficiently." marginalize() needs `area` to remain a genuine BATCH
    dimension (since profile_k has one value per area and marginalisation
    is conceptually "for each area, sum over that area's own candidate
    values") — declaring area as a core dim of one opaque multi-area
    CustomDist call (as the gather-based version does) hides that
    structure from it. Fixed by expressing the whole computation as a
    per-area-ROW (core dims = year only) calculation using masking/
    switching instead of cross-area index-gathering: for EVERY year in a
    row, compute both the "active" (offset-marginalised) and "inactive"
    (direct) log-density, then pt.switch on whether that year is active,
    rather than only computing the relevant one for a pre-selected
    subset. Mathematically identical to
    _build_temporal_reallocation_likelihood; area is now a proper batch
    dimension pymc_extras can marginalise profile_k across.

    mu_area is passed in as an explicit dist_param (mu_area_pt) rather
    than closed over as a fixed (n_areas,) array, for the same reason —
    it must batch over area alongside z, not be baked into the graph as
    an opaque per-area lookup.

    The CustomDist's logp returns one value PER AREA (summed over years),
    not one value per (area, year) cell. This also means the natural LOO
    unit for a model built this way is "one area" (all 10 years
    together), not "one area-year cell" like M11-M13 — a real difference
    in what "held out" means, not just an implementation detail.

    Returns pi_offset only (no agreement_prob — see above).
    Must be called inside a pm.Model() context. z must NOT be a
    pm.Deterministic (see _build_z_prior_profile_library_horseshoe's
    wrap_z_as_deterministic=False).
    """
    n_areas, n_years = obs.shape

    offsets = np.arange(-max_offset, max_offset + 1)
    K = len(offsets)
    pi_offset = pm.Dirichlet(f'pi_offset_{name}', a=np.ones(K))

    # Shifted-year index/validity arrays as a pure function of year t —
    # identical for every area, so this can be precomputed once and
    # reused inside the per-area-row logp.
    years = np.arange(n_years)
    shifted_year = years[:, None] + offsets[None, :]               # (n_years, K)
    valid        = (shifted_year >= 0) & (shifted_year < n_years)   # (n_years, K)
    shifted_year_clipped = np.clip(shifted_year, 0, n_years - 1)    # (n_years, K)

    # One-hot selection tensor for the offset gather below, built once as
    # a fixed numpy constant. Used via einsum (contracting only z's LAST
    # axis) rather than z[shifted_year_clipped] fancy indexing: plain
    # fancy indexing assumes z is exactly 1-D and breaks once
    # pymc_extras.marginalize() vectorises this logp over extra leading
    # batch dims (area, and the profile_k candidates being marginalised)
    # — einsum with a leading '...' broadcasts correctly over however
    # many such dims get inserted, fancy indexing does not.
    select = np.zeros((n_years, K, n_years))
    select[years[:, None], np.arange(K)[None, :], shifted_year_clipped] = 1.0
    select_pt = pt.as_tensor_variable(select)

    mu_area_pt = pt.as_tensor_variable(mu_area.astype('float64'))

    def logp(value, z, mu_area_i, sigma_agree, sigma_disagree, rho, pi_offset):
        # value, z: (..., year) — one area's row, PLUS however many extra
        # leading batch dims pymc_extras.marginalize()'s vectorisation
        # inserts (area, and the profile_k candidates being enumerated).
        # mu_area_i/sigma_agree/sigma_disagree/rho: (...,) scalar-per-batch.
        # pi_offset: (..., K). Every axis insertion below uses trailing
        # (Ellipsis-relative) indexing rather than [:, None]/[None, :],
        # which would assume a fixed, wrong number of leading dims.
        is_active = pt.abs(value) > active_threshold                      # (..., year)

        z_shifted = pt.einsum('...y,tky->...tk', z, select_pt)            # (..., year, K)
        log_p_k   = pm.logp(
            pm.StudentT.dist(nu=nu_obs, mu=z_shifted, sigma=sigma_agree[..., None, None]),
            value[..., :, None])                                          # (..., year, K)
        log_pi    = pt.log(pi_offset)[..., None, :]                       # (..., 1, K)
        log_terms = pt.switch(valid, log_pi + log_p_k, -1e10)             # (..., year, K)
        log_p_agree_offset = pt.logsumexp(log_terms, axis=-1)             # (..., year)

        log_p_agree_direct = pm.logp(
            pm.StudentT.dist(nu=nu_obs, mu=z, sigma=sigma_agree[..., None]), value)  # (..., year)

        log_p_agree = pt.switch(is_active, log_p_agree_offset, log_p_agree_direct)

        log_p_disagree = pm.logp(
            pm.StudentT.dist(nu=nu_obs, mu=mu_area_i[..., None], sigma=sigma_disagree[..., None]),
            value)                                                        # (..., year)

        log_lik_agree    = pt.log(rho)[..., None]     + log_p_agree
        log_lik_disagree = pt.log(1 - rho)[..., None] + log_p_disagree
        log_lik = pt.logaddexp(log_lik_agree, log_lik_disagree)           # (..., year)

        return log_lik.sum(axis=-1)  # (...,) — scalar per area (per batch)

    pm.CustomDist(f'{name}_like', z, mu_area_pt, sigma_agree, sigma_disagree, rho, pi_offset,
                  logp=logp, observed=obs,
                  signature='(year),(),(),(),(),(offset)->()')

    return pi_offset


def _build_backward_reallocation_likelihood(z, obs, boundary_target,
                                            sigma_obs, sigma_noise, rho,
                                            nu_obs, name):
    """
    Per-source, per-record 3-way mixture likelihood for the AZ family —
    replaces both _build_lag's population-wide convolution kernel and
    _build_temporal_reallocation_likelihood['_marginalizable']'s
    active/inactive-gated offset marginalisation with a single mechanism
    applied UNIFORMLY to every (area, year) cell:

      same-year (prob rho[0]):  obs[a,t] ~ StudentT(z[a, t],        sigma_obs)
      prior-year(prob rho[1]):  obs[a,t] ~ StudentT(z[a, t-1],      sigma_obs)
      noise     (prob rho[2]):  obs[a,t] ~ StudentT(0,              sigma_noise)

    No active/inactive split: every cell gets the full 3-way mixture,
    whether or not anything happened that year — deliberately, per
    design discussion (a fixed active_threshold was rejected as an
    arbitrary distinction the mechanism shouldn't need).

    Backward-only: the prior-year component is z[a, t-1], never
    z[a, t+1] — a record can be explained by activity that happened
    earlier and was logged late, never by activity that hasn't happened
    yet. Only one year back (not _build_temporal_reallocation_
    likelihood's +/-max_offset window) — simplest version that covers
    the single most informative reassignment; revisit only if
    diagnostics show real two-year-back structure going unexplained.

    boundary_target[a] fills the t=0 "prior-year" slot, where no z[a,-1]
    parameter exists — a fixed proxy (mean of the pre-inference-window P
    and E observations immediately before the inference start), not a
    modelled parameter. No uncertainty propagates from it.

    rho ~ Dirichlet(ones(3)) is a single global 3-way split, shared
    across every (area, year) cell for this source — same status as
    lambda_weights/pi_offset in earlier models: not identifiable per
    cell from one observation alone. What DOES vary per cell is the
    POSTERIOR responsibility (resp_same/resp_prior/resp_noise below),
    exactly analogous to agreement_prob_P/E in M11-M16.

    sigma_obs is shared between the same-year and prior-year components
    (both represent genuine planning/BEN measurement noise around a real
    z value — only the target year differs, not the measurement
    process). sigma_noise is a separate, wider scale for the noise
    component, which explains a record with no structured relationship
    to z at all.

    Implemented via pm.Potential + logsumexp (same pattern as
    _build_agreement_gated_likelihood) — the mixture is already
    continuous/marginalised in log-space, no discrete latent, so this
    stays fully nutpie-compatible.

    Returns (resp_same, resp_prior, resp_noise), each dims=('area', 'year').
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    prior_target = pt.concatenate([
        pt.as_tensor_variable(boundary_target.astype('float64'))[:, None],
        z[:, :-1],
    ], axis=1)  # (n_areas, n_years)

    log_p_same  = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=z,            sigma=sigma_obs),   obs)
    log_p_prior = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=prior_target, sigma=sigma_obs),   obs)
    log_p_noise = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=0.0,          sigma=sigma_noise), obs)

    log_terms = pt.stack([
        pt.log(rho[0]) + log_p_same,
        pt.log(rho[1]) + log_p_prior,
        pt.log(rho[2]) + log_p_noise,
    ], axis=-1)                                 # (area, year, 3)
    log_lik = pt.logsumexp(log_terms, axis=-1)   # (area, year)

    pm.Potential(f'{name}_like', log_lik.sum())
    pm.Deterministic(f'{name}_like_pointwise', log_lik, dims=('area', 'year'))

    resp = pt.exp(log_terms - log_lik[:, :, None])  # (area, year, 3)
    resp_same  = pm.Deterministic(f'resp_same_{name}',  resp[:, :, 0], dims=('area', 'year'))
    resp_prior = pm.Deterministic(f'resp_prior_{name}', resp[:, :, 1], dims=('area', 'year'))
    resp_noise = pm.Deterministic(f'resp_noise_{name}', resp[:, :, 2], dims=('area', 'year'))

    return resp_same, resp_prior, resp_noise


def _build_backward_reallocation_likelihood_2way(z, obs, boundary_target,
                                                  sigma_obs, rho, nu_obs, name):
    """
    _build_backward_reallocation_likelihood with the "noise" branch
    dropped entirely, instead of re-fixed — same-year vs one-year-prior
    only, sharing ONE sigma_obs between both branches (no separate
    sigma_noise to introduce).

    Motivation: AZ0's 3-way version (this function's sibling above)
    diverged catastrophically on real data (max r-hat 17.8, ~83% of
    draws divergent) because sigma_noise_P/E collapsed toward 0,
    building an unbounded-density spike at StudentT(mu=0, sigma->0)'s
    peak — exploitable because 54.3%/28.2% of P_obs/E_obs cells are
    EXACTLY 0, a large, fixed-point-coincident mass of real data for a
    freely-shrinking scale to collapse onto. Both branches here are
    centred on a genuinely MOVING target (z[a,t] or z[a,t-1]), not a
    fixed point, and sigma_obs isn't a new parameter (it's the same
    sigma_plan/sigma_ben AZ0a already samples cleanly) — so there's no
    equivalent degenerate mode available, without needing to re-litigate
    or bound the noise branch that actually broke it.

    log p(obs[a,t]) = logsumexp(
        log(rho)     + StudentT_lpdf(obs[a,t] | z[a,t],            sigma_obs),
        log(1 - rho) + StudentT_lpdf(obs[a,t] | prior_target[a,t], sigma_obs),
    )

    Same backward-only / boundary-proxy semantics as
    _build_backward_reallocation_likelihood: prior_target[a,t] is
    z[a,t-1] for t>0, and the fixed pre-inference-window proxy
    boundary_target[a] for t=0 — never a future year.

    rho ~ (caller-supplied prior, e.g. Beta(2,2) matching M11's
    rho_agree convention) is the shared, population-level weight on the
    same-year branch — not identifiable per cell from a single
    observation. resp_same_{name} (the per-cell POSTERIOR responsibility)
    is what actually varies per (area, year).

    Returns resp_same, dims=('area', 'year').
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    prior_target = pt.concatenate([
        pt.as_tensor_variable(boundary_target.astype('float64'))[:, None],
        z[:, :-1],
    ], axis=1)  # (n_areas, n_years)

    log_p_same  = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=z,            sigma=sigma_obs), obs)
    log_p_prior = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=prior_target, sigma=sigma_obs), obs)

    log_lik_same  = pt.log(rho)       + log_p_same
    log_lik_prior = pt.log(1 - rho)   + log_p_prior
    log_lik = pt.logaddexp(log_lik_same, log_lik_prior)  # (area, year)

    pm.Potential(f'{name}_like', log_lik.sum())
    pm.Deterministic(f'{name}_like_pointwise', log_lik, dims=('area', 'year'))

    resp_same = pm.Deterministic(
        f'resp_same_{name}', pt.exp(log_lik_same - log_lik), dims=('area', 'year'))

    return resp_same


def _build_noise_mixture_likelihood(z, obs, sigma_obs, sigma_noise_floor,
                                    nu_obs, name):
    """
    Per-cell 2-way mixture: obs[a,t] is explained either by z[a,t]
    (genuine signal) or by a fixed-at-zero noise/outlier branch, with a
    HARD FLOOR on the noise branch's scale.

    log p(obs[a,t]) = logsumexp(
        log(rho)     + StudentT_lpdf(obs[a,t] | z[a,t], sigma_obs),
        log(1 - rho) + StudentT_lpdf(obs[a,t] | 0,       sigma_noise),
    )
    sigma_noise = sigma_noise_floor + HalfNormal(...)   -- floor is a hard,
    additive lower bound on the RV itself, not just a prior scale hint.

    This is the SAME degree of freedom that broke AZ0 catastrophically
    (max r-hat 17.8, ~83% divergent draws): a noise branch centred at a
    FIXED point (0) with a free-to-shrink scale has unbounded density
    exactly at that point, and 54.3%/28.2% of P_obs/E_obs cells are
    EXACTLY 0 -- a large, fixed-point-coincident mass a freely-shrinking
    scale can collapse onto. AZ0b's fix was to drop the noise branch
    entirely (both remaining components centred on a moving target, so no
    fixed point to collapse onto). This model needs the noise branch back
    -- that's the whole point, to catch genuine outliers a moving-target
    branch can never explain (see docs/az-family-work-plan.md Phase 3,
    e.g. E01001774: P_sum=460 against a decade census total D=18, a
    magnitude mismatch no amount of lag/reallocation could fix) -- so this
    time the fix is a hard floor instead: `HalfNormal(...)` alone has its
    MODE at 0 regardless of its own sigma parameter, so a small prior
    scale is NOT a floor and would reproduce AZ0's exact failure. Adding a
    fixed positive constant is what actually prevents sigma_noise from
    ever reaching 0.

    sigma_noise_floor should sit well above sigma_obs's typical scale
    (AZ0a's sigma_plan/sigma_ben converge around 7-9) so the noise branch
    is unambiguously "much more tolerant", not a near-duplicate of the
    signal branch -- see AZ3's docstring for the calibrated value.

    rho ~ Beta(prior favouring rho close to 1 -- most cells are expected
    to be genuine signal, only a minority true outliers), shared/
    population-level, not identifiable per cell from one observation.
    resp_noise_{name} (the per-cell POSTERIOR responsibility) is what
    actually varies per (area, year), and is exactly what a downstream
    automatic-flagging report should read (see
    diagnostics.posterior_outlier_summary).

    Returns (rho, sigma_noise, resp_noise). resp_noise has
    dims=('area', 'year').
    Must be called inside a pm.Model() context with 'area'/'year' coords.
    """
    rho = pm.Beta(f'rho_{name}', alpha=8, beta=2)  # prior mean 0.8

    sigma_noise_excess = pm.HalfNormal(f'sigma_noise_{name}_excess', sigma=15)
    sigma_noise = pm.Deterministic(
        f'sigma_noise_{name}', sigma_noise_floor + sigma_noise_excess)

    log_p_signal = pm.logp(pm.StudentT.dist(nu=nu_obs, mu=z, sigma=sigma_obs), obs)
    log_p_noise  = pm.logp(
        pm.StudentT.dist(nu=nu_obs, mu=0, sigma=sigma_noise), obs)

    log_lik_signal = pt.log(rho)       + log_p_signal
    log_lik_noise  = pt.log(1 - rho)   + log_p_noise
    log_lik = pt.logaddexp(log_lik_signal, log_lik_noise)  # (area, year)

    pm.Potential(f'{name}_like', log_lik.sum())
    pm.Deterministic(f'{name}_like_pointwise', log_lik, dims=('area', 'year'))

    resp_noise = pm.Deterministic(
        f'resp_noise_{name}', pt.exp(log_lik_noise - log_lik), dims=('area', 'year'))

    return rho, sigma_noise, resp_noise


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


