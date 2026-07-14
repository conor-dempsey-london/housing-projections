import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from housing_projections.config import (
    ALL_COLS_BEN,
    ALL_COLS_PLAN,
    CENSUS_ABS_FLOOR,
    CENSUS_REL_ERROR,
    DEFAULT_SAMPLE_KWARGS,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
    INFER_YEARS,
)
from housing_projections.spatial import build_spatial_weights

from .base import DwellingModel

# __all__ and ALL_MODELS are assembled at the bottom of this file, once all
# model classes below are defined — that class list is the single source of
# truth for "every implemented model"; housing_projections.models and the CLI
# both derive their registries from ALL_MODELS rather than re-listing models.

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


# ── AZ0: Anchored zero-sum prior + backward-only reallocation mixture ────────

class AZ0(DwellingModel):
    """
    First model in a new family (AZ — Anchored Zero-sum), built from a
    fresh-slate rethink rather than as an M16 patch. M9-M16 all shared one
    underlying failure, made visible by z_flatness_summary (see
    diagnostics.py): several of them (M9, M10, M15) produced z that is
    ~95-100% flat despite ~95% of areas having real active P/E signal —
    good sampling diagnostics (r-hat, divergences) do NOT imply the model
    is doing anything with the data. This family starts over from the
    generative story we actually want, rather than patching the previous
    one's symptoms:

      "capture the area average; let E and P inform temporal profiles;
       raise or lower any offset needed to satisfy the census constraint
       by distributing it across years without other signal; where P and
       E disagree, check first whether either could be explained by
       completions in a PREVIOUS year (never a future one), including
       partial reassignment; where that doesn't resolve it, attribute
       the disagreement to noise."

    Two new mechanisms, one per stage:

    1. Prior (_build_zero_sum_z_prior): z[a,:] = mu_area[a] + delta[a,:],
       delta ~ ZeroSumNormal(sigma_delta[a]), sigma_delta[a] = floor +
       k*|D[a]|. Replaces M9-M13's soft census likelihood
       (_build_census_constraint — z only APPROXIMATELY sums to D) and
       M14-M16's discrete profile library (exact, but needs a discrete
       profile_k latent, CategoricalGibbsMetropolis or
       pymc_extras.marginalize(), and — per z_flatness_summary — M15's
       horseshoe collapsed to tau_amplitude~0.0004, i.e. z~flat, despite
       lam_amplitude showing real ~40x per-area differentiation). A
       ZeroSumNormal profile is continuous, has no discrete latent, and
       is fully nutpie-compatible. See _build_zero_sum_z_prior's
       docstring for the spike-vs-trough and k-calibration checks that
       motivated this specific parameterisation.

    2. Likelihood (_build_backward_reallocation_likelihood): per-source,
       per-record 3-way mixture over same-year / one-year-prior / noise,
       applied to EVERY (area, year) cell uniformly (no active/inactive
       split, unlike _build_temporal_reallocation_likelihood's
       active_threshold gate). Backward-only — never a future-year
       reassignment. Replaces _build_lag's population-wide convolution
       kernel (which blends every cell by the same fixed amount
       regardless of whether reallocation is warranted for that specific
       record — see _build_temporal_reallocation_likelihood's docstring
       for the E01002694 case this caused) and
       _build_temporal_reallocation_likelihood['_marginalizable']'s
       +/-max_offset active-record marginalisation (which allows FORWARD
       reassignment and needs an arbitrary active_threshold this design
       explicitly avoids).

    rho_P/rho_E ~ Dirichlet(ones(3)) are the shared, population-level
    PRIOR over the 3-way split for each source — not identifiable
    per-cell from a single observation. What varies per (area, year) is
    the POSTERIOR responsibility (resp_same_P/E, resp_prior_P/E,
    resp_noise_P/E), exactly the same prior-vs-posterior distinction as
    agreement_prob_P/E in M11-M16.

    k/floor (sigma_delta_floor/k_sigma_delta below) are fixed
    hyperparameters, first-pass calibrated against real-data checks (see
    _build_zero_sum_z_prior's docstring) — not yet validated against an
    actual posterior fit of this model. Revisit once real traces exist.

    `sample_kwargs` overrides target_accept to 0.95, matching every other
    mixture-likelihood model in this module (M11-M16) — the logsumexp
    mixture geometry is the shared reason, not per-area hierarchy funnels
    this time (there is no hierarchy on sigma_delta here).
    """

    name        = 'AZ0'
    description = ('Anchored zero-sum z prior (exact census sum, no discrete '
                   'latent) + per-record backward-only same-year/prior-year/'
                   'noise mixture likelihood for P and E')
    var_names   = ['sigma_obs_P', 'sigma_obs_E', 'sigma_noise_P', 'sigma_noise_E',
                   'rho_P', 'rho_E']
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    sigma_delta_floor = 3.0
    k_sigma_delta     = 0.08

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, max_lag=1, source='P')
        pre_inference_E = _build_pre_inference(data, max_lag=1, source='E')
        boundary_target = (pre_inference_P[:, 0] + pre_inference_E[:, 0]) / 2.0

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_obs_P   = pm.HalfNormal('sigma_obs_P', sigma=3)
            sigma_obs_E   = pm.HalfNormal('sigma_obs_E', sigma=3)
            sigma_noise_P = pm.HalfNormal('sigma_noise_P', sigma=20)
            sigma_noise_E = pm.HalfNormal('sigma_noise_E', sigma=20)
            rho_P         = pm.Dirichlet('rho_P', a=np.ones(3))
            rho_E         = pm.Dirichlet('rho_E', a=np.ones(3))

            _build_backward_reallocation_likelihood(
                z, data['P_obs'], boundary_target,
                sigma_obs_P, sigma_noise_P, rho_P, self.nu_obs, name='P')
            _build_backward_reallocation_likelihood(
                z, data['E_obs'], boundary_target,
                sigma_obs_E, sigma_noise_E, rho_E, self.nu_obs, name='E')

        self.model = model
        return model


# ── AZ0a: AZ0's z prior alone, simplest possible likelihood ──────────────────

class AZ0a(DwellingModel):
    """
    Diagnostic sibling of AZ0 — isolates whether the anchored zero-sum z
    prior (_build_zero_sum_z_prior) samples cleanly on its own, before
    the blame for AZ0's catastrophic non-convergence (real 200-area data:
    max r-hat 18.7, mean r-hat 8.6, 4991/6000 divergent draws, 3/4 chains
    flagged as trapped in a distinct mode) gets pinned on the prior.

    Direct inspection of AZ0's real trace already found a specific,
    sufficient cause in the LIKELIHOOD, not the prior: sigma_noise_P
    collapsed to 1e-81..1e-108 (a different absurd magnitude per chain —
    the signature of each chain diving down its own version of the same
    degenerate mode), because 54.3% of P_obs cells (28.2% of E_obs) are
    EXACTLY 0, and the noise component's StudentT(mu=0, sigma=sigma_noise)
    has unbounded density at 0 as sigma_noise -> 0 — a classic
    degenerate-variance mixture pathology, structural, not a tuning
    issue. That finding is sufficient on its own to explain the
    divergences, but "sufficient" isn't "exhaustive": this model removes
    _build_backward_reallocation_likelihood entirely and swaps in the
    simplest possible likelihood (add_observation_likelihoods — plain
    StudentT(z, sigma), identical structure to M0/M1) to test the prior
    in isolation, the same one-variable-at-a-time discipline used
    throughout this session rather than assuming the diagnosis above is
    the WHOLE story.

    sigma_plan/sigma_ben priors and structure copied directly from M0 (the
    repo's own "simplest baseline") rather than AZ0's HalfNormal(3) — the
    point of this model is to be the most boring possible likelihood, not
    a slightly-simplified AZ0.

    If this converges cleanly, the zero-sum prior is exonerated and the
    fix belongs entirely in AZ0's likelihood, as already diagnosed. If
    this ALSO fails to converge, the prior needs another look before
    anything is layered back on top of it.
    """

    name        = 'AZ0a'
    description = ('AZ0\'s anchored zero-sum z prior alone + plain StudentT P/E '
                   'likelihood (no mixture) — isolates the prior from AZ0\'s '
                   'likelihood-collapse pathology')
    var_names   = ['sigma_plan', 'sigma_ben']

    sigma_delta_floor = 3.0  # same as AZ0
    k_sigma_delta     = 0.08  # same as AZ0

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                             sigma_plan=sigma_plan,
                                             sigma_ben=sigma_ben)

        self.model = model
        return model


# ── AZ0b: AZ0a + 2-way backward-only reallocation, no noise branch ───────────

class AZ0b(DwellingModel):
    """
    AZ0a + a 2-way backward-only reallocation mixture per source
    (same-year vs one-year-prior) — the next simple addition after
    AZ0a's clean convergence (max r-hat 1.006, 0 divergences on real
    200-area data), deliberately smaller than AZ0's original 3-way
    mixture, which diverged catastrophically (max r-hat 17.8, ~83% of
    draws divergent, 3/4 chains trapped) because its "noise" branch's
    sigma_noise_P/E collapsed toward 0 to build an unbounded-density
    spike at the 54.3%/28.2% exact-zero mass in P_obs/E_obs. This
    version drops the noise branch entirely (see
    _build_backward_reallocation_likelihood_2way's docstring for why
    that removes the specific degenerate mode, not just papers over it)
    and reuses AZ0a's existing sigma_plan/sigma_ben rather than adding
    any new free scale.

    Directly evidenced, not just inherited from earlier design
    discussion: a fresh cross-source lag-matching check on this exact
    200-area sample (each active P_obs record matched against the
    nearest same-sign E_obs record within +/-2 years) found a real
    backward skew — 48.1% of matches have E leading P by 1-2 years,
    vs. only 27.2% with E lagging, 24.7% same-year. That asymmetry is a
    property of the raw data, not an artefact of any particular model's
    z fit.

    Also checked and explicitly NOT addressed here: spatial
    misallocation to Queen-contiguity neighbours (raised as an
    alternative hypothesis for AZ0a's largest residuals, where a single
    year's P or E exceeds the area's entire decade census total D).
    Moran's I on AZ0a's residuals (per year, per source) was mostly
    non-significant, and where significant the sign was POSITIVE
    (neighbours move together — the wrong sign for a misallocation
    story); a neighbour-cancellation permutation test gave only a weak,
    source-inconsistent signal (P: p=0.037 one-sided; E: p=0.147, not
    significant); and the specific large D-exceeding spikes inspected
    directly had no neighbour anywhere near the right magnitude to
    plausibly be the "true" location. Not pursued.

    Expectation, calibrated against AZ0a's own residuals rather than
    assumed: this should meaningfully improve LOO for the subset of
    cells where the plain same-year fit is bad AND a 1-year-back shift
    clearly helps (~30% of AZ0a's largest-residual P cells) — NOT the
    cells where a single year's P or E value exceeds D (~half of
    AZ0a's worst-fit cells), which look like genuine outliers/source
    noise rather than timing misalignment and aren't what this addition
    targets. A properly-floored noise/outlier mechanism for THOSE is a
    later, separate addition — not bundled in here.

    rho_P/rho_E ~ Beta(2, 2) — weight on the same-year branch, matching
    M11's rho_agree convention for a 2-way mixture weight, rather than
    the Dirichlet(ones(3)) AZ0 used for its now-removed 3-way split.

    At target_accept=0.9 (the default) this samples with max r-hat 1.09
    and min ESS 31 (4 params over the 1.01 threshold), noticeably worse
    than AZ0a's clean 1.006/3180 on the same data. A moment-matched
    PSIS-LOO check (pm.stats.loo_moment_match, full run over all 561
    P_like points with pareto k>0.7) found the affine shift/scale
    correction could not improve a SINGLE one of them (elpd unchanged to
    the decimal after ~45 min of correction attempts) -- a much stronger
    signal than ordinary sampling inefficiency, consistent with genuinely
    non-Gaussian/near-discrete leave-one-out posteriors (removing certain
    cells likely flips which mixture component -- same-year vs
    prior-year -- gets credited for nearby cells, not a smooth parameter
    shift).

    Tested (not just assumed) whether this is a step-size/tuning problem:
    re-sampled at target_accept=0.98 and it made r-hat/ESS WORSE, not
    better (max r-hat 1.09 -> 1.12, min ESS 31 -> 22). That rules out
    tuning as the fix and confirms the convergence issue is structural to
    the 2-way mixture's discrete component-identity ambiguity, not a
    step-size problem -- so target_accept is left at the default here
    rather than paying extra compute for no benefit. Resolving this
    properly needs either exact k-fold CV (expensive) or a redesign of
    the reallocation mechanism away from a hard same-year/prior-year
    mixture (e.g. a continuous lag-weighting or an analytically
    marginalized version, as used elsewhere in this file) -- not yet
    attempted.
    """

    name        = 'AZ0b'
    description = ('AZ0a + 2-way backward-only reallocation (same-year vs '
                   'one-year-prior) per source, no noise branch')
    var_names   = ['sigma_plan', 'sigma_ben', 'rho_P', 'rho_E']

    sigma_delta_floor = 3.0  # same as AZ0/AZ0a
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, max_lag=1, source='P')
        pre_inference_E = _build_pre_inference(data, max_lag=1, source='E')
        boundary_target = (pre_inference_P[:, 0] + pre_inference_E[:, 0]) / 2.0

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)
            rho_P      = pm.Beta('rho_P', alpha=2, beta=2)
            rho_E      = pm.Beta('rho_E', alpha=2, beta=2)

            _build_backward_reallocation_likelihood_2way(
                z, data['P_obs'], boundary_target, sigma_plan, rho_P, self.nu_obs, name='P')
            _build_backward_reallocation_likelihood_2way(
                z, data['E_obs'], boundary_target, sigma_ben, rho_E, self.nu_obs, name='E')

        self.model = model
        return model


class AZ1a(DwellingModel):
    """
    AZ0a + a fully-pooled, continuous temporal-lag convolution on BOTH P
    and E, via the SAME _build_lag/_build_pre_inference machinery M1
    already validated (M1 only applied it to planning; here both sources
    get their own independent Dirichlet lag-weight vector, since the
    cross-source lag-matching evidence found in the AZ0b investigation is
    about a real asymmetry BETWEEN the two sources, not just one of them
    lagging a noiseless truth).

    Deliberately NOT AZ0b's discrete same-year/prior-year mixture. That
    mixture's per-cell hidden "which year explains this" choice was
    directly responsible for AZ0b's worse convergence (max r-hat 1.09 vs
    AZ0a's 1.006) AND its unrecoverable PSIS-LOO reliability problem (full
    moment-matching run could not fix a single one of 561 bad-k P_like
    points -- see AZ0b's docstring). A convolution instead computes one
    smooth, deterministic-given-the-weights blend of nearby years for
    every cell:

        P_mean[a,t] = sum_k  lambda_P[k] * z[a, t-k]      (k=0..max_lag)

    with lambda_P ~ Dirichlet(lag_alpha) shared across all 200 areas (the
    base case of the pooled -> partially-pooled -> unpooled ladder --
    borough- or D-band-partial-pooling is a deliberately separate next
    step, not bundled in here, so any convergence/fit change can be
    attributed to one addition at a time). There is no discrete latent
    anywhere in this construction, so there's no analogous
    component-identity ambiguity for NUTS or PSIS-LOO to trip over.

    max_lag=2 matches the empirically-validated window from the old
    M-family (M13/M14's max_offset=2, chosen from a clear signal in a
    direct ±1/±2/±3 window-sensitivity check on raw data) rather than a
    fresh guess. lag_alpha's default ([4, 2, 1][:n_lags], see
    DwellingModel.lag_alpha) concentrates prior mass on same-year
    (k=0) over longer lags, consistent with the raw cross-source check
    finding same-year matches (24.7%) less common than 1-2-year lag
    matches combined, but still the single largest individual category.

    Uses _build_planning_likelihood_simple for both P_like and E_like
    (that builder already supports a name= override for exactly this
    dual-use -- see its docstring) rather than add_observation_likelihoods,
    since the likelihood mean is now P_mean/E_mean (the convolved series),
    not z directly.
    """

    name        = 'AZ1a'
    description = ('AZ0a + fully-pooled continuous lag convolution '
                   '(P and E each get their own Dirichlet lag weights)')
    var_names   = ['sigma_plan', 'sigma_ben', 'lambda_weights_P', 'lambda_weights_E']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ0b
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ0b
    max_lag           = 2     # matches M13/M14's empirically-validated max_offset

    # chains=8 (up from DEFAULT_SAMPLE_KWARGS' 4), cores=8 -- same diagnostic
    # treatment given to AZ1b's genuine hard multimodality (see
    # docs/az-ess-diagnosis.md). This is NOT expected to fix max r-hat (a
    # fully-pooled kernel mechanically cannot represent two areas' truly
    # different lag patterns with one shared vector, so the two-mode split
    # is architectural, not a sampling failure more chains can converge
    # away) -- it exists to get a trustworthy chain-level mode-split
    # estimate via hierarchical_mode_summary, the same way 8 chains did for
    # AZ1b, rather than a noisy 4-chain 2-vs-2 split.
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_lag(z, pre_inference_P, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag,
                                  name='lambda_weights_P')
            _, E_mean = _build_lag(z, pre_inference_E, n_areas, n_years,
                                  self.n_lags, self.lag_alpha, self.max_lag,
                                  name='lambda_weights_E')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(E_mean, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1b(DwellingModel):
    """
    AZ1a + area-level hierarchically-pooled lag weights, replacing AZ1a's
    single shared (fully-pooled) kernel per source with a per-area kernel
    shrunk toward a shared population kernel (see
    _build_hierarchical_lag's docstring for the full construction).

    Directly motivated by a diagnosed failure, not a guess: AZ1a's
    lambda_weights_E was genuinely bimodal across chains (two chains
    settled on "E is same-year", two settled on "E lags true completion by
    1 year") -- a single shared kernel has no one right answer when
    different areas' true lag patterns differ, which is exactly what a
    fully-pooled kernel forces. This reproduces a failure mode already
    documented in this codebase's history for a different model (the old
    M-family's M11: "shared lambda_weights... the real problem was the
    architecture, not the parameterization").

    Deliberately NOT borough-level pooling, which was the original idea
    for this step -- reconsidered and rejected before building it: there's
    no strong mechanistic reason a lag driven by site/development-level
    completion-to-registration timing would align with administrative
    borough boundaries ("delays are caused by things operating below
    Borough level"). Area-level hierarchical pooling targets that directly
    without needing to hand-pick a grouping, and avoids the identifiability
    risk of literal per-area independence (M12's failure: ~10 obs/area/
    source can't identify 200 independent 3-parameter kernels) by shrinking
    weakly-informed areas hard toward the shared kernel via an estimated
    between-area variance (tau) -- tau->0 recovers AZ1a's answer exactly.

    See docs/az-family-work-plan.md for the full reasoning and the
    empirical-lag-clustering alternative considered and deferred (real
    circularity risk: using the same data to define groups and fit the
    hierarchy within them risks looking like an improvement that's really
    just overfitting) in favour of trying this, mechanistically cleaner,
    option first.

    sample_kwargs bumps target_accept to 0.95 (AZ1a used the 0.9 default):
    the hierarchical softmax-over-logits construction is new geometry this
    family hasn't sampled before, and every other model in this codebase
    that introduced new hierarchical/mixture structure (AZ0, M9, M11) made
    the same precautionary bump rather than finding out the hard way.

    chains=8 (up from the DEFAULT_SAMPLE_KWARGS 4), cores=8 -- a deliberate
    choice, not just "more is better". Investigated the residual bad
    r-hat/ESS properly (see docs/az-family-work-plan.md): it isn't a
    geometry/tuning problem (tightening tau_sigma 1.5->0.5 made it WORSE,
    not better, tested not assumed) -- it's genuine hard multimodality for
    ~10-15% of areas. Checked directly: individual chains spent all 1500
    draws entirely inside one of two disconnected modes and never crossed
    over. With only ~10 obs/area/source, some areas' data genuinely can't
    distinguish between two candidate lag years that each explain a spike
    about equally well -- the likelihood itself has two separated peaks
    there, not a single one NUTS is struggling to find.

    Given that, forcing r-hat->1 (chasing it via tighter priors, etc.) would
    mean suppressing real epistemic ambiguity, not fixing a bug -- the
    chosen path here is to report that ambiguity honestly rather than
    force it away. But doing so needs enough chains to trust the *reported
    split* between modes: with only 4 chains, a 3-vs-1 or 2-vs-2 split is
    far too noisy to trust as an estimate of each mode's true relative
    posterior mass (a single chain's initial trajectory into one basin
    or the other is close to a coin flip). 8 chains halves that noise and
    gives a meaningfully more trustworthy per-area mode-split estimate --
    see diagnostics.hierarchical_mode_summary, built alongside this change
    specifically to characterize and report the split per area, rather
    than only flagging it via a scalar max-r-hat number that can't
    distinguish "genuinely multimodal, here's the split" from "just not
    converged yet".
    """

    name        = 'AZ1b'
    description = ('AZ1a + area-level hierarchically-pooled lag weights '
                   '(replacing the single shared kernel per source)')
    var_names   = ['sigma_plan', 'sigma_ben',
                   'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ0b/AZ1a
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ0b/AZ1a
    max_lag           = 2     # same as AZ1a
    tau_sigma         = 1.5   # see docstring -- tightening this to 0.5 made
                              # r-hat/ESS WORSE, not better (tested, not assumed)

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        # Same prior mean as AZ1a's Dirichlet(lag_alpha) encoded in logit
        # space, so switching constructions doesn't also silently change
        # the prior's central tendency (a "prefer short lags" bias).
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_P')
            _, E_mean = _build_hierarchical_lag(
                z, pre_inference_E, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_E')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(E_mean, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1c(DwellingModel):
    """
    AZ1b + a HARD ceiling on tau (see _build_hierarchical_lag_capped's
    docstring for the full construction and motivation) -- a fix attempt
    for AZ1b's genuine hard multimodality (docs/az-ess-diagnosis.md),
    trying AZ1b's own "Status: open" option 1 (much more aggressive
    shrinkage) properly instead of leaving it untested.

    AZ1b's residual max r-hat (1.12-1.19) / min ESS (23-29) were diagnosed
    as genuine hard multimodality for ~10-15% of areas, not a
    sampling/geometry problem: individual chains spend all their draws in
    one of two disconnected modes for areas whose sparse data can't
    distinguish between two candidate lag years explaining a spike about
    equally well. AZ1b already tried tightening tau's PRIOR (tau_sigma
    1.5 -> 0.5) and found it made things WORSE (max r-hat 1.12 -> 1.24,
    min ESS 23 -> 12) because the posterior tau barely moved -- the
    likelihood's pull toward per-area divergence overrides a merely
    tighter prior. This model tests the mechanically different fix that
    observation implies is actually needed: a HARD ceiling on tau itself
    (`tau = tau_cap * Beta(2,2)`, bounded by construction, not just
    discouraged by a prior) rather than a smaller scale on the same free
    HalfNormal.

    Expected, documented tradeoff: capping tau well below AZ1b's
    converged 2.6-5.9 range should suppress the multimodality (less room
    for an area to diverge sharply from the population kernel), but at a
    cost to the very flexibility that produced AZ1b's headline
    E01033711 win (a genuinely large per-area divergence). This is
    checked directly, not assumed -- see docs/az-ess-diagnosis.md for
    whether the spike-tracking win survives.

    Otherwise identical to AZ1b: same z-prior, same sigma_plan/sigma_ben,
    same prior_logit centering, same P_like/E_like construction.
    """

    name        = 'AZ1c'
    description = ('AZ1b + a hard ceiling on tau (tau = tau_cap * Beta(2,2)) '
                   'instead of a free HalfNormal, to test whether capping '
                   'per-area divergence suppresses the hard multimodality')
    var_names   = ['sigma_plan', 'sigma_ben',
                   'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b
    max_lag           = 2     # same as AZ1a/AZ1b
    tau_cap           = 1.5   # see _build_hierarchical_lag_capped's docstring --
                              # well below AZ1b's converged 2.6-5.9 (P) / 5.1-5.9 (E)

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')

        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag_capped(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_cap=self.tau_cap, name='lag_P')
            _, E_mean = _build_hierarchical_lag_capped(
                z, pre_inference_E, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_cap=self.tau_cap, name='lag_E')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(E_mean, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1d(DwellingModel):
    """
    AZ1b with E's hierarchical lag mechanism REMOVED entirely -- E is
    compared directly against SAME-YEAR z (AZ0a's plain likelihood,
    `_build_planning_likelihood_simple(z, ...)`), while P keeps AZ1b's
    area-hierarchical lag unchanged. A targeted asymmetric simplification,
    not a guess: E's lag category has been the disproportionately
    unstable one at every step of this round --

    - AZ1a (fully-pooled): `lambda_weights_E` was the dramatically
      bimodal one across chains; `lambda_weights_P` was comparatively
      mild (see AZ1a's docstring/the ess-diagnosis doc's 8-chain re-check:
      37.5%/62.5% split for E vs a much milder correlated wobble for P).
    - AZ4's causal-chain analysis (az4-diagnostics artifact): it was
      specifically `sigma_noise_E`, not `sigma_noise_P`, that collapsed
      once the noise-mixture's signal branch became lag-aware.
    - AZ1b's own log-likelihood-gap check (docs/az-ess-diagnosis.md,
      Phase 1b follow-up): for flagged areas, the gap between competing
      lag-category chain-groups is near-tied (< 2 nats, genuine epistemic
      ambiguity) MORE OFTEN for `lag_E` (31.6% of flagged areas) than for
      `lag_P` (13.3%) -- E's ambiguity is more often the "real, can't be
      resolved by more sampling" kind.

    If E's lag mechanism is disproportionately responsible for AZ1b's
    residual convergence problems, removing it should resolve a
    meaningful share of them. Documented, checked-not-assumed cost: E
    loses its own ability to explain a genuinely lagged spike (e.g. a
    registration recorded a year after the real completion) -- that
    signal either gets absorbed into z's zero-sum flexibility directly
    (same mechanism AZ0a always had) or shows up as a same-year residual
    tolerated by `sigma_ben`. Whether AZ1b's flagship spike-tracking win
    (LSOA E01033711) survives this is checked directly on real data, not
    assumed either way.
    """

    name        = 'AZ1d'
    description = ("AZ1b with E's lag mechanism removed (same-year only), "
                   "keeping P's hierarchical lag -- testing whether E's "
                   "lag ambiguity is disproportionately responsible for "
                   "AZ1b's residual r-hat/ESS problems")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_tau']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b/AZ1c
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b/AZ1c
    max_lag           = 2     # same as AZ1a/AZ1b/AZ1c (P side only, here)
    tau_sigma         = 1.5   # same as AZ1b (P's hierarchy unchanged)

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_P')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(z, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1e(DwellingModel):
    """
    AZ1d + horseshoe-style local/global tau for P's hierarchical lag (see
    _build_hierarchical_lag_horseshoe's docstring for the full
    construction and motivation) — a fix attempt for AZ1d's residual
    P-lag ambiguity, proposed as an "outside the box" alternative to
    AZ1c's/AZ4b's uniform tau cap (both of which made things worse by
    constraining every area identically, see
    docs/ess-rhat-diagnostic-guide.md Pattern 3).

    AZ1d's own deep-dive (docs/az-ess-diagnosis.md) found a persistent
    13-area core still driving residual r-hat/ESS problems, with a
    log-likelihood-gap check showing this is a MIX of genuine tied
    ambiguity (3 areas) and chains stuck in a decisively worse mode (6
    areas) — median |D| for this core is 7.6x the dataset median, i.e.
    concentrated in the areas with real, large spikes that plausibly
    justify more per-area divergence than the other 187 areas need. A
    single shared tau cannot serve both populations well simultaneously;
    a per-area tau, shrunk MORE aggressively toward the population kernel
    for the well-behaved majority while allowing the flagged minority to
    diverge further without moving the shared scale, targets that
    asymmetry directly rather than constraining everyone the same way.

    Known, documented risk (not assumed away): HalfCauchy's heavy tails
    are a well-known source of their own funnel geometry (see the
    pymc-extras skill's horseshoe-prior guidance) — this may trade one
    ESS problem for a different one in `lag_P_local_scale`/`lag_P_global_tau`
    rather than cleanly fixing anything. target_accept bumped to 0.98
    (above AZ1d's 0.95) as a precaution given this, following the same
    skill's guidance that horseshoe geometry often needs it.
    """

    name        = 'AZ1e'
    description = ("AZ1d + horseshoe-style local/global tau for P's lag "
                   "(per-area tau instead of one shared value) -- testing "
                   "whether asymmetric per-area shrinkage succeeds where "
                   "AZ1c's/AZ4b's uniform tau cap failed")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b/AZ1c/AZ1d
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ0b/AZ1a/AZ1b/AZ1c/AZ1d
    max_lag           = 2     # same as AZ1a/AZ1b/AZ1c/AZ1d
    global_tau_sigma  = 1.5   # same population-scale prior as AZ1d's tau_sigma
    local_scale_beta  = 1.0   # HalfCauchy scale for the per-area multiplier

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.98,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag_horseshoe(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit,
                global_tau_sigma=self.global_tau_sigma,
                local_scale_beta=self.local_scale_beta, name='lag_P')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(z, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1f(DwellingModel):
    """
    AZ1d with P's likelihood changed from mean-mixing to a genuine
    marginalized mixture over lag categories (see
    _build_hierarchical_lag_marginalized/_build_planning_likelihood_marginalized_lag
    docstrings) -- same hierarchical lambda_weights construction (mu_logit,
    tau, raw_offset, same priors, same tau_sigma) as AZ1d, only the
    likelihood's functional form changes.

    Motivated directly by AZ1d's own deep-dive (docs/az-ess-diagnosis.md):
    4 areas show genuine, irreducible lag-category ambiguity (gaps
    0.11-0.60 nats between competing categories), and even after fixing
    the resolvable majority of flagged areas via 16 chains + informed init,
    lag_P_mu_logit stayed at r-hat 1.02-1.03 (mu_logit[0] actually got
    WORSE, 1.02->1.03) -- i.e. more chains alone cannot get the shared
    population-level lag hyperparameters under 1.01 while genuine
    per-area ambiguity persists under the CURRENT mean-mixing likelihood.

    The hypothesis this tests: mean-mixing forces every observation to be
    scored against P_mean = sum_k lambda_k * z_shifted_k, a single blended
    value that no individual lag category actually predicts -- a known
    modelling anti-pattern for representing genuine discrete-category
    uncertainty (each planning completion really does have ONE true lag,
    we just don't know which; it isn't literally smeared across years).
    Marginalizing at the likelihood instead lets an honestly-ambiguous
    area's lambda_weights sit anywhere in the simplex while every
    observation is still scored against whichever category actually fits
    it, weighted by that category's plausibility -- removing the pressure
    toward a hard vertex commitment that mean-mixing creates. NOT assumed
    to work -- this is the direct empirical test of that argument, and a
    documented toy check (chat transcript, not reproduced here) found the
    simple "valley in lambda" framing does not hold up as a general claim,
    so the honest generative-story argument (mixture of densities vs a
    blended mean) is the actual basis for trying this, not a proven
    geometric mechanism.
    """

    name        = 'AZ1f'
    description = ("AZ1d with P's lag likelihood marginalized (genuine "
                   "mixture over lag categories via logsumexp) instead of "
                   "mean-mixed -- testing whether this removes the residual "
                   "lag_P_mu_logit/tau r-hat problem that persisted even "
                   "after 16 chains + informed init")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_tau']

    sigma_delta_floor = 3.0   # same as AZ1d
    k_sigma_delta     = 0.08  # same as AZ1d
    max_lag           = 2     # same as AZ1d
    tau_sigma         = 1.5   # same as AZ1d

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            lambda_weights, shifted = _build_hierarchical_lag_marginalized(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_P')

            _build_planning_likelihood_marginalized_lag(
                shifted, lambda_weights, data['P_obs'],
                self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(z, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1g(DwellingModel):
    """
    AZ1d + a REGULARIZED (slab-capped) horseshoe for P's per-area tau -- see
    _build_hierarchical_lag_regularized_horseshoe's docstring for the full
    construction and motivation.

    Direct follow-through on AZ1e's own docstring, which flagged (but deliberately
    didn't build) a regularized horseshoe as the standard literature fix for the exact
    failure AZ1e hit: `local_scale` (unregularized HalfCauchy) exploring to 3.09 million,
    producing this whole investigation's only divergences (17, scattered) rather than
    cleanly fixing AZ1d's residual r-hat/ESS. This is that follow-through, motivated by a
    much stronger case than AZ1e had: this session's own direct check
    (results/scratch/az1d_leakage_mechanism.py) found a chain's own lag_P_tau draw
    correlates r=0.85-0.98 with how far that SAME chain's flagged-area logits sit from
    their own mean, confirmed on TWO independently-sampled 200-area subsets (Islington,
    the original; Croydon, a fresh resample -- results/scratch/az1d_altsample_generalization.py)
    -- tau's between-chain disagreement isn't a diffuse hierarchical-pooling side-effect,
    it is mechanically the disagreement about which lag-category mode the same ~6-6.5%
    minority of large-|D| areas landed in, on both samples tested. AZ1c's/AZ4b's uniform tau
    cap already failed twice at constraining this minority's freedom directly; this instead
    gives that minority its own escape valve (a per-area local multiplier) while bounding
    it with a slab so the escape valve itself can't reproduce AZ1e's blowup.
    """

    name        = 'AZ1g'
    description = ("AZ1d + a regularized (slab-capped) horseshoe for P's "
                   "per-area lag tau -- AZ1e's fix idea, redone with a slab "
                   "bounding the local multiplier so it can't reproduce "
                   "AZ1e's local_scale blowup to 3.09 million")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau']

    sigma_delta_floor = 3.0   # same as AZ1d
    k_sigma_delta     = 0.08  # same as AZ1d
    max_lag           = 2     # same as AZ1d
    global_tau_sigma  = 1.5   # same population-scale prior as AZ1d's tau_sigma
    local_scale_beta  = 1.0   # same HalfCauchy scale as AZ1e, unchanged deliberately
    slab_scale        = 10.0 # see _build_hierarchical_lag_regularized_horseshoe's docstring

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.98,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag_regularized_horseshoe(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit,
                global_tau_sigma=self.global_tau_sigma,
                local_scale_beta=self.local_scale_beta,
                slab_scale=self.slab_scale, name='lag_P')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(z, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ1h(DwellingModel):
    """
    AZ1g moved onto the CANONICAL regularized-horseshoe recipe (see
    `_build_hierarchical_lag_regularized_horseshoe_v2`'s docstring) — a sampled slab
    (`c2 ~ InverseGamma(2, slab_c2_beta)`) instead of AZ1g's fixed `slab_scale=10.0`, and a
    `global_tau` prior derived from this model's own expected sparsity via the reference
    formula `tau0 = p0/(n_areas-p0)/sqrt(n_years)` instead of reusing AZ1d's unrelated
    `tau_sigma=1.5`.

    Motivated directly by AZ1g's own follow-up: `check-multimodality` confirmed AZ1g's
    residual bad-r-hat scalars (`lag_P_global_tau`, `lag_P_mu_logit`, and via them
    `sigma_ben`/`sigma_plan`) are `not_multimodal` on BOTH the Islington and Croydon 200-area
    samples tested — a mild, generic "shallow basin" (clean autocorrelation, smooth per-chain
    rank-histogram tilt, no discrete cluster split), not multimodality leaking upward (that
    channel is what AZ1g itself already fixed). Re-reading this codebase's own `pymc-modeling`/
    `pymc-extras` skill references for the standard treatment of this exact signature
    surfaced two concrete, textbook corrections AZ1g's first pass didn't yet apply — this is
    that follow-through, not a new idea: `troubleshooting.md`'s Identifiability Issues section
    names this symptom directly ("R-hat > 1.01 despite long chains, different chains
    converging to different solutions"), and `references/r2d2_horseshoe.md`'s own regularized-
    horseshoe implementation samples its slab and derives its global scale from expected
    sparsity rather than fixing either by hand — both of which AZ1g's first pass skipped.
    """

    name        = 'AZ1h'
    description = ("AZ1g moved onto the canonical Piironen-Vehtari regularized horseshoe: "
                   "sampled slab (c2) instead of a fixed slab_scale, and a sparsity-"
                   "calibrated global_tau prior (tau0 formula) instead of AZ1d's reused "
                   "tau_sigma -- testing whether the standard recipe closes AZ1g's residual "
                   "not_multimodal shallow-basin r-hat on global_tau/mu_logit")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau', 'lag_P_c2']

    sigma_delta_floor = 3.0    # same as AZ1d/AZ1g
    k_sigma_delta     = 0.08   # same as AZ1d/AZ1g
    max_lag           = 2      # same as AZ1d/AZ1g
    p0                = 15     # expected # of genuinely-divergent areas -- see docstring
    local_scale_beta  = 1.0    # same HalfCauchy scale as AZ1e/AZ1g, unchanged deliberately
    slab_c2_beta      = 100.0  # InverseGamma(2, .) prior mean == AZ1g's old fixed slab_scale^2

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.99,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            _, P_mean = _build_hierarchical_lag_regularized_horseshoe_v2(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, p0=self.p0,
                local_scale_beta=self.local_scale_beta,
                slab_c2_beta=self.slab_c2_beta, name='lag_P')

            _build_planning_likelihood_simple(P_mean, data['P_obs'],
                                             self.nu_obs, sigma_plan, name='P_like')
            _build_planning_likelihood_simple(z, data['E_obs'],
                                             self.nu_obs, sigma_ben, name='E_like')

        self.model = model
        return model


class AZ2(DwellingModel):
    """
    AZ0a + a single top-D-quartile boost to sigma_delta (see
    _build_zero_sum_z_prior_top_boost's docstring for the full
    construction and motivation) -- "Option 2" from
    docs/az-family-work-plan.md, adopted after a full 4-band hierarchy
    (Attempts 1-2, see that doc and _build_zero_sum_z_prior_banded's
    docstring) was shown, on real data, to be the wrong level of
    complexity: only the top quartile's band ever picked up real
    "excess" flexibility; the bottom 3 bands collapsed to an
    undifferentiated floor-only fit and made those areas WORSE than
    AZ0a's baseline (frac_flat_despite_active by band: 45.7% / 43.4% /
    27.5% / 0.0%). This construction is the direct, minimal-risk
    consequence of that finding: keep AZ0a's already-validated
    floor + k*|D| formula completely unchanged for every area, and add
    exactly one new sampled scalar that only applies to the top quartile.

    Deliberately an INDEPENDENT branch off AZ0a, not layered on AZ1a/AZ1b's
    lag work (per this round's "one specific addition at a time" rule) --
    mechanistically distinct from the lag work too: this targets how much
    a *given* year's deviation from the census-implied mean is allowed to
    be, not which year gets credited for a spike. Compare against AZ0a
    directly; combining with whichever lag variant is chosen is a later,
    separate step (Phase 4 in docs/az-family-work-plan.md).
    """

    name        = 'AZ2'
    description = ('AZ0a + a single top-D-quartile boost to sigma_delta '
                   '(replacing the 4-band hierarchy that over-complicated this)')
    var_names   = ['sigma_plan', 'sigma_ben', 'sigma_delta_top_boost']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ1a/AZ1b
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ1a/AZ1b
    top_quantile      = 0.75  # top quartile of |D| gets the extra boost term

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        # Fixed numpy, computed once from data, not sampled (same
        # convention as mu_area/sigma_delta elsewhere).
        top_cutoff = np.quantile(np.abs(D), self.top_quantile)
        is_top     = np.abs(D) >= top_cutoff

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, _, z = _build_zero_sum_z_prior_top_boost(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta, is_top=is_top)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                            sigma_plan, sigma_ben)

        self.model = model
        return model


class AZ2b(DwellingModel):
    """
    AZ2 + a smooth (sigmoid-ramp) top-boost weight instead of a hard
    top-quartile step (see _build_zero_sum_z_prior_top_boost_smooth's
    docstring for the full construction and motivation) -- a fix attempt
    for AZ2's residual low bulk ESS (docs/az-ess-diagnosis.md).

    AZ2's diagnose-table min ESS (bulk=47, on sigma_delta_top_boost) was
    traced to a small, persistent BETWEEN-chain disagreement rather than
    slow mixing or a simple pairwise ridge among the three named scalars
    (all checked directly, see docs/az-family-work-plan.md Phase 2's
    follow-up investigation). Untested candidate mechanism: `is_top` is a
    hard 0/1 step exactly at the 75th |D| percentile -- areas straddling
    that boundary see sigma_delta respond discontinuously to top_boost,
    which is exactly the kind of sharp threshold that can produce a
    shallow, hard-to-mix ridge. This model replaces the step with a
    smooth logistic ramp over |D|'s rank percentile, adding no new
    sampled parameters (still exactly one extra scalar, top_boost) --
    isolates whether the DISCONTINUITY itself, not the boost mechanism in
    general, was the source of the bulk-ESS problem.

    Otherwise identical to AZ2: same floor/k, same top_quantile,
    same sigma_plan/sigma_ben, same observation likelihoods.

    Branch verdict (docs/az-ess-diagnosis.md): the smooth ramp fixed
    sigma_delta_top_boost's own ESS (47 -> 800+) as intended, but a full
    whole-model scan (not just this class's own var_names) found the same
    underlying pathology had resurfaced on sigma_plan/sigma_ben instead of
    being resolved -- confirmed AZ2b-specific (not P_obs/E_obs data
    sparsity: AZ0a converges cleanly, ESS 3000+, on the identical data with
    no top-boost mechanism) via a causal AZ0a-vs-AZ2b comparison, but no
    single clean mechanism (leakage from ambiguous z areas, a pairwise
    scalar ridge, or a top-quartile residual split) explains HOW -- all
    three checked directly and came back weak/inconclusive. `sample_kwargs`
    below (8 chains, target_accept=0.97) is the best config found across
    four sampler-setting permutations tested: it reliably fixes sigma_ben
    (ESS 570, r_hat 1.015) but sigma_plan remains a disclosed, accepted
    limitation of this branch (ESS 234-534 depending on config, never
    reliably >=400) -- not fixable by sampler tuning alone.
    """

    name        = 'AZ2b'
    description = ('AZ2 + a smooth sigmoid ramp (over |D| rank percentile) '
                   'replacing the hard top-quartile step, to test whether '
                   'the discontinuity was driving the low bulk ESS')
    var_names   = ['sigma_plan', 'sigma_ben', 'sigma_delta_top_boost']

    sigma_delta_floor = 3.0   # same as AZ0/AZ0a/AZ1a/AZ1b/AZ2
    k_sigma_delta     = 0.08  # same as AZ0/AZ0a/AZ1a/AZ1b/AZ2
    top_quantile      = 0.75  # same centre as AZ2's hard cutoff
    transition_width  = 0.08  # see builder docstring -- ~16 areas' worth of rank

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.97,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, _, z, _ = _build_zero_sum_z_prior_top_boost_smooth(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta,
                top_quantile=self.top_quantile,
                transition_width=self.transition_width)

            sigma_plan = pm.HalfNormal('sigma_plan', sigma=2)
            sigma_ben  = pm.HalfNormal('sigma_ben',  sigma=2)

            self.add_observation_likelihoods(z, data['P_obs'], data['E_obs'],
                                            sigma_plan, sigma_ben)

        self.model = model
        return model


class AZ3(DwellingModel):
    """
    AZ0a + a per-cell noise/outlier mixture on both P and E (see
    _build_noise_mixture_likelihood's docstring for the full construction
    and the AZ0-collapse history this is built to avoid repeating).

    Deliberately an INDEPENDENT branch off AZ0a, not layered on AZ1a/AZ1b's
    lag work or AZ2's top-boost (per this round's "one specific addition
    at a time" rule). Sequenced last among the single-change branches on
    purpose (see docs/az-family-work-plan.md Phase 3): some cells that
    currently look "impossible to reconcile" might actually be lag or
    scale problems that Phases 1b/2 would have explained away, and the
    noise flag should be reserved for cells nothing else can fix. Purpose
    is explicitly dual: (1) let sigma_plan/sigma_ben stop being dragged
    loose by genuinely irreconcilable cells (AZ0a's sigma_plan=7.45 has to
    be loose enough, via StudentT's heavy tails, to "tolerate" even
    impossible cells like E01001774's -- a residual of 762 only costs
    ~20 nats there, cheap), and (2) expose resp_noise_P/E as a
    stakeholder-facing "flagged for investigation" signal (see
    diagnostics.posterior_outlier_summary), extending outliers.py's
    raw-data threshold-based flagging with a model-informed one that
    accounts for what z can and can't explain.

    sigma_noise_floor=25.0: well above AZ0a's converged sigma_plan/
    sigma_ben (~7-9), so the noise branch is unambiguously more tolerant
    than the signal branch rather than a near-duplicate of it. This is a
    first-pass value (same status as sigma_delta's floor/k in
    _build_zero_sum_z_prior) -- revisit if diagnostics show the noise
    branch is too eager (flagging cells a human would call genuine
    signal) or not eager enough (still failing to catch clear cases like
    E01001774).

    sigma_obs_floor=2.0 -- ADDED after diagnosing the root cause of AZ3's
    ESS problem (see docs/az-family-work-plan.md Phase 3). Without a
    floor, sigma_plan/sigma_ben (originally plain HalfNormal(2), same as
    AZ0a) collapsed to ~0.58/0.99 -- an order of magnitude below every
    other AZ0a-family model's ~7-9 -- because the noise-mixture lets
    outliers escape via the noise branch instead of inflating the signal
    branch's scale. That collapse created classic small-scale funnel
    geometry (confirmed via autocorrelation: sigma_plan's ACF was still
    0.4-0.7 at lag 20, only reaching ~0 by lag 100-200 -- the worst ESS
    in the model, bulk=43). A HalfNormal ALONE is not a floor -- its mode
    is at 0 regardless of its sigma parameter, the same mistake that
    broke AZ0's noise branch originally. floor + HalfNormal(3) mirrors
    the sigma_noise fix exactly, and 2.0 is chosen to sit below AZ0a's
    validated ~7-9 (still allowing the signal branch to tighten somewhat,
    which is the whole point of adding a noise branch) while ruling out
    the collapse-to-near-zero regime that caused the funnel.

    sample_kwargs bumps target_accept to 0.95, matching every other model
    in this codebase that introduced new mixture structure (AZ0, AZ0b,
    AZ1b) -- new mixture geometry, same precautionary bump.
    """

    name        = 'AZ3'
    description = ('AZ0a + floored noise/outlier mixture on P and E, '
                   'with automatic per-cell outlier flagging')
    var_names   = ['sigma_plan', 'sigma_ben', 'rho_P', 'rho_E',
                   'sigma_noise_P', 'sigma_noise_E']

    sigma_delta_floor = 3.0    # same as AZ0/AZ0a/AZ1a/AZ1b/AZ2
    k_sigma_delta     = 0.08   # same as AZ0/AZ0a/AZ1a/AZ1b/AZ2
    sigma_noise_floor = 25.0   # see docstring -- well above sigma_plan/ben's
                               # converged ~7-9, so the noise branch is
                               # unambiguously more tolerant, not a near-duplicate
    sigma_obs_floor   = 2.0    # see docstring -- fixes sigma_plan/ben's
                               # collapse-to-near-zero funnel

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            sigma_plan_excess = pm.HalfNormal('sigma_plan_excess', sigma=3)
            sigma_ben_excess  = pm.HalfNormal('sigma_ben_excess',  sigma=3)
            sigma_plan = pm.Deterministic('sigma_plan', self.sigma_obs_floor + sigma_plan_excess)
            sigma_ben  = pm.Deterministic('sigma_ben',  self.sigma_obs_floor + sigma_ben_excess)

            _build_noise_mixture_likelihood(
                z, data['P_obs'], sigma_plan, self.sigma_noise_floor, self.nu_obs, name='P')
            _build_noise_mixture_likelihood(
                z, data['E_obs'], sigma_ben, self.sigma_noise_floor, self.nu_obs, name='E')

        self.model = model
        return model


class AZ4(DwellingModel):
    """
    Phase 4 — combines the three independently-validated AZ-family pieces
    into one model, on top of AZ0a's base (docs/az-family-work-plan.md
    Phase 4). Each piece was deliberately validated as its OWN single-
    change branch off AZ0a first, specifically so any problem in the
    combination could be traced to a known interaction rather than a
    tangle of unknowns (see docs/az-ess-diagnosis.md for the most recent
    round of that validation work, including which variant of each piece
    won):

    1. **z-prior**: AZ2b's smooth top-D-quartile boost
       (`_build_zero_sum_z_prior_top_boost_smooth`), NOT AZ2's original
       hard-cutoff version -- AZ2b measurably improved bulk ESS on
       `sigma_delta_top_boost` (47 -> 605) with no cost to
       frac_flat_despite_active, so it is strictly the better piece to
       carry forward (docs/az-ess-diagnosis.md).
    2. **Lag structure**: AZ1b's area-hierarchical lag
       (`_build_hierarchical_lag`, `tau_sigma=1.5`), NOT AZ1c's
       tau-capped variant -- AZ1c was tried specifically to fix AZ1b's
       residual multimodality and made both max r-hat and min ESS worse
       (docs/az-ess-diagnosis.md), so AZ1b's original "accept and report
       via 8 chains" treatment is what's carried forward here, applied
       to BOTH P and E (mirroring AZ1b itself, not AZ1a's fully-pooled
       version, for the same lambda_weights_E-bimodality reason AZ1b
       existed in the first place).
    3. **Likelihood**: AZ3's floored noise-mixture
       (`_build_noise_mixture_likelihood`) with the floored
       `sigma_plan`/`sigma_ben` fix (`sigma_obs_floor=2.0 +
       HalfNormal(3)` excess) validated in the `az3-floor-followup`
       artifact -- applied to the LAG-CONVOLVED `P_mean`/`E_mean` (this
       model's per-cell "genuine signal" mean), not to raw `z` directly,
       since the signal branch should still benefit from lag-awareness;
       AZ3 itself had no lag structure to convolve, so this is a genuine
       novel composition, not a copy-paste of AZ3's call site.

    **Composition risk, flagged going in rather than discovered
    afterward**: AZ1b's residual hard multimodality and AZ3's per-cell
    signal/noise ambiguity are BOTH manifestations of the same underlying
    pattern -- sparse, sometimes-contradictory per-area/per-year data
    forcing a near-discrete choice between two roughly-equally-good
    explanations (which lag year explains a spike, vs whether a value is
    signal or noise). Combining them could compound rather than cancel:
    an area already uncertain about ITS lag category now also has that
    uncertainty propagating through a signal/noise gate on the SAME
    cells. Checked empirically after sampling, not assumed either way --
    see docs/az-ess-diagnosis.md's Phase 4 follow-up entry (added once
    this model has been sampled and diagnosed) for whether this
    materialised.

    chains=8, cores=8, target_accept=0.95 -- inherited from AZ1b/AZ1c's
    precedent (the hierarchical lag component alone already justified 8
    chains for a trustworthy mode-split estimate), applied here since
    that same component is present unchanged.
    """

    name        = 'AZ4'
    description = ('AZ0a + AZ2b\'s smooth top-boost z-prior + AZ1b\'s '
                   'area-hierarchical lag (P and E) + AZ3\'s floored '
                   'noise-mixture likelihood on the lag-convolved means')
    var_names   = ['sigma_plan', 'sigma_ben', 'sigma_delta_top_boost',
                   'lag_P_mu_logit', 'lag_P_tau', 'lag_E_mu_logit', 'lag_E_tau',
                   'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E']

    sigma_delta_floor = 3.0    # same as AZ0/AZ0a/.../AZ3
    k_sigma_delta     = 0.08   # same as AZ0/AZ0a/.../AZ3
    top_quantile      = 0.75   # same as AZ2/AZ2b
    transition_width  = 0.08   # same as AZ2b
    max_lag           = 2      # same as AZ1a/AZ1b/AZ1c
    tau_sigma         = 1.5    # AZ1b's value (NOT AZ1c's rejected hard cap)
    sigma_noise_floor = 25.0   # same as AZ3
    sigma_obs_floor   = 2.0    # same as AZ3 (floored sigma_plan/sigma_ben fix)

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, _, z, _ = _build_zero_sum_z_prior_top_boost_smooth(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta,
                top_quantile=self.top_quantile,
                transition_width=self.transition_width)

            _, P_mean = _build_hierarchical_lag(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_P')
            _, E_mean = _build_hierarchical_lag(
                z, pre_inference_E, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_sigma=self.tau_sigma, name='lag_E')

            sigma_plan_excess = pm.HalfNormal('sigma_plan_excess', sigma=3)
            sigma_ben_excess  = pm.HalfNormal('sigma_ben_excess',  sigma=3)
            sigma_plan = pm.Deterministic('sigma_plan', self.sigma_obs_floor + sigma_plan_excess)
            sigma_ben  = pm.Deterministic('sigma_ben',  self.sigma_obs_floor + sigma_ben_excess)

            _build_noise_mixture_likelihood(
                P_mean, data['P_obs'], sigma_plan, self.sigma_noise_floor, self.nu_obs, name='P')
            _build_noise_mixture_likelihood(
                E_mean, data['E_obs'], sigma_ben, self.sigma_noise_floor, self.nu_obs, name='E')

        self.model = model
        return model


class AZ5(DwellingModel):
    """
    AZ1g + AZ3 combined -- the best-validated AZ1 branch (P-only regularized-horseshoe
    hierarchical lag, `_build_hierarchical_lag_regularized_horseshoe`) plus AZ3's floored
    noise/outlier mixture likelihood, on top of AZ0a's base. A deliberately narrower
    combination than AZ4/AZ4b: exactly two validated pieces, not three, following this
    round's "combine at most two at once" rule -- AZ2b's smooth top-boost z-prior is NOT
    included here, so any regression can be attributed to the AZ1g/AZ3 interaction alone,
    not tangled with a third simultaneous change.

    Why AZ1g specifically, not AZ1b/AZ1d/AZ1h: AZ1g is the one AZ1-branch model with a
    real, mechanistically-confirmed fix (regularized/slab-capped horseshoe) for the
    per-area lag leakage problem that AZ1d first isolated, checked on two independently-
    sampled 200-area datasets (see `docs/az-ess-diagnosis.md`'s AZ1g section) -- AZ1h's
    attempt to move it onto the fully-canonical recipe (sampled slab, sparsity-calibrated
    tau0) was tried and rejected (max r-hat 1.02->1.96), so AZ1g's simpler fixed-slab
    design is what's carried forward, not AZ1h. Like AZ1d/AZ1g, only P gets a lag
    mechanism -- E is compared same-year against z directly, since every AZ1-branch
    diagnosis this round found E's lag category disproportionately unstable and never
    produced a validated fix for it (AZ1a's clean bimodality, AZ4's causal-chain finding
    that `sigma_noise_E` specifically collapsed once lag-aware).

    Composition, mirroring AZ4's precedent: AZ3's floored noise-mixture applies to the
    LAG-CONVOLVED `P_mean` (AZ1g's per-cell "genuine signal" mean) for P, but to raw `z`
    directly for E, since E has no lag-convolved mean here to use instead -- the same
    "signal branch should still benefit from lag-awareness where lag-awareness exists"
    principle AZ4 established, just asymmetric because only one source has that structure
    this time. `sigma_plan`/`sigma_ben` use AZ3's floored construction
    (`sigma_obs_floor=2.0 + HalfNormal(3)` excess), not AZ1g's plain `HalfNormal(2)` --
    AZ1g never carried the floor fix because it predates AZ3's own diagnosis of the
    collapse-to-near-zero funnel; skipping it here would just reintroduce a known,
    already-fixed failure mode into a new model.

    Composition risk, flagged going in per AZ4's own precedent (checked empirically once
    sampled, not assumed): the regularized horseshoe already gives P's flagged-minority
    areas their own escape valve, and AZ3's noise mixture gives every cell a second,
    independent escape valve (call it noise instead) -- these two mechanisms compete for
    the same ambiguous large-|D| cells, unlike AZ4's composition where the leakage ran
    through a completely different piece (`sigma_delta_top_boost`) that has no analogue
    here. Whether that competition helps (each mechanism absorbs a distinct part of the
    ambiguity) or compounds (the same areas get doubly-flexible, unidentifiably split
    between "different lag" and "just noise") is the open empirical question this model
    answers.

    target_accept=0.98, chains=8, cores=8 -- AZ1g's own settings carried forward
    unchanged (not relaxed to AZ4's 0.95/AZ3's default), since the horseshoe geometry
    that required them doesn't go away when combined with a second likelihood piece, and
    if anything the added noise-mixture mode competition argues for keeping the more
    conservative sampler settings, not loosening them.
    """

    name        = 'AZ5'
    description = ("AZ1g's P-only regularized-horseshoe hierarchical lag + AZ3's "
                   "floored noise-mixture likelihood -- a deliberate two-piece "
                   "combination (not three, unlike AZ4/AZ4b), isolating the "
                   "AZ1g/AZ3 interaction specifically")
    var_names   = ['sigma_plan', 'sigma_ben', 'lag_P_mu_logit', 'lag_P_global_tau',
                   'rho_P', 'rho_E', 'sigma_noise_P', 'sigma_noise_E']

    sigma_delta_floor = 3.0    # same as AZ0/AZ0a/.../AZ1g/AZ3
    k_sigma_delta     = 0.08   # same as AZ0/AZ0a/.../AZ1g/AZ3
    max_lag           = 2      # same as AZ1a-AZ1h
    global_tau_sigma  = 1.5    # same as AZ1g
    local_scale_beta  = 1.0    # same as AZ1g
    slab_scale        = 10.0   # same as AZ1g
    sigma_noise_floor = 25.0   # same as AZ3
    sigma_obs_floor   = 2.0    # same as AZ3 (floored sigma_plan/sigma_ben fix)

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.98,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, z = _build_zero_sum_z_prior(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta)

            _, P_mean = _build_hierarchical_lag_regularized_horseshoe(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit,
                global_tau_sigma=self.global_tau_sigma,
                local_scale_beta=self.local_scale_beta,
                slab_scale=self.slab_scale, name='lag_P')

            sigma_plan_excess = pm.HalfNormal('sigma_plan_excess', sigma=3)
            sigma_ben_excess  = pm.HalfNormal('sigma_ben_excess',  sigma=3)
            sigma_plan = pm.Deterministic('sigma_plan', self.sigma_obs_floor + sigma_plan_excess)
            sigma_ben  = pm.Deterministic('sigma_ben',  self.sigma_obs_floor + sigma_ben_excess)

            _build_noise_mixture_likelihood(
                P_mean, data['P_obs'], sigma_plan, self.sigma_noise_floor, self.nu_obs, name='P')
            _build_noise_mixture_likelihood(
                z, data['E_obs'], sigma_ben, self.sigma_noise_floor, self.nu_obs, name='E')

        self.model = model
        return model


class AZ4b(DwellingModel):
    """
    AZ4 + a hard cap on tau (`_build_hierarchical_lag_capped`, same
    `tau_cap=1.5` as AZ1c), testing whether the fix that LOST on its own
    turf is nonetheless a net win here (docs/az-ess-diagnosis.md's Phase 4
    follow-up). AZ1c capped tau within AZ1b alone and made AZ1b's OWN
    r-hat/ESS worse (max r-hat 1.19->1.32, min ESS 29->19) -- but AZ4's
    problem is qualitatively different from "AZ1b's own convergence":
    it's genuine per-area lag ambiguity LEAKING into `sigma_noise_E` and
    `sigma_delta_top_boost`, two scalars that were completely clean in
    isolation (AZ3's sigma_noise_E: ESS>3600; AZ2b's top_boost: ESS 605)
    and collapsed to ESS 33/40 respectively once combined -- traced
    directly (see the az4-diagnostics artifact) to 96% of z-unstable
    areas and 92% of top-quartile-D areas being lag-ambiguous. Capping
    how far any one area's lag weights can diverge from the population
    kernel is aimed exactly at that leakage channel, not at AZ1b's own
    r-hat number -- worth testing on its own terms rather than assuming
    AZ1c's verdict transfers unchanged, since "does this help the shared
    scalars downstream" is a different question from "does this help the
    lag weights themselves."

    Expected, documented tradeoff carried over from AZ1c: capping tau
    should cost some of AZ1b's per-area flexibility (though AZ1c's own
    check found this cost didn't show up in E01033711's spike-tracking
    specifically) — checked here again for AZ4's own flagship cases, not
    assumed to replicate.

    Otherwise identical to AZ4: same z-prior, same P_mean/E_mean
    construction, same floored noise-mixture likelihood.
    """

    name        = 'AZ4b'
    description = ('AZ4 + a hard ceiling on tau (tau_cap=1.5, as AZ1c) -- '
                   'testing whether capping the leakage channel into '
                   'sigma_noise_E/top_boost is a net win even though the '
                   'same fix lost on its own turf in AZ1c')
    var_names   = AZ4.var_names

    sigma_delta_floor = AZ4.sigma_delta_floor
    k_sigma_delta     = AZ4.k_sigma_delta
    top_quantile      = AZ4.top_quantile
    transition_width  = AZ4.transition_width
    max_lag           = AZ4.max_lag
    tau_cap           = 1.5    # same as AZ1c
    sigma_noise_floor = AZ4.sigma_noise_floor
    sigma_obs_floor   = AZ4.sigma_obs_floor

    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, 'target_accept': 0.95,
                     'chains': 8, 'cores': 8}

    def build(self):
        data, n_areas, n_years, D, _ = self._build_context()

        pre_inference_P = _build_pre_inference(data, self.max_lag, source='P')
        pre_inference_E = _build_pre_inference(data, self.max_lag, source='E')
        prior_logit = np.log(self.lag_alpha[1:] / self.lag_alpha[0])

        with pm.Model(coords=self._default_coords()) as model:

            _, _, _, _, z, _ = _build_zero_sum_z_prior_top_boost_smooth(
                D, n_areas, n_years,
                floor=self.sigma_delta_floor, k=self.k_sigma_delta,
                top_quantile=self.top_quantile,
                transition_width=self.transition_width)

            _, P_mean = _build_hierarchical_lag_capped(
                z, pre_inference_P, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_cap=self.tau_cap, name='lag_P')
            _, E_mean = _build_hierarchical_lag_capped(
                z, pre_inference_E, n_areas, n_years, self.n_lags,
                self.max_lag, prior_logit, tau_cap=self.tau_cap, name='lag_E')

            sigma_plan_excess = pm.HalfNormal('sigma_plan_excess', sigma=3)
            sigma_ben_excess  = pm.HalfNormal('sigma_ben_excess',  sigma=3)
            sigma_plan = pm.Deterministic('sigma_plan', self.sigma_obs_floor + sigma_plan_excess)
            sigma_ben  = pm.Deterministic('sigma_ben',  self.sigma_obs_floor + sigma_ben_excess)

            _build_noise_mixture_likelihood(
                P_mean, data['P_obs'], sigma_plan, self.sigma_noise_floor, self.nu_obs, name='P')
            _build_noise_mixture_likelihood(
                E_mean, data['E_obs'], sigma_ben, self.sigma_noise_floor, self.nu_obs, name='E')

        self.model = model
        return model


# ── Model registry (single source of truth) ────────────────────────────────
#
# Every implemented model class is listed here exactly once. Adding a new
# model means adding it to this list — housing_projections.models re-exports
# it automatically, and the CLI / notebooks discover it through ALL_MODELS
# rather than needing their own copy of this list.
#
# NOTE: this is a plain dict literal, not a `{cls.name: cls for cls in [...]}`
# comprehension — mypy resolves `name` (an abstract property on DwellingModel,
# overridden as a plain str attribute on each subclass) to `str` per literal
# class reference, but widens it to the unbound property getter once the
# classes are joined into a single iterable, which breaks every `ALL_MODELS
# [name]` lookup downstream.
ALL_MODELS: dict[str, type[DwellingModel]] = {
    M0.name: M0, M0h.name: M0h, M1.name: M1, M1h.name: M1h,
    M5.name: M5, M6.name: M6, M7.name: M7, M8.name: M8,
    M9.name: M9, M10.name: M10, M11.name: M11, M12.name: M12,
    M13.name: M13, M14.name: M14, M15.name: M15, M16.name: M16,
    AZ0.name: AZ0, AZ0a.name: AZ0a, AZ0b.name: AZ0b,
    AZ1a.name: AZ1a, AZ1b.name: AZ1b, AZ1c.name: AZ1c,
    AZ1d.name: AZ1d, AZ1e.name: AZ1e, AZ1f.name: AZ1f, AZ1g.name: AZ1g, AZ1h.name: AZ1h,
    AZ2.name: AZ2, AZ2b.name: AZ2b, AZ3.name: AZ3,
    AZ4.name: AZ4, AZ4b.name: AZ4b, AZ5.name: AZ5,
}

__all__ = ["ALL_MODELS", *ALL_MODELS]
