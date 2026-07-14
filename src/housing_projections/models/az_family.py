"""
AZ-family model classes (AZ0-AZ4b).

A separate progression from the M-family, branching off a single validated baseline, AZ0a
(anchored zero-sum census-conditioned z prior + plain same-year StudentT P/E likelihoods) —
see CLAUDE.md's "What this is" section and docs/az-family-work-plan.md for the full per-model
narrative and accept/reject decisions. Split out of the single `models.py` file (which had
grown to ~5,000 lines) purely for navigability; no behavioural change. See `m_family.py` for
the separate M0-M16 progression and `builders.py` for the shared construction pieces both
families use.
"""
import numpy as np
import pymc as pm

from housing_projections.config import DEFAULT_SAMPLE_KWARGS

from .base import DwellingModel
from .builders import (
    _build_backward_reallocation_likelihood,
    _build_backward_reallocation_likelihood_2way,
    _build_hierarchical_lag,
    _build_hierarchical_lag_capped,
    _build_hierarchical_lag_horseshoe,
    _build_hierarchical_lag_marginalized,
    _build_hierarchical_lag_regularized_horseshoe,
    _build_hierarchical_lag_regularized_horseshoe_v2,
    _build_lag,
    _build_noise_mixture_likelihood,
    _build_planning_likelihood_marginalized_lag,
    _build_planning_likelihood_simple,
    _build_pre_inference,
    _build_zero_sum_z_prior,
    _build_zero_sum_z_prior_top_boost,
    _build_zero_sum_z_prior_top_boost_smooth,
)

__all__ = [
    "AZ0", "AZ0a", "AZ0b",
    "AZ1a", "AZ1b", "AZ1c", "AZ1d", "AZ1e", "AZ1f", "AZ1g", "AZ1h",
    "AZ2", "AZ2b", "AZ3", "AZ4", "AZ4b", "AZ5",
]

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
