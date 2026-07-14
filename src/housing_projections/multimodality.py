"""
Multimodality diagnostic pipeline.

A dedicated, manually-run pipeline (see `housing-projections check-multimodality`) for the
specific question that came up repeatedly diagnosing the AZ1 family (docs/az-ess-diagnosis.md,
docs/ess-rhat-diagnostic-guide.md Pattern 2, docs/multimodality-characterization-guide.md):
when a model has bad r-hat/ESS on a per-area hierarchical lag-category simplex, how much of
that is genuine, expected multimodality (not a defect) versus a real, fixable sampling
problem versus something that still needs a manual deep-dive?

This module assembles four checks already used ad hoc, repeatedly, by hand this session into
one reusable classifier:
  - chain purity (existing `diagnostics.hierarchical_mode_summary`)
  - within-chain switch rate (NEW here — the check that caught AZ1d's E01035649 case, where
    a "7 vs 1 chains" purity-based split turned out to be every chain visiting both
    categories dozens of times, not a clean separation)
  - the log-likelihood-gap check (formalizes the ad hoc script used throughout
    az-ess-diagnosis.md into a reusable function, generalized to N categories/groups)
  - a validated fix (informed-init reseeding, `resolve_stuck_areas`) for the one category
    (stuck_fixable / Pattern 2b) that sampling harder can actually repair

See docs/multimodality-diagnostic-pipeline.md for the full worked walkthrough, category
definitions, and worked thresholds; docs/ess-rhat-diagnostic-guide.md §2 step 2 and
docs/multimodality-characterization-guide.md §0/Step 3 for the underlying methodology this
formalizes.
"""
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from housing_projections.diagnostics import diagnostics_summary, hierarchical_mode_summary

__all__ = [
    "ModeClassificationThresholds",
    "compute_switch_rates",
    "compute_loglik_gap",
    "compute_logp_gap",
    "classify_multimodality",
    "classify_scalar_multimodality",
    "resolve_stuck_areas",
    "verify_resolution",
    "lag_vars_in_trace",
    "derive_loglik_var",
    "adjusted_diagnostics_report",
    "multimodality_report",
    "adjusted_diagnostics_summary",
]


@dataclass(frozen=True)
class ModeClassificationThresholds:
    """
    The four thresholds that jointly decide a classify_multimodality/
    classify_scalar_multimodality category (see `_categorize`) — bundled as one object
    because they are only ever meaningful together, not because they're independently
    tunable knobs. All defaults were derived empirically from the AZ1 family this session,
    not from a universal statistical rule — see classify_multimodality's own docstring for
    the full calibration caveat; re-check before trusting them on a materially different
    model. `rhat_threshold` (which cells get classified at all) and `sigma_threshold`
    (scalar-only mode discovery) are deliberately NOT included here — both are used well
    beyond this classification step (shared with `hierarchical_mode_summary`/
    `diagnostics_summary`, or specific to `_discover_modes_from_chain_means`), so folding
    them in here would misrepresent them as part of this one decision.
    """
    purity_threshold: float = 0.95
    switch_rate_threshold: float = 0.02
    gap_tied_threshold: float = 2.0
    gap_decisive_threshold: float = 10.0


CAT_HARD_GENUINE    = "hard_genuine"
CAT_STUCK_FIXABLE   = "stuck_fixable"
CAT_ROUND_TRIPPING  = "round_tripping"
CAT_MIXED           = "mixed"
CAT_NEEDS_REVIEW    = "needs_review"
CAT_NOT_MULTIMODAL  = "not_multimodal"

_FIXABLE_CATEGORIES = (CAT_STUCK_FIXABLE,)  # only category resolve_stuck_areas will touch


def compute_switch_rates(trace, lag_var_name):
    """
    Per-(chain, area) rate of dominant-category switches across draws.

    dominant_category[c, d, a] = argmax(lambda_weights[c, d, a, :]) — the lag category
    that chain c's draw d currently favours for area a. A chain genuinely STUCK in one
    mode (hard multimodality, e.g. AZ1d's E01002702) has ~0 switches across its whole run;
    a chain that keeps crossing between categories (round-tripping, e.g. E01035649 — every
    one of 8 chains switched 26-55 times across 1500 draws, roughly evenly) has many. This
    is the cheap, direct check that a chain-purity-only view (majority vote per chain) can
    miss entirely — purity alone reported E01035649 as a clean "7 vs 1 chains" split, which
    the switch-rate check revealed was actually every chain wandering through both.

    Parameters
    ----------
    trace        : az.InferenceData with posterior[lag_var_name], shape
                   (chain, draw, area, category)
    lag_var_name : e.g. 'lag_P_lambda_weights'

    Returns
    -------
    (switch_rate, dominant) : switch_rate is np.ndarray (n_chains, n_areas) — n_switches /
    (n_draws - 1); dominant is the raw (chain, draw, area) argmax array, reused by
    compute_loglik_gap so it's only computed once per call site.
    """
    lw = trace.posterior[lag_var_name].values  # (chain, draw, area, category)
    dominant = lw.argmax(axis=-1)              # (chain, draw, area)
    n_transitions = dominant.shape[1] - 1
    if n_transitions <= 0:
        return np.zeros(dominant.shape[::2]), dominant
    switches = (dominant[:, 1:, :] != dominant[:, :-1, :]).sum(axis=1)  # (chain, area)
    return switches / n_transitions, dominant


def compute_loglik_gap(trace, dominant, area_idx, loglik_var_name, purity_threshold=0.95):
    """
    Log-likelihood-gap check (docs/az-ess-diagnosis.md), generalized to N groups.

    Splits chains into groups by their own per-chain dominant category, keeping only
    chains at or above `purity_threshold` (chains below it are "mixed" and excluded — they
    don't cleanly belong to any one group, which is itself informative and reported
    separately, not silently dropped). For each group, computes the mean total
    log-likelihood (summed over all non-chain/draw dims for this area, e.g. year) across
    that group's chains, then reports each group's gap to the single best (highest mean
    log-likelihood) group.

    A near-zero gap (<~2 nats) between two groups means the competing lag-category
    explanations are genuinely, statistically tied — real epistemic ambiguity no amount of
    sampling resolves. A large gap (>~10 nats) means that group's chains are simply stuck in
    a decisively worse mode — a sampling failure more/better sampling (via
    `resolve_stuck_areas`) can fix, not a genuine tie. See docs/ess-rhat-diagnostic-guide.md
    §4 for the full worked derivation and validated thresholds.

    Parameters
    ----------
    trace            : az.InferenceData with log_likelihood[loglik_var_name],
                       shape (chain, draw, area, ...) — any trailing dims (e.g. year)
                       are summed over per chain-draw.
    dominant         : (chain, draw, area) argmax array from compute_switch_rates
    area_idx         : positional index into the area dimension
    loglik_var_name  : e.g. 'P_like' — must be populated in trace.log_likelihood
                       (pm.compute_log_likelihood, or the Potential-based
                       '<name>_pointwise' convention — both already produce this
                       group via DwellingModel.sample())
    purity_threshold : chains below this purity are excluded as "mixed", not assigned
                       to any group (same convention as hierarchical_mode_summary)

    Returns
    -------
    dict with keys:
        best_category    : int or None (None if fewer than 2 pure groups exist)
        group_means      : {category: mean_total_loglik} for every pure group found
        gaps             : {category: best_group_mean - this_group_mean} for every
                           NON-best pure group (0.0 for the best group itself, omitted)
        n_mixed_chains   : chains excluded for falling below purity_threshold
    """
    n_chains = dominant.shape[0]
    loglik = trace.log_likelihood[loglik_var_name].values  # (chain, draw, area, ...)
    area_loglik = loglik[:, :, area_idx, ...]
    area_loglik = area_loglik.reshape(n_chains, area_loglik.shape[1], -1).sum(axis=-1)
    chain_mean_loglik = area_loglik.mean(axis=1)  # (n_chains,)

    groups: dict[int, list[float]] = {}
    n_mixed = 0
    for c in range(n_chains):
        cats, counts = np.unique(dominant[c, :, area_idx], return_counts=True)
        top = counts.argmax()
        purity = counts[top] / counts.sum()
        if purity < purity_threshold:
            n_mixed += 1
            continue
        cat = int(cats[top])
        groups.setdefault(cat, []).append(chain_mean_loglik[c])

    group_means = {cat: float(np.mean(vals)) for cat, vals in groups.items()}
    if len(group_means) < 2:
        return {"best_category": None, "group_means": group_means,
                "gaps": {}, "n_mixed_chains": n_mixed}

    best_category = max(group_means, key=group_means.get)
    best_mean = group_means[best_category]
    gaps = {cat: best_mean - mean for cat, mean in group_means.items() if cat != best_category}
    return {"best_category": best_category, "group_means": group_means,
            "gaps": gaps, "n_mixed_chains": n_mixed}


def _cluster_chain_means(chain_means, within_std, sigma_threshold=3.0):
    """
    Group chain means into clusters by standardized gap: sort the means, start a new
    cluster whenever the gap to the previous (sorted) value exceeds `sigma_threshold` times
    the pooled within-chain draw standard deviation — the same sigma-relative-to-noise
    principle `diagnostics.check_chain_agreement` already uses for its own trapped-chain
    check (`logp_gap_sigma`), applied here to chain MEANS instead of per-draw logp. A gap
    smaller than this threshold means the two candidate "modes" aren't cleanly separated
    relative to ordinary draw-to-draw noise, so they're treated as one cluster rather than
    two ambiguous ones — round_tripping/needs_review is a later, separate question from
    whether distinct modes exist at all.

    KDE peak-finding (tried first) was empirically unreliable at typical chain counts
    (4-16): scipy's default KDE bandwidth over-smooths an obvious cluster split into one
    lump, while a hand-tuned smaller bandwidth that DOES separate it also false-positives on
    pure unimodal noise and still misses a minority 1-vs-N chain split. This
    standardized-gap approach was checked against exactly those failure cases (an obvious
    N/2-vs-N/2 split, a 3-vs-1 minority split, pure unimodal noise, a linear/no-cluster
    spread, and a borderline gap-equals-noise-scale case) before being adopted — see
    docs/multimodality-diagnostic-pipeline.md.

    Returns a list of clusters, each a list of positional indices into `chain_means`.
    """
    order = np.argsort(chain_means)
    clusters = [[int(order[0])]]
    for i in range(1, len(order)):
        gap = chain_means[order[i]] - chain_means[order[i - 1]]
        if within_std > 0 and gap / within_std > sigma_threshold:
            clusters.append([int(order[i])])
        else:
            clusters[-1].append(int(order[i]))
    return clusters


def _discover_modes_from_chain_means(values, sigma_threshold=3.0):
    """
    Chain-level mode discovery for a non-categorical (chain, draw) variable cell —
    generalizes hierarchical_mode_summary's chain-purity idea to a variable with no known
    discrete categories to argmax over (e.g. a scalar hyperparameter like lag_E_mu_logit).
    Clusters each chain's own mean by standardized gap (`_cluster_chain_means`), relative to
    the pooled within-chain draw standard deviation — robust to within-chain
    autocorrelation/slow mixing, and cheap (n_chains points, not n_chains * n_draws).

    Returns the discovered mode LOCATIONS (np.ndarray, one per cluster — each cluster's own
    mean chain-mean), or an empty array if fewer than 2 clusters are found — i.e. this cell
    isn't multimodal in this sense at all; its bad r-hat is a real problem this pipeline
    doesn't explain (see CAT_NOT_MULTIMODAL).

    Calibration caveat, stated plainly (same discipline as classify_multimodality's own
    thresholds): `sigma_threshold=3.0` is a fresh starting point, checked only against the
    synthetic cases in `_cluster_chain_means`'s own docstring, not yet against a real
    multimodal scalar. Re-validate before trusting it on models beyond the one it's first
    checked against (see docs/multimodality-diagnostic-pipeline.md).
    """
    chain_means = values.mean(axis=1)
    within_std = values.std(axis=1).mean()
    clusters = _cluster_chain_means(chain_means, within_std, sigma_threshold=sigma_threshold)
    if len(clusters) < 2:
        return np.array([])
    return np.array([chain_means[idx].mean() for idx in clusters])


def _nearest_mode_labels(values, mode_locations):
    """
    (chain, draw) values -> (chain, draw) int labels, each the index of its nearest
    mode_location — the direct generalization of `lw.argmax(axis=-1)` (known categories) to
    "nearest empirically-discovered mode" for a continuous variable. Feeds the same
    downstream switch-rate/purity/gap logic classify_multimodality already uses.
    """
    diffs = np.abs(values[:, :, None] - mode_locations[None, None, :])  # (chain, draw, modes)
    return diffs.argmin(axis=-1)


def _switch_rate_from_dominant(dominant):
    """
    Per-chain rate of dominant-label switches across draws, for a plain (chain, draw) label
    array — the same transition-counting compute_switch_rates uses per area, generalized to
    a variable with no batch/area dimension (a scalar/vector cell is classified one at a
    time, so there's nothing to batch over here).

    Returns (n_chains,) switch rate array.
    """
    n_transitions = dominant.shape[1] - 1
    if n_transitions <= 0:
        return np.zeros(dominant.shape[0])
    switches = (dominant[:, 1:] != dominant[:, :-1]).sum(axis=1)
    return switches / n_transitions


def compute_logp_gap(trace, dominant, purity_threshold=0.95):
    """
    Generalization of compute_loglik_gap for a variable with no dedicated per-observation
    log-likelihood group: groups chains by their own dominant MODE label (from
    _nearest_mode_labels, not argmax over known categories) and compares each group's mean
    TOTAL log-posterior density (sample_stats.logp or .lp — already available for any
    already-sampled model, the same auto-detection check_chain_agreement already uses)
    instead of a named source's log-likelihood.

    Expressed in units of pooled within-chain logp STANDARD DEVIATION (mirroring
    check_chain_agreement's own `logp_gap_sigma` convention), NOT absolute nats: total model
    logp sums the log-density of every parameter and observation in the model, not just the
    ones relevant to the one cell being classified, so its scale is dominated by
    cell-irrelevant noise — the lambda_weights pipeline's fixed 2/10-nat tied/decisive
    thresholds would be meaningless applied to it directly.

    Parameters
    ----------
    trace            : az.InferenceData with sample_stats.logp or .lp
    dominant         : (chain, draw) int array — per-draw nearest-mode label for ONE cell
                       (unlike compute_loglik_gap's dominant, there's no area dim: a scalar
                       cell is classified one at a time, not batched across areas)
    purity_threshold : chains below this purity are excluded as "mixed" (same convention as
                       compute_loglik_gap/hierarchical_mode_summary)

    Returns
    -------
    dict, same shape as compute_loglik_gap: best_category, group_means, gaps (now in sigma
    units, not nats), n_mixed_chains. best_category is None (in addition to the <2 pure
    groups case) when the trace has no logp/lp in sample_stats at all — nothing to
    gap-check against.
    """
    # '/sample_stats' in trace.groups (not `in trace.sample_stats` directly) — accessing
    # .sample_stats on a DataTree with no such group raises AttributeError rather than
    # behaving like an absent/empty mapping (same guard convention used for
    # '/log_likelihood' elsewhere in this module).
    has_sample_stats = "/sample_stats" in trace.groups
    logp_name = None
    if has_sample_stats:
        logp_name = ('logp' if 'logp' in trace.sample_stats
                    else ('lp' if 'lp' in trace.sample_stats else None))
    if logp_name is None:
        return {"best_category": None, "group_means": {}, "gaps": {}, "n_mixed_chains": 0}

    logp = trace.sample_stats[logp_name].values  # (chain, draw)
    n_chains = dominant.shape[0]
    chain_mean_logp = logp.mean(axis=1)
    within_std = logp.std(axis=1).mean()

    groups: dict[int, list[float]] = {}
    n_mixed = 0
    for c in range(n_chains):
        cats, counts = np.unique(dominant[c], return_counts=True)
        top = counts.argmax()
        purity = counts[top] / counts.sum()
        if purity < purity_threshold:
            n_mixed += 1
            continue
        groups.setdefault(int(cats[top]), []).append(chain_mean_logp[c])

    group_means = {cat: float(np.mean(vals)) for cat, vals in groups.items()}
    if len(group_means) < 2:
        return {"best_category": None, "group_means": group_means,
                "gaps": {}, "n_mixed_chains": n_mixed}

    best_category = max(group_means, key=group_means.get)
    best_mean = group_means[best_category]
    scale = within_std if within_std > 0 else 1.0
    gaps = {cat: (best_mean - mean) / scale
            for cat, mean in group_means.items() if cat != best_category}
    return {"best_category": best_category, "group_means": group_means,
            "gaps": gaps, "n_mixed_chains": n_mixed}


def _categorize(mean_switch_rate, gap_result, thresholds, action_templates, unit_label="nats"):
    """
    Shared five-category decision logic, extracted so classify_multimodality (known
    categories, nat-scale gaps from a named per-source log-likelihood) and
    classify_scalar_multimodality (discovered modes, sigma-scale gaps from total logp) apply
    IDENTICAL branching — the decision itself doesn't care where switch_rate/gap_result came
    from, only the recommended_action text (action_templates) and the printed gap unit
    (unit_label, e.g. 'nats' vs 'pooled within-chain logp std devs') differ per call site.

    Parameters
    ----------
    mean_switch_rate     : float
    gap_result            : dict from compute_loglik_gap/compute_logp_gap (best_category,
                            group_means, gaps, n_mixed_chains)
    thresholds            : ModeClassificationThresholds
    action_templates      : dict with keys 'round_tripping', 'needs_review_none',
                            'needs_review_ambiguous' (a .format(gaps_str=..., unit_label=...)
                            template), 'stuck_fixable', 'hard_genuine', 'mixed'
    unit_label            : substituted into 'needs_review_ambiguous'

    Returns
    -------
    (category, recommended_action)
    """
    if mean_switch_rate > thresholds.switch_rate_threshold:
        return CAT_ROUND_TRIPPING, action_templates["round_tripping"]
    if gap_result["best_category"] is None:
        return CAT_NEEDS_REVIEW, action_templates["needs_review_none"]

    gaps = gap_result["gaps"]
    tied     = [c for c, g in gaps.items() if g < thresholds.gap_tied_threshold]
    decisive = [c for c, g in gaps.items() if g > thresholds.gap_decisive_threshold]
    ambiguous = [c for c in gaps if c not in tied and c not in decisive]
    if ambiguous:
        gaps_str = [round(gaps[c], 1) for c in ambiguous]
        return CAT_NEEDS_REVIEW, action_templates["needs_review_ambiguous"].format(
            gaps_str=gaps_str, unit_label=unit_label)
    if decisive and not tied:
        return CAT_STUCK_FIXABLE, action_templates["stuck_fixable"]
    if tied and not decisive:
        return CAT_HARD_GENUINE, action_templates["hard_genuine"]
    return CAT_MIXED, action_templates["mixed"]


_LAG_VAR_ACTION_TEMPLATES = {
    "round_tripping": "report pooled posterior as-is; do not attempt per-mode isolation/reseeding",
    "needs_review_none": "fewer than 2 pure chain-groups found; inspect manually",
    "needs_review_ambiguous": ("gap(s) {gaps_str} {unit_label} fall between the "
                               "tied/decisive thresholds; inspect manually"),
    "stuck_fixable": "reseed via resolve_stuck_areas and resample",
    "hard_genuine": "report honestly via hierarchical_mode_summary; no fix exists",
    "mixed": ("genuine tie among some groups, decisively-worse straggler(s) among others; "
             "NOT auto-resolved (see category docstring) — requires a manual judgment call"),
}


def classify_multimodality(trace, lag_var_name, loglik_var_name,
                           rhat_threshold=1.01, thresholds=ModeClassificationThresholds()):
    """
    Classify every area `hierarchical_mode_summary` flags into one of five categories.

    All thresholds below were derived empirically from the AZ1 family this session, not
    from a universal statistical rule — they should be re-checked, not blindly trusted,
    against a materially different model (different sigma_obs scale, different
    observations/area, different lag structure). See the module's own thresholds as a
    documented STARTING POINT, not a proof.

    Categories
    ----------
    hard_genuine    : switch rate low (chains don't round-trip) AND every non-best pure
                      group is within `gap_tied_threshold` nats of the best — a genuine,
                      irreducible tie (AZ1d's E01002702). No fix exists; report honestly
                      via hierarchical_mode_summary, don't try to resolve it.
    stuck_fixable   : switch rate low AND every non-best pure group is more than
                      `gap_decisive_threshold` nats worse than the best — chains stuck in
                      a decisively worse mode (Pattern 2b). Fixable: see
                      resolve_stuck_areas.
    round_tripping  : switch rate high — chains cross between categories repeatedly, no
                      chain is "stuck" anywhere (AZ1d's E01035649: every one of 8 chains
                      switched 26-55 times across 1500 draws). The per-chain purity-based
                      grouping this whole classification otherwise relies on is NOT
                      meaningful here (see docs/multimodality-characterization-guide.md
                      §3's purity>=95% validity caveat) — don't attempt to isolate or
                      reseed "modes" that aren't really separate populations of chains.
                      The ordinary pooled posterior is usually already a reasonable
                      summary of this (real, but shallow/overlapping) ambiguity.
    mixed           : switch rate low, but SOME non-best pure groups are genuinely tied
                      with the best while at least one OTHER group is decisively worse
                      (AZ1d's E01035708: a genuine tie between two categories, PLUS one
                      lone chain stuck in a clearly worse third). Deliberately NOT
                      auto-resolved by resolve_stuck_areas — nutpie's `init_mean` seeds
                      the WHOLE area toward one target for every chain, so "fixing" the
                      stuck straggler would also force every genuinely-tied chain toward
                      the same target, destroying real ambiguity you want to keep. Needs a
                      human judgment call.
    needs_review    : doesn't cleanly fit any of the above — fewer than 2 pure groups
                      found (can't compute a gap at all), or a gap that falls between
                      `gap_tied_threshold` and `gap_decisive_threshold` (neither clearly
                      tied nor clearly decisive). Flag for manual inspection rather than
                      force a label the evidence doesn't clearly support.

    Returns
    -------
    pd.DataFrame, one row per area hierarchical_mode_summary flags:
        area, max_rhat, mean_switch_rate, n_pure_chains, n_mixed_chains, best_category,
        gaps (dict), category, recommended_action
    """
    columns = ["area", "max_rhat", "mean_switch_rate", "n_pure_chains", "n_mixed_chains",
               "best_category", "gaps", "category", "recommended_action"]

    mode_df = hierarchical_mode_summary(trace, lag_var_name, rhat_threshold=rhat_threshold,
                                        purity_threshold=thresholds.purity_threshold)
    if mode_df.empty:
        return pd.DataFrame(columns=columns)

    lsoa_codes = (trace.posterior.coords["area"].values.tolist()
                  if "area" in trace.posterior.coords else None)
    switch_rate, dominant = compute_switch_rates(trace, lag_var_name)

    rows = []
    for _, row in mode_df.iterrows():
        area_idx = int(row["group"])
        area_code = lsoa_codes[area_idx] if lsoa_codes is not None else area_idx
        mean_switch_rate = float(switch_rate[:, area_idx].mean())

        gap_result = compute_loglik_gap(trace, dominant, area_idx, loglik_var_name,
                                        purity_threshold=thresholds.purity_threshold)
        n_pure = int(row["n_chains"]) - int(row["n_mixed_chains"])

        category, action = _categorize(
            mean_switch_rate, gap_result, thresholds,
            _LAG_VAR_ACTION_TEMPLATES, unit_label="nats")

        rows.append({
            "area": area_code,
            "max_rhat": float(row["max_rhat"]),
            "mean_switch_rate": mean_switch_rate,
            "n_pure_chains": n_pure,
            "n_mixed_chains": int(row["n_mixed_chains"]),
            "best_category": gap_result["best_category"],
            "gaps": gap_result["gaps"],
            "category": category,
            "recommended_action": action,
        })

    return pd.DataFrame(rows, columns=columns)


_SCALAR_ACTION_TEMPLATES = {
    "round_tripping": ("report pooled posterior as-is; this cell's chain clusters overlap "
                       "too much to isolate/reseed"),
    "needs_review_none": "fewer than 2 pure chain-groups found; inspect manually",
    "needs_review_ambiguous": ("gap(s) {gaps_str} {unit_label} fall between the "
                               "tied/decisive thresholds; inspect manually"),
    "stuck_fixable": ("one or more chain-groups are decisively worse than the best — likely "
                      "a stuck/trapped chain, not genuine ambiguity; consider a longer run "
                      "or a manually-informed reseed (resolve_stuck_areas only targets "
                      "*_lambda_weights raw_offset, not arbitrary scalars)"),
    "hard_genuine": "irreducible tie between distinct chain-level modes; report honestly, no fix expected",
    "mixed": ("genuine tie among some chain-groups, decisively-worse straggler(s) among "
             "others; requires a manual judgment call"),
}


def classify_scalar_multimodality(trace, var_names, rhat_threshold=1.01,
                                  thresholds=ModeClassificationThresholds(),
                                  sigma_threshold=3.0):
    """
    Generic-variable counterpart to classify_multimodality: classifies every individual cell
    of every named scalar/vector variable in `var_names` with elevated r-hat, using
    DISCOVERED (not known) chain-level modes — see `_discover_modes_from_chain_means` — and a
    sigma-normalized total-logp gap check (`compute_logp_gap`) instead of a named
    per-observation log-likelihood, since a generic scalar hyperparameter has no dedicated
    `log_likelihood` group to check against the way `*_lambda_weights` has `P_like`/`E_like`.

    A sixth outcome, not possible on the categorical (`*_lambda_weights`) path, shows up here:
    `CAT_NOT_MULTIMODAL` — a flagged cell where fewer than 2 chain-level clusters were
    discovered at all. That cell's bad r-hat/ESS is a REAL problem this pipeline doesn't
    explain (slow mixing, a funnel, etc — not multimodality) — it is NEVER excluded from
    `adjusted`/`best_case`, and is reported as a status distinct from `needs_review`: "we're
    confident there ISN'T a mode-separation explanation" versus "there might be, we can't
    tell yet."

    Parameters mirror `classify_multimodality` where the concept carries over directly;
    `sigma_threshold` replaces its implicit reliance on known categories for mode discovery
    (see `_discover_modes_from_chain_means`'s own calibration caveat — this is a fresh,
    not-yet-validated-on-real-data starting point).

    Parameters
    ----------
    trace          : az.InferenceData
    var_names      : iterable of variable names to check (e.g. a model's own `var_names`,
                     same convention `diagnostics_summary` uses) — names absent from
                     `trace.posterior` are silently skipped
    rhat_threshold : per-cell r-hat above this triggers classification for that cell
    thresholds     : ModeClassificationThresholds — see classify_multimodality
    sigma_threshold : passed to `_discover_modes_from_chain_means` for mode discovery

    Returns
    -------
    pd.DataFrame, one row per flagged (var, cell): `var`, `cell` (index tuple into the
    variable's non-chain/draw dims — `()` for a true scalar), `max_rhat`, `mean_switch_rate`,
    `n_pure_chains`, `n_mixed_chains`, `best_category`, `gaps`, `category`,
    `recommended_action` — the same column set as `classify_multimodality`'s per-area table
    (`area` → `var` + `cell`), so `multimodality_report` can treat both tables uniformly.
    """
    columns = ["var", "cell", "max_rhat", "mean_switch_rate", "n_pure_chains",
               "n_mixed_chains", "best_category", "gaps", "category", "recommended_action"]
    rows = []
    for var_name in var_names:
        if var_name not in trace.posterior:
            continue
        values_all = trace.posterior[var_name].values  # (chain, draw, *extra)
        n_chains = values_all.shape[0]
        rhat_vals = az.rhat(trace, var_names=[var_name])[var_name].values  # shape == *extra

        for idx in np.argwhere(np.isfinite(rhat_vals) & (rhat_vals > rhat_threshold)):
            cell = tuple(int(i) for i in idx)
            max_rhat = float(rhat_vals[cell])
            cell_values = values_all[(slice(None), slice(None), *cell)]  # (chain, draw)

            mode_locations = _discover_modes_from_chain_means(
                cell_values, sigma_threshold=sigma_threshold)
            if len(mode_locations) < 2:
                rows.append({
                    "var": var_name, "cell": cell, "max_rhat": max_rhat,
                    "mean_switch_rate": float("nan"), "n_pure_chains": 0, "n_mixed_chains": 0,
                    "best_category": None, "gaps": {}, "category": CAT_NOT_MULTIMODAL,
                    "recommended_action": (
                        "no distinct chain-level clusters found; this cell's bad r-hat/ESS "
                        "is NOT attributable to multimodality — see "
                        "ess-rhat-diagnostic-guide.md instead"),
                })
                continue

            dominant = _nearest_mode_labels(cell_values, mode_locations)
            mean_switch_rate = float(_switch_rate_from_dominant(dominant).mean())
            gap_result = compute_logp_gap(trace, dominant,
                                          purity_threshold=thresholds.purity_threshold)

            n_pure = 0
            for c in range(n_chains):
                _, counts = np.unique(dominant[c], return_counts=True)
                if counts[counts.argmax()] / counts.sum() >= thresholds.purity_threshold:
                    n_pure += 1

            category, action = _categorize(
                mean_switch_rate, gap_result, thresholds, _SCALAR_ACTION_TEMPLATES,
                unit_label="pooled within-chain logp std devs")

            rows.append({
                "var": var_name, "cell": cell, "max_rhat": max_rhat,
                "mean_switch_rate": mean_switch_rate, "n_pure_chains": n_pure,
                "n_mixed_chains": n_chains - n_pure,
                "best_category": gap_result["best_category"], "gaps": gap_result["gaps"],
                "category": category, "recommended_action": action,
            })

    return pd.DataFrame(rows, columns=columns)


def resolve_stuck_areas(model_class, data, classification_df, lag_var_name,
                        chains=16, seed_logit=3.0, other_logit=-1.0, **sample_kwargs):
    """
    Reseed and resample for areas classified `stuck_fixable` — the validated fix from
    docs/az-ess-diagnosis.md (AZ1d's 16-chain informed-init experiment): seed EVERY
    chain's `init_mean` toward the empirically-favoured (highest-log-likelihood) category
    for each stuck area, then resample with more chains. Cross-validated this session:
    seeding predicted exactly which areas would resolve (the stuck_fixable ones) and which
    would stay split (the hard_genuine/mixed ones) even when seeded as a control.

    Deliberately does NOT touch hard_genuine, round_tripping, mixed, or needs_review areas
    — see classify_multimodality's category docstrings for why each is excluded (no target
    to seed toward, no stable population to isolate, or seeding would destroy real
    ambiguity you want to keep since nutpie's `init_mean` applies one shared starting point
    to every chain — it cannot selectively reseed a subset of chains within one area).

    Parameters
    ----------
    model_class       : a DwellingModel subclass (e.g. AZ1d) — must have `max_lag` set
    data              : data dict whose areas EXACTLY match classification_df's (e.g.
                        built via the same `_data_matching_traces` helper cli.py uses)
    classification_df : output of classify_multimodality, from the trace this data matches
    lag_var_name      : e.g. 'lag_P_lambda_weights' — the raw_offset variable name is
                        derived by convention (strip '_lambda_weights', append
                        '_raw_offset'), matching _build_hierarchical_lag's own naming
    chains            : chain count for the reseeded run (default 16, matching the
                        validated AZ1d experiment — more than the usual 8 to make the
                        post-hoc purity check trustworthy)
    seed_logit        : raw_offset value written for the favoured category's free
                        dimension (default 3.0, matching the validated experiment)
    other_logit       : raw_offset value written for every OTHER free dimension of a
                        seeded area (default -1.0, discourages them without a hard
                        exclusion)
    **sample_kwargs   : passed through to model.sample() (draws, tune, target_accept,
                        random_seed, ...) — chains is set separately via the `chains` arg

    Returns
    -------
    (trace, seeded_areas) : trace is None if no stuck_fixable areas were found (nothing to
    do); seeded_areas is the list of area codes actually reseeded.
    """
    prefix = lag_var_name.removesuffix("_lambda_weights")
    raw_offset_name = f"{prefix}_raw_offset"

    stuck = classification_df[classification_df["category"].isin(_FIXABLE_CATEGORIES)]
    if stuck.empty:
        return None, []

    lsoa_codes = data["gdf"]["LSOA21CD"].tolist()
    area_index = {code: i for i, code in enumerate(lsoa_codes)}
    n_areas = data["n_areas"]

    m = model_class(data)
    n_free = m.n_lags - 1

    raw_offset_init = np.zeros((n_areas, n_free))
    seeded_areas = []
    for _, row in stuck.iterrows():
        code = row["area"]
        cat = row["best_category"]
        # pandas upcasts this column to float64 when any row's best_category is None
        # (e.g. a needs_review row elsewhere in the same classification_df) — cast back
        # to int before using it as an array index.
        if code not in area_index or cat is None or (isinstance(cat, float) and np.isnan(cat)):
            continue
        cat = int(cat)
        if cat == 0:
            # category 0 is the softmax reference (already the all-zeros default) —
            # nothing to seed toward if that's already the favoured category.
            continue
        idx = area_index[code]
        free_dim = cat - 1
        raw_offset_init[idx, free_dim] = seed_logit
        for d in range(n_free):
            if d != free_dim:
                raw_offset_init[idx, d] = other_logit
        seeded_areas.append(code)

    if not seeded_areas:
        return None, []

    init_mean = {raw_offset_name: raw_offset_init}
    m.sample(use_nutpie=True, chains=chains, cores=sample_kwargs.pop("cores", chains),
             init_mean=init_mean, **sample_kwargs)
    return m.trace, seeded_areas


def verify_resolution(new_trace, lag_var_name, seeded_areas, purity_threshold=0.75):
    """
    After resolve_stuck_areas resamples with informed init, check per-area whether the
    seeded run actually converged to a clear majority on one category — confirms genuine
    resolution rather than assuming the seed "worked" just because it ran.

    A seeded stuck_fixable area should resolve (this was validated directly this session:
    AZ1d's Pattern-2b areas resolved to 69-100% chain agreement once seeded — e.g.
    E01033700 at 69%, E01035656 at 81%, E01033711 at 100%, all reported as genuine
    resolutions in docs/az-ess-diagnosis.md). purity_threshold defaults to 0.75, not
    0.95, deliberately: with a realistic chain count (16 in the validated recipe), 0.95
    effectively demands total unanimity (15/16 = 93.75% already falls short of it), which
    is stricter than this session's own precedent for what counts as "resolved" — a
    stricter threshold can be passed explicitly if that precedent isn't the right bar for
    a given use case, but 0.75 matches what was actually validated here. If a seeded area
    does NOT clear even this bar, that itself is informative — it means the earlier
    classification was wrong for that specific area (it wasn't really cleanly "stuck", e.g.
    its switch rate/gap sat right at a threshold boundary, or the small baseline chain
    count made its favoured-category estimate noisy) — treat as needing a fresh look, not
    a silent failure. The raw `chain_agreement_frac` is always returned alongside the
    boolean specifically so this can be judged directly rather than trusting one cutoff.

    Parameters
    ----------
    new_trace        : trace from resolve_stuck_areas
    lag_var_name     : e.g. 'lag_P_lambda_weights'
    seeded_areas     : area codes list from resolve_stuck_areas
    purity_threshold : min fraction of chains that must agree on the SAME dominant
                       category for the area to count as resolved

    Returns
    -------
    pd.DataFrame: area, chain_agreement_frac, resolved (bool)
    """
    lsoa_codes = new_trace.posterior.coords["area"].values.tolist()
    area_index = {code: i for i, code in enumerate(lsoa_codes)}
    lw = new_trace.posterior[lag_var_name].values
    dominant = lw.argmax(axis=-1)
    n_chains = dominant.shape[0]

    rows = []
    for code in seeded_areas:
        if code not in area_index:
            continue
        a = area_index[code]
        dominant_cats = []
        for c in range(n_chains):
            cats, counts = np.unique(dominant[c, :, a], return_counts=True)
            dominant_cats.append(int(cats[counts.argmax()]))
        agreement_frac = max(dominant_cats.count(k) for k in set(dominant_cats)) / n_chains
        rows.append({
            "area": code,
            "chain_agreement_frac": agreement_frac,
            "resolved": agreement_frac >= purity_threshold,
        })
    return pd.DataFrame(rows)


def _within_chain_ess(trace, var_name, area_idx, category):
    """
    Per-chain (not cross-chain) ESS for one (area, category) cell of a lag-weights
    variable — checks whether each individual chain's OWN sampling is well-mixed, as
    distinct from the standard cross-chain r-hat/ESS, which is EXPECTED to look bad for a
    genuinely multimodal area (chains correctly disagreeing on which mode they're in isn't
    a mixing failure). Reused from ess-rhat-diagnostic-guide.md §2 step 1's own
    per-chain-autocorrelation idea, generalized into a reusable numeric check.

    Returns list of per-chain ess_bulk values (one per chain).
    """
    values = trace.posterior[var_name].values[:, :, area_idx, category]  # (chain, draw)
    n_chains = values.shape[0]
    ess_per_chain = []
    for c in range(n_chains):
        single = az.convert_to_dataset({var_name: values[c:c + 1]})
        ess_per_chain.append(float(az.ess(single, method="bulk")[var_name].values))
    return ess_per_chain


def adjusted_diagnostics_report(trace, lag_var_name, loglik_var_name,
                                rhat_threshold=1.01, resolved_verification=None,
                                thresholds=None):
    """
    The "at a glance" summary this pipeline exists to produce: how much of the raw
    bad-r-hat/low-ESS picture is genuine (expected) multimodality versus a real problem.

    Combines classify_multimodality with the model's own z r-hat picture
    (diagnostics.z_identifiability_summary) to report:
      - counts per category (hard_genuine, stuck_fixable, round_tripping, mixed,
        needs_review), and how many stuck_fixable areas were confirmed resolved (if
        resolved_verification, from verify_resolution, is supplied)
      - adjusted_max_rhat / adjusted_min_ess: the same lag_var_name r-hat/ESS numbers,
        but excluding cells belonging to areas classified hard_genuine or round_tripping
        (their poor cross-chain agreement is EXPECTED and uninformative about real
        sampling health — including them in a raw "max r-hat across the model" number
        conflates "the sampler is unhealthy" with "the posterior genuinely disagrees",
        which is exactly the ambiguity this whole pipeline exists to resolve). Cells for
        stuck_fixable/mixed/needs_review areas are DELIBERATELY still included here —
        they haven't been established as benign yet, so this number stays a genuine
        "is there still a sampling problem" read.
      - best_case_max_rhat / best_case_min_ess: the same numbers again, but excluding
        EVERY flagged area regardless of category or resolution status — the reading
        for "the rest of the model" once every flagged area has been triaged one way or
        another (fixed, accepted as genuine, or just noted and set aside). Unlike
        adjusted_max_rhat/min_ess, this is a best case rather than a currently-justified
        one: an unresolved stuck_fixable or an unreviewed needs_review area is excluded
        here even though nothing has actually confirmed it's fine — don't quote this
        number as "the model's real r-hat" until n_needs_deep_dive is actually zero (or
        every stuck_fixable area has a confirmed resolution); until then it's a preview
        of where the headline number is headed, not where it already is.
      - within_chain_ess: per-chain ESS for each hard_genuine/round_tripping area's own
        best category, confirming each chain's OWN mixing is healthy even where the
        cross-chain comparison is expected to fail
      - n_needs_deep_dive: needs_review count, PLUS any stuck_fixable area that
        resolved_verification shows did NOT actually resolve (the classification's own
        wrong guesses, surfaced rather than silently trusted)

    Parameters
    ----------
    thresholds : ModeClassificationThresholds, passed through to classify_multimodality
                 (default: ModeClassificationThresholds() if None)

    Returns
    -------
    dict — see the five bullet groups above for keys; also includes the full
    `classification_df` for detailed per-area follow-up.
    """
    classification_df = classify_multimodality(
        trace, lag_var_name, loglik_var_name, rhat_threshold=rhat_threshold,
        thresholds=thresholds or ModeClassificationThresholds())

    counts = classification_df["category"].value_counts().to_dict()
    for cat in (CAT_HARD_GENUINE, CAT_STUCK_FIXABLE, CAT_ROUND_TRIPPING, CAT_MIXED,
                CAT_NEEDS_REVIEW):
        counts.setdefault(cat, 0)

    n_resolved = 0
    n_stuck_unresolved = 0
    if resolved_verification is not None and not resolved_verification.empty:
        n_resolved = int(resolved_verification["resolved"].sum())
        n_stuck_unresolved = int((~resolved_verification["resolved"]).sum())

    # mixed areas are deliberately never auto-resolved (see the category's own
    # docstring) so they always need a human call, same as needs_review.
    n_needs_deep_dive = counts[CAT_NEEDS_REVIEW] + counts[CAT_MIXED] + n_stuck_unresolved

    excluded_areas = set(classification_df.loc[
        classification_df["category"].isin([CAT_HARD_GENUINE, CAT_ROUND_TRIPPING]), "area"])

    rhat_da = az.rhat(trace, var_names=[lag_var_name])[lag_var_name]
    ess_da  = az.ess(trace, var_names=[lag_var_name], method="bulk")[lag_var_name]
    lsoa_codes = (trace.posterior.coords["area"].values.tolist()
                  if "area" in trace.posterior.coords else list(range(rhat_da.shape[0])))

    keep_mask = np.array([code not in excluded_areas for code in lsoa_codes])
    raw_rhat_vals = rhat_da.values[np.isfinite(rhat_da.values)]
    raw_ess_vals  = ess_da.values[np.isfinite(ess_da.values)]
    adj_rhat_vals = rhat_da.values[keep_mask][np.isfinite(rhat_da.values[keep_mask])]
    adj_ess_vals  = ess_da.values[keep_mask][np.isfinite(ess_da.values[keep_mask])]

    # best-case: exclude every area classify_multimodality flagged at all, not just
    # the two categories already established as benign — see the docstring's own
    # caveat on treating this as a preview, not a currently-justified number.
    all_flagged_areas = set(classification_df["area"])
    best_case_mask = np.array([code not in all_flagged_areas for code in lsoa_codes])
    best_rhat_vals = rhat_da.values[best_case_mask][np.isfinite(rhat_da.values[best_case_mask])]
    best_ess_vals  = ess_da.values[best_case_mask][np.isfinite(ess_da.values[best_case_mask])]

    within_chain_ess = {}
    lw_all = trace.posterior[lag_var_name].values  # (chain, draw, area, category)
    for _, row in classification_df.iterrows():
        if row["category"] not in (CAT_HARD_GENUINE, CAT_ROUND_TRIPPING):
            continue
        area_idx = lsoa_codes.index(row["area"]) if row["area"] in lsoa_codes else None
        if area_idx is None:
            continue
        cat = row["best_category"]
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            # round_tripping areas usually have no stable pure-chain-group majority
            # (that's the whole point), so gap_result never found >=2 pure groups —
            # fall back to whichever category the POOLED posterior favours overall,
            # just to have a representative category to check each chain's own
            # mixing quality against.
            cat = int(lw_all[:, :, area_idx, :].mean(axis=(0, 1)).argmax())
        else:
            cat = int(cat)
        within_chain_ess[row["area"]] = _within_chain_ess(
            trace, lag_var_name, area_idx, cat)

    return {
        "n_areas_total":     len(lsoa_codes),
        "n_flagged":         len(classification_df),
        "n_hard_genuine":    counts[CAT_HARD_GENUINE],
        "n_stuck_fixable":   counts[CAT_STUCK_FIXABLE],
        "n_round_tripping": counts[CAT_ROUND_TRIPPING],
        "n_mixed":           counts[CAT_MIXED],
        "n_needs_review":    counts[CAT_NEEDS_REVIEW],
        "n_resolved":        n_resolved,
        "n_stuck_unresolved": n_stuck_unresolved,
        "n_needs_deep_dive": n_needs_deep_dive,
        "raw_max_rhat":      float(raw_rhat_vals.max()) if len(raw_rhat_vals) else float("nan"),
        "raw_min_ess":       float(raw_ess_vals.min())  if len(raw_ess_vals)  else float("nan"),
        "adjusted_max_rhat": float(adj_rhat_vals.max()) if len(adj_rhat_vals) else float("nan"),
        "adjusted_min_ess":  float(adj_ess_vals.min())  if len(adj_ess_vals)  else float("nan"),
        "best_case_max_rhat": float(best_rhat_vals.max()) if len(best_rhat_vals) else float("nan"),
        "best_case_min_ess":  float(best_ess_vals.min())  if len(best_ess_vals)  else float("nan"),
        "within_chain_ess":  within_chain_ess,
        "classification_df": classification_df,
        # full filtered value arrays (not just their max/min) — multimodality_report
        # concatenates these across every lag var/scalar source so
        # adjusted_diagnostics_summary can compute mean_rhat/n_bad_rhat over the exact
        # same population max_rhat/min_ess are drawn from, rather than re-deriving it
        # separately and risking the two disagreeing.
        "raw_rhat_values":         raw_rhat_vals,
        "raw_ess_values":          raw_ess_vals,
        "adjusted_rhat_values":    adj_rhat_vals,
        "adjusted_ess_values":     adj_ess_vals,
        "best_case_rhat_values":   best_rhat_vals,
        "best_case_ess_values":    best_ess_vals,
    }


def lag_vars_in_trace(trace):
    """Every `*_lambda_weights` variable in trace.posterior — a model may have one
    (planning only), two (planning + BEN), or none."""
    return [v for v in trace.posterior.data_vars if v.endswith("_lambda_weights")]


def derive_loglik_var(lag_var_name):
    """'lag_P_lambda_weights' -> 'P_like', 'lag_E_lambda_weights' -> 'E_like' — the
    naming convention every _build_hierarchical_lag*/likelihood pair in models.py uses."""
    prefix = lag_var_name.removesuffix("_lambda_weights")
    source = prefix.rsplit("_", 1)[-1]
    return f"{source}_like"


def _scalar_diagnostics_report(trace, var_names, rhat_threshold=1.01, thresholds=None):
    """
    Scalar-variable counterpart to adjusted_diagnostics_report: runs
    classify_scalar_multimodality over `var_names` and reports counts across all SIX
    categories (including CAT_NOT_MULTIMODAL) plus raw/adjusted/best_case r-hat/ESS scoped
    to exactly the (var, cell) pairs actually examined — every cell of every name in
    `var_names` for raw, with flagged cells excluded per category for adjusted/best_case —
    mirroring adjusted_diagnostics_report's per-lag-var scoping, keyed by (var, cell)
    instead of area.

    NEVER excludes CAT_NOT_MULTIMODAL cells from adjusted/best_case — that category means
    the bad r-hat/ESS is a real, unexplained problem, not benign multimodality, so excluding
    it would misrepresent best_case as "resolved" when nothing has actually been explained.
    `stuck_fixable` always counts toward `n_needs_deep_dive` here (unlike the lag-var path,
    where an attempted-and-confirmed resolution can remove it) — there is no automated
    resolution path for an arbitrary scalar the way `resolve_stuck_areas` targets
    `*_lambda_weights` raw_offset, so an "attempted, not yet confirmed" state doesn't exist.

    Returns
    -------
    dict: n_flagged, n_hard_genuine, n_stuck_fixable, n_round_tripping, n_mixed,
    n_needs_review, n_not_multimodal, n_needs_deep_dive, raw_max_rhat/min_ess,
    adjusted_max_rhat/min_ess, best_case_max_rhat/min_ess, classification_df.
    """
    classification_df = classify_scalar_multimodality(
        trace, var_names, rhat_threshold=rhat_threshold,
        thresholds=thresholds or ModeClassificationThresholds())

    counts = classification_df["category"].value_counts().to_dict()
    for cat in (CAT_HARD_GENUINE, CAT_STUCK_FIXABLE, CAT_ROUND_TRIPPING, CAT_MIXED,
                CAT_NEEDS_REVIEW, CAT_NOT_MULTIMODAL):
        counts.setdefault(cat, 0)

    n_needs_deep_dive = (counts[CAT_NEEDS_REVIEW] + counts[CAT_MIXED]
                         + counts[CAT_STUCK_FIXABLE])

    excluded_adjusted = set()
    excluded_best_case = set()
    for _, row in classification_df.iterrows():
        key = (row["var"], row["cell"])
        if row["category"] in (CAT_HARD_GENUINE, CAT_ROUND_TRIPPING):
            excluded_adjusted.add(key)
        if row["category"] != CAT_NOT_MULTIMODAL:
            excluded_best_case.add(key)

    raw_rhat_vals, raw_ess_vals = [], []
    adj_rhat_vals, adj_ess_vals = [], []
    best_rhat_vals, best_ess_vals = [], []
    for var_name in var_names:
        if var_name not in trace.posterior:
            continue
        rhat_vals = az.rhat(trace, var_names=[var_name])[var_name].values
        ess_vals  = az.ess(trace, var_names=[var_name], method="bulk")[var_name].values
        for cell in np.ndindex(rhat_vals.shape):
            r, e = rhat_vals[cell], ess_vals[cell]
            key = (var_name, cell)
            # rhat and ess are recorded independently by their OWN finiteness (matching
            # diagnostics_summary's own convention) -- a cell with e.g. NaN rhat (a fully
            # degenerate/constant series) but a finite ess must not lose that finite ess
            # just because rhat happened to be unusable.
            if np.isfinite(r):
                raw_rhat_vals.append(r)
                if key not in excluded_adjusted:
                    adj_rhat_vals.append(r)
                if key not in excluded_best_case:
                    best_rhat_vals.append(r)
            if np.isfinite(e):
                raw_ess_vals.append(e)
                if key not in excluded_adjusted:
                    adj_ess_vals.append(e)
                if key not in excluded_best_case:
                    best_ess_vals.append(e)

    def _agg(vals, reducer):
        return float(reducer(vals)) if vals else float("nan")

    return {
        "n_flagged":         len(classification_df),
        "n_hard_genuine":    counts[CAT_HARD_GENUINE],
        "n_stuck_fixable":   counts[CAT_STUCK_FIXABLE],
        "n_round_tripping":  counts[CAT_ROUND_TRIPPING],
        "n_mixed":           counts[CAT_MIXED],
        "n_needs_review":    counts[CAT_NEEDS_REVIEW],
        "n_not_multimodal":  counts[CAT_NOT_MULTIMODAL],
        "n_needs_deep_dive": n_needs_deep_dive,
        "raw_max_rhat":      _agg(raw_rhat_vals, max),
        "raw_min_ess":       _agg(raw_ess_vals, min),
        "adjusted_max_rhat": _agg(adj_rhat_vals, max),
        "adjusted_min_ess":  _agg(adj_ess_vals, min),
        "best_case_max_rhat": _agg(best_rhat_vals, max),
        "best_case_min_ess":  _agg(best_ess_vals, min),
        "classification_df": classification_df,
        # see adjusted_diagnostics_report's own comment on these — full filtered arrays,
        # not just their max/min, for multimodality_report to concatenate across sources.
        "raw_rhat_values":       np.array(raw_rhat_vals),
        "raw_ess_values":        np.array(raw_ess_vals),
        "adjusted_rhat_values":  np.array(adj_rhat_vals),
        "adjusted_ess_values":   np.array(adj_ess_vals),
        "best_case_rhat_values": np.array(best_rhat_vals),
        "best_case_ess_values":  np.array(best_ess_vals),
    }


def _passthrough_report(rhat_da, ess_da):
    """
    Minimal report shape for a lag var multimodality_report can't classify (no matching
    log_likelihood entry — see derive_loglik_var) — contributes its cells UNFILTERED to
    raw/adjusted/best_case (nothing to exclude, since nothing could be classified) rather
    than dropping them from the aggregate, matching diagnostics_summary's own convention of
    reporting an unclassifiable variable unchanged rather than silently ignoring it.
    """
    finite_rhat = rhat_da.values[np.isfinite(rhat_da.values)]
    finite_ess  = ess_da.values[np.isfinite(ess_da.values)]
    max_rhat = float(finite_rhat.max()) if len(finite_rhat) else float("nan")
    min_ess  = float(finite_ess.min())  if len(finite_ess)  else float("nan")
    return {
        "n_flagged": 0, "n_hard_genuine": 0, "n_stuck_fixable": 0, "n_round_tripping": 0,
        "n_mixed": 0, "n_needs_review": 0, "n_needs_deep_dive": 0,
        "raw_max_rhat": max_rhat, "raw_min_ess": min_ess,
        "adjusted_max_rhat": max_rhat, "adjusted_min_ess": min_ess,
        "best_case_max_rhat": max_rhat, "best_case_min_ess": min_ess,
        "raw_rhat_values": finite_rhat, "raw_ess_values": finite_ess,
        "adjusted_rhat_values": finite_rhat, "adjusted_ess_values": finite_ess,
        "best_case_rhat_values": finite_rhat, "best_case_ess_values": finite_ess,
    }


def multimodality_report(trace, lag_vars=None, var_names=None, rhat_threshold=1.01,
                         resolved_trace=None, thresholds=None):
    """
    The actual entry point `check-multimodality` uses: combines adjusted_diagnostics_report
    across EVERY `*_lambda_weights` variable a trace has (by default — pass `lag_vars` to
    restrict to a subset) AND, if `var_names` is given, classify_scalar_multimodality across
    every named scalar/vector variable too — so raw/adjusted/best_case r-hat/ESS are computed
    over ONE unified scope (`var_names` ∪ `lag_vars`), matching plain `diagnose`'s own scope
    for `max_rhat`/`min_ess` exactly. That's the whole point: `raw_max_rhat`/`min_ess` here IS
    that same unfiltered reading, and `best_case_max_rhat`/`min_ess` is the SAME scope with
    every cell attributable to multimodality excluded — a genuinely comparable before/after,
    not two numbers that happen to share a name while covering different variables.

    History (why this function exists at all, in two steps):
    (1) `check-multimodality --models AZ4b` used to default to checking only
        `lag_P_lambda_weights`, silently missing `lag_E_lambda_weights`'s own flagged areas —
        fixed by lag_vars defaulting to `lag_vars_in_trace(trace)` (every lag var found).
    (2) Even with (1) fixed, `best_case` was scoped ONLY to `*_lambda_weights` cells while
        `diagnose`'s plain `max_rhat`/`min_ess` also includes every scalar hyperparameter
        (sigma_plan, lag_E_mu_logit, ...) unfiltered — so the two were still answering
        different-scope questions, and a real, separate scalar problem (lag_E_mu_logit) could
        make "best case, multimodality resolved" look worse than it should, or a clean
        best_case could hide an unrelated scalar problem entirely. `var_names` fixes this by
        bringing scalars into the SAME classified, SAME-scope reading — see
        classify_scalar_multimodality / CAT_NOT_MULTIMODAL for how a scalar's bad r-hat that
        ISN'T attributable to multimodality is still visible (never quietly excluded).

    Parameters
    ----------
    trace           : az.InferenceData
    lag_vars        : optional list of `*_lambda_weights` variable names to restrict to
                      (default: every one found in trace.posterior, via lag_vars_in_trace)
    var_names       : optional list of scalar/vector variable names to ALSO classify via
                      classify_scalar_multimodality (default: None — skip scalar scanning
                      entirely, preserving lag-var-only behaviour for callers that don't pass
                      it). Pass a model's own `var_names` (same convention diagnostics_summary
                      uses) to bring the model's full diagnostic scope into this one report.
    rhat_threshold  : passed through to classify_multimodality/classify_scalar_multimodality
    resolved_trace  : optional resampled trace (from resolve_stuck_areas, or
                      `check-multimodality --resolve`'s saved `{model}_resolved.nc`) — used
                      to verify resolution for EACH lag var's own stuck_fixable subset (never
                      applies to scalars — there is no automated scalar resolution path)
    thresholds      : ModeClassificationThresholds, passed through to
                      classify_multimodality/classify_scalar_multimodality (default:
                      ModeClassificationThresholds() if None)

    Returns
    -------
    dict: n_flagged, n_hard_genuine, n_stuck_fixable, n_round_tripping, n_mixed,
    n_needs_review (summed across every lag var AND, if given, var_names), n_not_multimodal
    (scalar-only — see classify_scalar_multimodality; always 0 if var_names not given),
    n_resolved/n_stuck_unresolved (lag-var-only), n_needs_deep_dive, raw_max_rhat/min_ess,
    adjusted_max_rhat/min_ess, best_case_max_rhat/min_ess, within_chain_ess (lag-var-only,
    keyed "lag_var:area"), classification_df (lag-var findings, one row per (area, lag_var),
    unchanged from before), PLUS:
      lag_vars_checked         : list of lag vars actually classified against
      lag_vars_skipped         : lag vars found in the trace with no matching
                                 log_likelihood entry to classify against (see
                                 derive_loglik_var) — reported explicitly so a silent
                                 coverage gap is never mistaken for "nothing to see"
      by_lag_var                : {lag_var: its own adjusted_diagnostics_report dict}
      scalar_classification_df : one row per flagged (var, cell) from
                                 classify_scalar_multimodality — kept as its OWN table
                                 rather than merged into classification_df, since its key
                                 (var, cell) isn't an area code and forcing a shared schema
                                 would only obscure that. Empty if var_names not given.

    classification_df is tagged by lag_var specifically because the SAME area code can
    independently appear under more than one lag var — e.g. flagged round_tripping for
    planning but perfectly healthy for BEN — and collapsing that into one row would silently
    lose which source the finding is actually about.
    """
    lag_vars = list(lag_vars) if lag_vars is not None else lag_vars_in_trace(trace)
    thresholds = thresholds or ModeClassificationThresholds()
    n_areas_total = (len(trace.posterior.coords["area"].values)
                     if "area" in trace.posterior.coords else 0)

    per_lag_reports = {}
    skipped = []
    passthrough_reports = []
    for lag_var in lag_vars:
        loglik_var = derive_loglik_var(lag_var)
        # see adjusted_diagnostics_summary's own comment: accessing .log_likelihood on a
        # DataTree with no such group raises AttributeError rather than behaving like an
        # absent/empty mapping, so check group membership first.
        has_loglik_group = "/log_likelihood" in trace.groups
        if not has_loglik_group or loglik_var not in trace.log_likelihood:
            skipped.append(lag_var)
            # Can't classify this lag var at all (no matching log_likelihood), but its
            # cells still belong in the model's diagnostic scope — contribute them
            # UNFILTERED to raw/adjusted/best_case (nothing to exclude, since nothing
            # could be classified) rather than silently dropping them from the aggregate.
            rhat_da = az.rhat(trace, var_names=[lag_var])[lag_var]
            ess_da  = az.ess(trace, var_names=[lag_var], method="bulk")[lag_var]
            passthrough_reports.append(_passthrough_report(rhat_da, ess_da))
            continue

        resolved_verification = None
        if resolved_trace is not None:
            prelim_df = classify_multimodality(
                trace, lag_var, loglik_var, rhat_threshold=rhat_threshold,
                thresholds=thresholds)
            stuck_areas = prelim_df.loc[
                prelim_df["category"] == CAT_STUCK_FIXABLE, "area"].tolist()
            if stuck_areas:
                resolved_verification = verify_resolution(resolved_trace, lag_var, stuck_areas)

        per_lag_reports[lag_var] = adjusted_diagnostics_report(
            trace, lag_var, loglik_var, rhat_threshold=rhat_threshold,
            resolved_verification=resolved_verification, thresholds=thresholds)

    scalar_report = None
    if var_names is not None:
        scalar_report = _scalar_diagnostics_report(
            trace, var_names, rhat_threshold=rhat_threshold, thresholds=thresholds)

    all_reports = (list(per_lag_reports.values()) + passthrough_reports
                  + ([scalar_report] if scalar_report else []))

    if not all_reports:
        return {
            "lag_vars_checked": [], "lag_vars_skipped": skipped,
            "n_areas_total": n_areas_total, "n_flagged": 0,
            "n_hard_genuine": 0, "n_stuck_fixable": 0, "n_round_tripping": 0,
            "n_mixed": 0, "n_needs_review": 0, "n_not_multimodal": 0, "n_resolved": 0,
            "n_stuck_unresolved": 0, "n_needs_deep_dive": 0,
            "raw_max_rhat": float("nan"), "raw_min_ess": float("nan"),
            "adjusted_max_rhat": float("nan"), "adjusted_min_ess": float("nan"),
            "best_case_max_rhat": float("nan"), "best_case_min_ess": float("nan"),
            "raw_rhat_values": np.array([]), "raw_ess_values": np.array([]),
            "adjusted_rhat_values": np.array([]), "adjusted_ess_values": np.array([]),
            "best_case_rhat_values": np.array([]), "best_case_ess_values": np.array([]),
            "within_chain_ess": {}, "classification_df": pd.DataFrame(),
            "scalar_classification_df": pd.DataFrame(), "by_lag_var": {},
        }

    def _combined(key, reducer):
        finite_vals = [v for v in (r[key] for r in all_reports) if not np.isnan(v)]
        return reducer(finite_vals) if finite_vals else float("nan")

    def _combined_values(key):
        return np.concatenate([r[key] for r in all_reports])

    combined_classification = (
        pd.concat([r["classification_df"].assign(lag_var=lag_var)
                  for lag_var, r in per_lag_reports.items()], ignore_index=True)
        if per_lag_reports else pd.DataFrame())

    combined_within_chain_ess = {
        f"{lag_var}:{area}": ess
        for lag_var, r in per_lag_reports.items()
        for area, ess in r["within_chain_ess"].items()
    }

    return {
        "lag_vars_checked":  list(per_lag_reports),
        "lag_vars_skipped":  skipped,
        "n_areas_total":     n_areas_total,
        "n_flagged":         sum(r["n_flagged"] for r in all_reports),
        "n_hard_genuine":    sum(r["n_hard_genuine"] for r in all_reports),
        "n_stuck_fixable":   sum(r["n_stuck_fixable"] for r in all_reports),
        "n_round_tripping":  sum(r["n_round_tripping"] for r in all_reports),
        "n_mixed":           sum(r["n_mixed"] for r in all_reports),
        "n_needs_review":    sum(r["n_needs_review"] for r in all_reports),
        "n_not_multimodal":  scalar_report["n_not_multimodal"] if scalar_report else 0,
        "n_resolved":        sum(r["n_resolved"] for r in per_lag_reports.values()),
        "n_stuck_unresolved": sum(r["n_stuck_unresolved"] for r in per_lag_reports.values()),
        "n_needs_deep_dive": sum(r["n_needs_deep_dive"] for r in all_reports),
        "raw_max_rhat":      _combined("raw_max_rhat", max),
        "raw_min_ess":       _combined("raw_min_ess", min),
        "adjusted_max_rhat": _combined("adjusted_max_rhat", max),
        "adjusted_min_ess":  _combined("adjusted_min_ess", min),
        "best_case_max_rhat": _combined("best_case_max_rhat", max),
        "best_case_min_ess":  _combined("best_case_min_ess", min),
        # concatenated (not reduced) — adjusted_diagnostics_summary uses these to compute
        # mean_rhat/n_bad_rhat over the exact population max_rhat/min_ess are drawn from.
        "raw_rhat_values":         _combined_values("raw_rhat_values"),
        "raw_ess_values":          _combined_values("raw_ess_values"),
        "adjusted_rhat_values":    _combined_values("adjusted_rhat_values"),
        "adjusted_ess_values":     _combined_values("adjusted_ess_values"),
        "best_case_rhat_values":   _combined_values("best_case_rhat_values"),
        "best_case_ess_values":    _combined_values("best_case_ess_values"),
        "within_chain_ess":  combined_within_chain_ess,
        "classification_df": combined_classification,
        "scalar_classification_df": (scalar_report["classification_df"] if scalar_report
                                     else pd.DataFrame()),
        "by_lag_var":        per_lag_reports,
    }


def adjusted_diagnostics_summary(traces, resolved_traces=None, data=None, var_names=None,
                                 rhat_threshold=1.01, exclude_reviewed=True):
    """
    Multimodality-adjusted counterpart to diagnostics.diagnostics_summary, for the workflow
    `housing-projections diagnose` -> (if a model looks bad) `check-multimodality` -> maybe
    `--resolve` -> `diagnose --adjust-for-multimodality`: reports max/mean r-hat and min ESS
    ASSUMING every cell `check-multimodality` would flag (per-area lag-category ambiguity AND,
    now, scalar hyperparameter multimodality) has already been triaged — resolved where it
    validly can be, and accepted/tracked separately where it can't — rather than continuing to
    let it dominate the headline number.

    This is now a thin wrapper around `multimodality_report`, called once per model — see that
    function's own docstring for the actual classification/scoping logic (auto-detects every
    `*_lambda_weights` var; also classifies every name in `var_names[model]`, or every
    non-lag-var posterior variable if a model has no entry, matching this function's own
    historical unrestricted-scope behaviour — see the performance caveat on
    `diagnostics_summary` itself: pass `var_names` for anything beyond a handful of small
    scalars). `max_rhat`/`mean_rhat`/`n_bad_rhat`/`min_ess` are computed from
    `multimodality_report`'s own `best_case_*_values` arrays (if `exclude_reviewed`, the
    default) or `adjusted_*_values` (if not) — the full filtered cell population, not just its
    reduced max/min, so `n_bad_rhat`/`mean_rhat` stay consistent with `max_rhat`/`min_ess`
    rather than being derived separately.

    `exclude_reviewed` (default True): True maps to `best_case_*` (exclude EVERY flagged cell,
    any category, regardless of resolution status — the "once everything's triaged" reading);
    False maps to `adjusted_*` (exclude only hard_genuine/round_tripping — cells already
    established as benign, the more conservative "is there still a problem" reading). This is a
    SIMPLER two-way mapping than before: previously `exclude_reviewed=False` also separately
    credited a confirmed-resolved stuck_fixable area, which `adjusted_*` does not do — if you
    want credit for a specific confirmed resolution, use `exclude_reviewed=True` (best_case) or
    check `n_resolved`.

    `n_flagged_multimodal`/`n_needs_deep_dive` now include scalar findings too (previously
    lag-var-only) — see `multimodality_report`'s own `n_flagged`/`n_needs_deep_dive`.
    `n_not_multimodal` is new: scalar cells with bad r-hat that AREN'T attributable to
    multimodality at all (see `CAT_NOT_MULTIMODAL`) — never excluded from
    `adjusted`/`best_case`, reported here so a real, unexplained problem is never mistaken for
    "already accounted for."

    Parameters
    ----------
    traces          : dict {model_name: az.InferenceData} — same shape as diagnostics_summary
    resolved_traces : optional dict {model_name: az.InferenceData} — traces from
                      resolve_stuck_areas (or `check-multimodality --resolve`'s saved
                      `{model}_resolved.nc`), one per model that's had resolution attempted;
                      used ONLY to verify which stuck_fixable areas actually resolved (see
                      multimodality_report) — NOT as an alternate r-hat/ESS source for the rest
                      of the model. If you want the resolved run's OWN diagnostics, run
                      `diagnose` on `{model}_resolved.nc` directly.
    data            : passed through to diagnostics_summary for the unchanged columns
                      (frac_flat_despite_active, coverage)
    var_names       : dict {model_name: [scalar names]} — same convention as
                      diagnostics_summary
    rhat_threshold  : passed through to classify_multimodality/classify_scalar_multimodality/
                      diagnostics_summary
    exclude_reviewed : see above

    Returns
    -------
    pd.DataFrame — same columns as diagnostics_summary (max_rhat/mean_rhat/n_bad_rhat/min_ess
    now reflecting the adjustment described above), plus: raw_max_rhat, raw_min_ess (the
    unadjusted diagnostics_summary values, preserved here so callers can compare against
    best_case_max_rhat/best_case_min_ess and state whether multimodality actually explains a
    bad raw r-hat/ESS), best_case_max_rhat, best_case_min_ess, n_lambda_weights_vars,
    n_flagged_multimodal, n_resolved, n_needs_deep_dive, n_not_multimodal.
    """
    resolved_traces = resolved_traces or {}
    base = diagnostics_summary(traces, data=data, rhat_threshold=rhat_threshold,
                               var_names=var_names)

    for name, trace in traces.items():
        names = var_names.get(name) if var_names else None
        resolved_trace = resolved_traces.get(name)

        lag_vars = lag_vars_in_trace(trace)
        # Preserves this function's own historical scope exactly: every declared name if
        # given, else every non-lag-var posterior variable (the same "unrestricted" fallback
        # diagnostics_summary itself uses, and the same performance caveat applies).
        scalar_names = names if names is not None else [
            v for v in trace.posterior.data_vars if v not in lag_vars]

        report = multimodality_report(
            trace, lag_vars=lag_vars, var_names=scalar_names, rhat_threshold=rhat_threshold,
            resolved_trace=resolved_trace)

        if exclude_reviewed:
            rhat_vals = report["best_case_rhat_values"]
            ess_vals  = report["best_case_ess_values"]
        else:
            rhat_vals = report["adjusted_rhat_values"]
            ess_vals  = report["adjusted_ess_values"]

        # Capture the unadjusted numbers before they're overwritten below — needed so callers
        # (e.g. cli.py's diagnose) can state whether best_case actually improved on raw, and
        # therefore whether multimodality can be named as the cause of a bad raw r-hat/ESS.
        base.loc[name, "raw_max_rhat"] = base.loc[name, "max_rhat"]
        base.loc[name, "raw_min_ess"]  = base.loc[name, "min_ess"]

        base.loc[name, "max_rhat"]   = float(rhat_vals.max())  if len(rhat_vals) else float("nan")
        base.loc[name, "mean_rhat"]  = float(rhat_vals.mean()) if len(rhat_vals) else float("nan")
        base.loc[name, "n_bad_rhat"] = int((rhat_vals > rhat_threshold).sum())
        base.loc[name, "min_ess"]    = int(ess_vals.min()) if len(ess_vals) else -1
        base.loc[name, "best_case_max_rhat"]     = report["best_case_max_rhat"]
        base.loc[name, "best_case_min_ess"]      = report["best_case_min_ess"]
        base.loc[name, "n_lambda_weights_vars"]  = len(lag_vars)
        base.loc[name, "n_flagged_multimodal"]   = report["n_flagged"]
        base.loc[name, "n_resolved"]             = report["n_resolved"]
        # STRICTER than report["n_needs_deep_dive"]: recomputed here from n_stuck_fixable -
        # n_resolved rather than trusting each lag var's own n_stuck_unresolved (which is 0
        # whenever no resolution was even ATTEMPTED for that lag var, not just when nothing
        # needed it) -- an un-attempted stuck_fixable area is "known and fixable, just not
        # fixed yet" from check-multimodality's own point of view, but by the time you're
        # running THIS aggregate you're asking "is there anything left to do", so it counts
        # here regardless of whether --resolve was ever run (matches this function's own
        # documented, deliberate strictness vs. adjusted_diagnostics_report/check-multimodality
        # — see the module docs). The scalar path already applies this same strict convention
        # on its own (see _scalar_diagnostics_report), so this recomputation only changes the
        # lag-var contribution.
        base.loc[name, "n_needs_deep_dive"] = (
            report["n_needs_review"] + report["n_mixed"]
            + (report["n_stuck_fixable"] - report["n_resolved"]))
        base.loc[name, "n_not_multimodal"] = report["n_not_multimodal"]

    for col in ("n_bad_rhat", "min_ess", "raw_min_ess", "n_lambda_weights_vars",
               "n_flagged_multimodal", "n_resolved", "n_needs_deep_dive", "n_not_multimodal"):
        if col in base.columns:
            base[col] = base[col].astype(int)
    return base
