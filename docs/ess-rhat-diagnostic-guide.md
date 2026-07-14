# ESS / r-hat diagnostic guide

A general-purpose reference for diagnosing convergence problems (high r-hat, low ESS) in PyMC
models — not specific to the AZ family, though every worked example below comes from it
(`az-family-work-plan.md`, `az-ess-diagnosis.md`). Written after a deep diagnostic pass across
the whole AZ model family surfaced the same handful of root mechanisms repeatedly, in
different costumes. Consult this before starting a fresh diagnosis — the decision procedure in
§2 is meant to be followed in order, not skipped to a guess.

Grounded in the `pymc-modeling` skill's `references/diagnostics.md`, `troubleshooting.md`, and
`mixtures.md` (standard theory: Gelman-Rubin r-hat, Vehtari et al. 2021 rank-normalized
bulk/tail ESS, funnel geometry, label switching) — but the real value of this doc is the
patterns in §3, which go beyond that generic material. The standard references tell you what
r-hat and ESS measure and list textbook fixes (non-centered parameterization, tighter priors,
more tuning); they do not tell you how to tell which of six very-differently-shaped problems
you're looking at, or that some of the textbook fixes actively make specific problems worse.
That's what this document is for.

## Table of contents

- [§1 — Two checks before any real diagnosis starts](#1-two-checks-before-any-real-diagnosis-starts)
- [§2 — Decision procedure](#2-decision-procedure)
- [§3 — Pattern catalog](#3-pattern-catalog)
  - [Pattern 1 — Architectural hard multimodality (no sampling fix exists)](#pattern-1-architectural-hard-multimodality-no-sampling-fix-exists)
  - [Pattern 2 — Per-group ambiguity after partial pooling: genuine vs. spurious](#pattern-2-per-group-ambiguity-after-partial-pooling-genuine-vs-spurious)
  - [Pattern 3 — Fighting genuine ambiguity with a tighter prior or a hard cap fails, repeatedly](#pattern-3-fighting-genuine-ambiguity-with-a-tighter-prior-or-a-hard-cap-fails-repeatedly)
  - [Pattern 4 — A hard threshold/discontinuity creates a shallow-basin disagreement](#pattern-4-a-hard-thresholddiscontinuity-creates-a-shallow-basin-disagreement)
  - [Pattern 5 — Scale-parameter collapse toward a boundary: funnel vs. degenerate fixed point](#pattern-5-scale-parameter-collapse-toward-a-boundary-funnel-vs-degenerate-fixed-point)
  - [Pattern 6 — Cross-parameter/cross-source coupling masquerading as an unrelated problem](#pattern-6-cross-parametercross-source-coupling-masquerading-as-an-unrelated-problem)
  - [Pattern 7 — Composition leakage: combining validated pieces destabilizes previously-clean shared parameters](#pattern-7-composition-leakage-combining-validated-pieces-destabilizes-previously-clean-shared-parameters)
  - [Pattern 8 — Diagnostic tooling itself can hide the problem](#pattern-8-diagnostic-tooling-itself-can-hide-the-problem)
- [§4 — The log-likelihood-gap check, worked recipe](#4-the-log-likelihood-gap-check-worked-recipe)
- [§5 — Quick-reference table](#5-quick-reference-table)
- [§6 — When to stop fixing and report instead](#6-when-to-stop-fixing-and-report-instead)

## §1 — Two checks before any real diagnosis starts

**1. Are you actually looking at the whole model, or just the named scalars?**
`az.rhat`/`az.ess` run on whatever `var_names` you pass. A routine `diagnose` script that
restricts to a model's ~10 named scalars (for good reason — checking every per-(area,year)
Deterministic elementwise was measured at ~220s for 2 small models) can report "12 bad
parameters" while the real, cell-level picture is 26% of `z` bad and 85% of a per-cell
Deterministic bad (AZ4 — see `az-ess-diagnosis.md`'s Phase 4 section and the
`az4-diagnostics` artifact). Before concluding a model's convergence is "basically fine" or
"about as bad as X," rerun r-hat/ESS on every relevant array-shaped variable at least once,
even if it's slow. A tiny named-scalar summary is a starting point, never sufficient evidence
on its own.

**2. Divergences.** Zero, always, no exceptions in this codebase's history — every genuine
model in the AZ family samples with zero divergences even when r-hat/ESS are bad. That is
itself informative: divergences indicate the *sampler* failing inside a single trajectory
(numerical/geometric failure); their total absence throughout this investigation means every
bad-r-hat pattern found here is a *between-chain* or *between-mode* disagreement, not a
leapfrog-integrator problem. If you ever DO see divergences in this family, that's a
qualitatively different, unencountered-so-far failure mode — don't reflexively reach for the
patterns below.

## §2 — Decision procedure

Given a scalar or variable with r-hat > 1.01 or low ESS, work through this in order:

```
1. Compute per-chain autocorrelation (lags 1, 5, 10, 20, 50, 100) for one chain.
   ├─ Decays slowly (still 0.4-0.9 at lag 20-50)?
   │    → SLOW MIXING / FUNNEL geometry (§3, Pattern 5). A real precision problem;
   │      more draws / non-centering / a floor may genuinely help.
   └─ Decays to ~0 by lag 10-20?
        → Autocorrelation is NOT the problem. Go to step 2.

2. Compare chain means (or chain-level dominant category, for a discrete-ish choice
   like a simplex/lag-weight vector) — AND count within-chain switches (how many times
   the dominant category changes across a single chain's draws) before concluding
   "hard" multimodality; a genuinely bimodal marginal can still have every chain
   crossing both peaks (see Pattern 2c) rather than getting trapped in one.
   ├─ Chains cluster into 2+ DISCONNECTED groups, EACH CHAIN HAS ~0 SWITCHES (its
   │  whole trajectory sits in one group, no chain visits both)?
   │    → HARD MULTIMODALITY. Go to step 3 to find out if it's genuine.
   ├─ Chains are all individually bimodal but each one switches many times (tens+ per
   │  1500 draws) between the same two/three peaks, with no chain permanently stuck?
   │    → ROUND-TRIPPING MULTIMODALITY (Pattern 2c) — a different, cheaper case; the
   │      ordinary pooled posterior mean is likely already a valid mass estimate.
   └─ Chains agree on the broad region but their means/marginals are visibly offset,
      no chain is "stuck" anywhere specific?
        → SHALLOW BASIN (a milder, different problem). Go to step 4.

3. For hard multimodality: is the disconnection ARCHITECTURAL (one parameter is
   mechanically forced to represent what should be several different true values,
   e.g. one shared vector for 200 heterogeneous areas) or LOCAL (a hierarchy gives
   each group its own parameter, but a handful of groups' OWN data can't distinguish
   two explanations)?
   ├─ Architectural → no sampling fix exists (Pattern 1). The fix is a structural
   │  change (partial pooling, more groups, marginalization) not a bigger chain count.
   └─ Local → run the LOG-LIKELIHOOD-GAP CHECK (§4) on every flagged group before
      calling it "genuine ambiguity". A near-zero gap between the competing chain-
      groups' actual log-likelihood is genuine (Pattern 2a); a large gap (>~10 nats)
      means some chains are simply stuck in a worse mode (Pattern 2b) — a sampling
      failure that MORE/BETTER sampling can fix, not an epistemic limit. Don't
      report a "posterior mass split" for a flagged group until you've checked this.

4. For a shallow basin: is this parameter a SHARED/POPULATION-level quantity that
   pools information from many groups (a hierarchy's hyperparameter, a single boost/
   scale applied across a subset)?
   ├─ Yes → check whether a SUBSET of its informative groups is independently
   │  unstable (per step 2/3 above) and is leaking that instability upward
   │  (Pattern 7), and check whether the mechanism connecting groups to this
   │  parameter has a HARD THRESHOLD/DISCONTINUITY (a 0/1 indicator, a clip) that a
   │  smooth function could replace (Pattern 4).
   └─ No → check whether it's a scale parameter sitting suspiciously close to a
      boundary (near 0), and whether a FIXED POINT with real data mass coincides
      with that boundary (Pattern 5's collapse variant, not its funnel variant).

5. Whatever pattern you land on, before touching the model: check whether the
   "obvious" fix is a tighter prior or a harder constraint (smaller prior scale, a
   hard cap, more aggressive shrinkage). Patterns 2b and 3 below both show this class
   of fix FAILING or making things WORSE in this codebase, repeatedly, across
   unrelated parts of the model — treat "just constrain it more" as a hypothesis to
   test, not a default, and check the actual posterior (not just the prior) moved
   before trusting a tighter-prior fix.
```

## §3 — Pattern catalog

Each pattern: the symptom signature, the mechanism, a worked example, and what did/didn't fix
it. Ordered roughly by how fundamental the problem is (architecture, through geometry, to
diagnostic tooling itself).

### Pattern 1 — Architectural hard multimodality (no sampling fix exists)

**Signature**: r-hat stays bad no matter how many chains or draws; chains cleanly split into
groups with a stable ratio (not converging toward 1 as chains increase).
**Mechanism**: a single shared parameter is mechanically forced to represent what are actually
several different true group-level values. This isn't a sampling problem at all — the
posterior genuinely has separated peaks because the *model* says one thing must explain many
different realities.
**Example**: AZ1a's fully-pooled `lambda_weights_E` — one Dirichlet lag-weight vector shared
across 200 areas. 8-chain re-check found a stable **37.5%/62.5%** split (not 50/50, as the
noisier 4-chain run suggested) between "E is same-year" and "E lags by 1 year" — both real
population subgroups, permanently in tension under one shared parameter.
**Fix that worked**: partial pooling (AZ1b) — give every area its own parameter, shrunk toward
a shared kernel. Not "more chains", not a tighter prior — a genuinely different architecture.
**Diagnostic tell**: if increasing chains from 4→8 makes r-hat/ESS numerically *worse* (not
better) while making the reported split *more stable*, that's this pattern, not a
sampling-effort shortfall (AZ1a: max r-hat 1.74→1.65, min ESS 6→12 — "worse" r-hat with more
data is expected and correct here, since r-hat penalizes chain disagreement and 8 chains
surface the real split more clearly than 4 ever could).

### Pattern 2 — Per-group ambiguity after partial pooling: genuine vs. spurious

**Signature**: residual r-hat/ESS problems concentrated in a MINORITY of groups after
switching to a hierarchy (Pattern 1's fix); doesn't go away entirely.
**Mechanism**: with a genuine per-group parameter now available, groups with enough data
converge fine. Groups with too little data to distinguish between two roughly-equally-good
explanations can still show real, group-local multimodality — a smaller-scale version of
Pattern 1, now local instead of global.
**Example**: AZ1b (per-area hierarchical lag) — 34/200 areas flagged for `lag_P`, 31/200 for
`lag_E`. Individual chains spend all 1500 draws in one of two disconnected modes and never
cross over.

**2a — Genuinely tied (irreducible)**: the log-likelihood-gap check (§4) shows the competing
chain-groups' actual fit is nearly identical (<~2 nats). No amount of sampling resolves this —
the data really can't tell the two stories apart. **Report it, don't force it away.**
`hierarchical_mode_summary` (built this session, `diagnostics.py`) does this properly: per
flagged group, which category each chain committed to (≥95% purity threshold), grouped into
modes, with a chain-count-based mass estimate — but only trust that mass estimate with enough
chains (4 chains give a noisy 2-vs-2 that's close to a coin flip either way; 8 roughly halves
that noise).

**2b — Chains stuck in a worse mode (a real sampling failure, mislabeled as ambiguity)**:
the SAME symptom (r-hat bad, chains split) but the log-likelihood-gap check shows one group is
decisively better (10-35+ nats, i.e. astronomically more probable) — some chains simply never
found it. **This is not epistemic ambiguity and should not be reported as a "posterior mass
split."** Found by explicit test in this codebase, not assumed: for AZ1b's flagged areas,
median gap 6.35 nats (P) / 2.98 nats (E), with only 13-32% actually tied (<2 nats) and 10-27%
decisively split (>10 nats). **The single worst offender by this measure, in BOTH AZ1b and
AZ1d, for BOTH P and E, was the SAME area (E01033711)** — this round's own flagship
spike-tracking example — with gaps up to 35 nats. A "genuinely ambiguous" area quoted with
confidence in earlier reporting was, on this closer look, more likely a sampler failure for at
least its most dramatic instance.
**Fix, since validated (not just implied)**: more chains + seeding ALL chains' initial values
(`init_mean` in nutpie) toward the gap check's identified better-supported category, for
specifically the large-gap subset. Tried on AZ1d's persistent 13-area core (8 chains → 16, plus
seeding): areas the gap check called decisive mostly resolved to near-unanimous chain
agreement (E01033711's 35-nat gap → 16/16 chains agree, up from a ~7/8 split; several others to
81-94%), while areas the gap check called genuinely tied stayed split even when seeded — because
there's no "right" mode to seed toward. **This is a rare case of a diagnostic check
cross-validating itself**: the classification (2a vs 2b) predicted which areas would respond to
seeding, and the seeding experiment confirmed the prediction area-by-area, not just in
aggregate. Named-scalar ESS also improved 5-6x. One exception found: a genuinely 3-way (not
2-way) split area didn't resolve with a 2-category seed — check `hierarchical_mode_summary`'s
`n_modes` for the flagged group before assuming a simple majority-vs-minority seed is the right
shape of intervention.

**How to tell 2a from 2b**: compute each chain's dominant category (or mode), group chains by
it, and compare each group's own mean total log-likelihood (sum over the relevant
observations) using that model's `log_likelihood` group in the trace (works directly whenever
the likelihood is a plain observed RV, not a `pm.Potential` — for Potential-based mixtures you
need the closed-form logp reconstruction approach used for AZ3's E01002702 deep-dive instead).
A gap under ~2 nats is 2a; a gap over ~10 nats is 2b; in between, treat cautiously.

**2c — Round-tripping: chains individually cross between two genuine modes, none of them
trapped.** A third, distinct signature found by checking within-chain switch counts before
assuming 2a/2b's "chains never cross" framing applies universally: a variable can be genuinely
bimodal at the individual-draw level (confirmed via a KDE peak check on the pooled marginal —
real separated peaks, not one broad hump) while every chain successfully visits both peaks many
times over its own run, rather than getting stuck in either. **Example**: AZ1d's
`lag_P_lambda_weights` for `E01035649`/`E01035646` — 8 chains each, switching 26-55 and 82-108
times respectively across 1500 draws (vs. `E01002702`'s hard-trapped case in the SAME model,
same variable, same run: all 8 chains show **zero** switches). This matters practically because
it changes what's needed for a mass estimate: since each round-tripping chain is (given enough
switches to be plausibly ergodic) already sampling both modes in roughly their true relative
proportion, **the ordinary pooled posterior mean is a legitimate direct mass estimate** — check
its ESS is adequate (100+ is a reasonable bar) and report it, no chain-fraction counting or
bridge sampling needed. This is a materially cheaper case than 2a/2b and should be told apart
from them before reaching for either's heavier treatment (see
`docs/multimodality-characterization-guide.md` for the full recipe once you're past this
diagnostic step).

**A caveat on the log-likelihood-gap check itself, found by using it on a case where the
flagged variable has a real indirect path to another likelihood term**: the recipe below
computes the gap using ONLY the likelihood the flagged variable is most directly attached to
(e.g. `P_like` for a `lag_P` variable). If that variable also indirectly affects a *different*
observed likelihood through a shared latent (e.g. the lag category pulls `z` in different
directions, and `z` also feeds `E_like`, which has no lag mechanism of its own), the single-
likelihood gap can badly understate how resolved the two modes really are. **Example**:
AZ1d's `E01002702` — the `P_like`-only gap is 0.32-0.43 nats (genuinely tied, reproduced
identically across two independent sampling runs), but the full `P_like`+`E_like` gap is ~7.5
nats (a real, non-decisive lean toward one mode). Neither number is "wrong" — they answer
different questions (does the variable's own direct data distinguish the modes? vs. does the
FULL joint posterior, as actually reported downstream, distinguish them?) — but only the
fuller measure reflects what a consumer of the model's output would actually see. **Compute
the gap on every observed likelihood downstream of the shared latent, not just the one the
flagged variable is nominally "about," whenever such a path exists.**

### Pattern 3 — Fighting genuine ambiguity with a tighter prior or a hard cap fails, repeatedly

**Signature**: tightening a hierarchy's between-group prior scale, or hard-capping it, makes
max r-hat and min ESS *worse*, not better, and the posterior for the constrained quantity pins
against the new constraint.
**Mechanism**: when per-group divergence is driven by genuine likelihood-level ambiguity
(Pattern 2a/architecture, not prior looseness), a softer prior barely moves the posterior — the
likelihood dominates — so tightening it just makes the geometry harder to sample without
removing the ambiguity. A HARD cap goes further: it forcibly relocates the unresolved tension
from the per-group parameter to the population-level hyperparameter instead of removing it,
since the groups that "want" to diverge sharply still do, now via a different channel.
**Examples, tested twice, same result both times**:
- AZ1b: `tau_sigma` 1.5→0.5 (softer prior). Posterior `tau` barely moved (e.g. `lag_E_tau`
  ~2.7-3.2 vs ~5.1-5.9) — max r-hat 1.12→1.24, min ESS 23→12. Worse.
- AZ1c (hard cap, `tau = 1.5 × Beta(2,2)`): tau pinned right at the 1.5 ceiling on every
  chain — max r-hat 1.19→1.32, min ESS 29→19, and the population-level `mu_logit` (previously
  fine) got worse (r-hat up to 1.32) as the relocated tension landed there. Worse again.
- AZ4b: the identical hard cap, applied inside the larger combined model, to test whether it
  might still help THERE even though it lost on its own turf — max r-hat 1.168→1.459, min ESS
  31→15. Worse a third time, confirming this isn't specific to AZ1b's exact setting.
**Lesson**: treat "tighten the prior" or "cap it" as a hypothesis, always check whether the
POSTERIOR (not just the prior) actually moved, and be prepared for tightening to make sampling
harder without buying anything. The fix that actually worked for the analogous problem
(Pattern 6 below, E's lag) was architectural — remove the mechanism causing the ambiguity
entirely for the disproportionately-affected part — not constrain it.

**A fourth attempt extends the same lesson to the OPPOSITE direction — loosening
asymmetrically, not uniformly, also failed, for a different and instructive reason.** Rather
than constrain every group's divergence identically (which failed 3 times above), tried
letting divergence vary PER GROUP via a horseshoe-style local/global decomposition (`tau[a,k] =
global_tau[k] * local_scale[a,k]`, `local_scale ~ HalfCauchy(1)`) — the intent was to let the
~13-area ambiguous core diverge freely while shrinking the other 187 areas tighter than a
uniform tau ever could. This is architecturally different from Pattern 3's cap (which
constrains everyone equally), not just a variant of it — and it still failed, but via a
DIFFERENT mechanism: an unregularized `HalfCauchy` local scale exploded (observed range 2e-6 to
**3.09 million** across posterior draws), producing **the first divergences seen anywhere in
this model family** (scattered 1-5 per chain, not clustered — a step-size struggle in the
extreme tail, distinct from a funnel-at-low-sigma). max r-hat and min ESS both got worse than
the model it was meant to fix, and year-allocation confidence became the worst of any model
checked this round (48.5% low-confidence). This is the textbook horseshoe-prior pathology (see
`troubleshooting.md`'s "Horseshoe Prior Challenge" and the pymc-extras skill) rearing up exactly
where the literature says it will, not a novel failure — but it's worth having on record as a
LIVE example of "a more sophisticated reparameterization is not automatically safer than a
naive one", extending Pattern 3's lesson from simple tightening/capping to more elaborate
alternatives too. A regularized horseshoe (with a slab scale bounding the local multiplier) is
the standard fix for this specific failure and was deliberately not tried on the first pass —
worth attempting before concluding per-group asymmetric shrinkage is a dead end, but the plain
version is not viable as tested.

**A fifth attempt changed WHAT the likelihood represents rather than the prior, and failed the
most badly of all four — for a mechanism worth its own line item.** Even after Pattern 2b's
validated fix (more chains + informed init), the shared population scalars (`lag_P_mu_logit`)
stayed at r-hat 1.02-1.03 — genuine per-area ambiguity that can't be resolved by sampling harder
was still leaking into the population level. Hypothesis: the likelihood mean-mixes lag
categories into one blended prediction before scoring it (`P_mean = sum_k lambda_k *
z_shifted_k`, one density around that blend) — a known anti-pattern for discrete-category
uncertainty, since it scores observations against a value no single category predicts. Tried
marginalizing instead (`log p(P_obs) = logsumexp_k(log lambda_k + logT(P_obs; z_shifted_k,
sigma))`, mixture of densities, not a blended mean) — architecturally distinct from every prior
attempt in this pattern (those all touched the PRIOR on tau; this changed the LIKELIHOOD's
functional form). Result: worse on every axis, badly — max r-hat 1.04→1.17, min ESS 169→33,
low-year-confidence 7.0%→33.0%, and `hierarchical_mode_summary` (a targeted diagnostic that
previously flagged a specific 16-area minority) now flags **all 200 areas**. Root cause
confirmed directly (not just inferred from the aggregate numbers): `sigma_plan` collapsed
5.01→0.72 while `lag_P_tau` collapsed to ~0.1 with excellent ESS/r-hat (a confident, converged
collapse, not noise) — marginalizing at the observation level let each individual (area, year)
term pick whichever category fit it best independently, destroying the SAME cross-year
consistency-of-commitment that gave the shared per-area lambda any meaning, and letting the
model fit almost every point opportunistically instead of via genuine signal. The spike-tracking
plot confirmed the practical cost: several areas now badly under-track real, large spikes they'd
tracked fine before (e.g. one area's P_obs=762 spike, posterior z reaches only 26). **Three
architecturally distinct interventions on this same hierarchical lag structure have now failed
via three distinct mechanisms** (tighter/capped prior fights genuine ambiguity; unregularized
horseshoe explodes in the tail; marginalized likelihood destroys cross-year identification) —
strong evidence that the residual ambiguity in this family is a property of the DATA (some areas
genuinely can't be assigned a year with confidence), not of any one modelling choice around it.

### Pattern 4 — A hard threshold/discontinuity creates a shallow-basin disagreement

**Signature**: a shared scale parameter shows small, PERSISTENT between-chain disagreement —
chain means stable but not converging toward each other over the whole run — with clean
autocorrelation (rules out slow mixing) and no clean multimodal split (rules out Pattern 1/2).
**Mechanism**: a fixed 0/1 indicator (e.g. "is this area in the top quartile of |D|?") makes a
downstream parameter's effect on the likelihood discontinuous right at the cutoff — areas
straddling the boundary contribute a sharp kink to the log-likelihood surface as a function of
the shared parameter, which is exactly the kind of geometry that produces a shallow,
hard-to-mix ridge.
**Example**: AZ2's `sigma_delta_top_boost`, a single HalfNormal scalar added only to the top
quartile of |D| via a hard `is_top` mask — min ESS 47 (bulk), chain means 28.5-30.3 (~6% of
their mean) never converging. Ruled out slow mixing (autocorrelation ~0 past lag 20) and a
pairwise ridge with `sigma_plan`/`sigma_ben` (all |r| < 0.12) before landing on the
discontinuity as the remaining candidate.
**Fix that worked**: replace the hard cutoff with a smooth logistic ramp over the rank
percentile of |D| (AZ2b) — same number of parameters, no new degrees of freedom, just smoothed
the boundary. `top_boost`'s bulk ESS went 47→605 (13x), chain-mean spread tightened from ~6% to
~1.3% of the mean. Directly confirms the discontinuity, not the boost mechanism per se, was
the cause.
**Generalization**: any fixed indicator/mask feeding into a shared parameter (quartile cutoffs,
thresholds, active/inactive splits) is a candidate for this pattern — check whether replacing
the hard split with a smooth function of the same underlying continuous quantity helps before
assuming the mechanism itself needs to be more complex.

### Pattern 5 — Scale-parameter collapse toward a boundary: funnel vs. degenerate fixed point

Two related but distinguishable sub-patterns, both showing a scale parameter drifting toward
its lower boundary — but with different mechanisms and different correct fixes.

**5a — Ordinary funnel (real, well-understood geometry)**: high, SLOWLY-decaying
autocorrelation (still 0.4-0.7 at lag 20, ~0 only by lag 100-200) is the tell — this is the
standard centered-hierarchy funnel from the general PyMC literature (small group-level
variance constrains individual effects to a narrow region NUTS struggles to step through).
Standard fix: non-centered parameterization (already used everywhere in this codebase) —
if you're already non-centered and still see this signature, the funnel may be coming from
somewhere less obvious (see AZ3 below, where a mixture likelihood's OWN structure, not the
z-prior's, was the actual source).
**Example**: AZ3's unfloored `sigma_plan`, collapsed to ~0.58 (an order of magnitude below
every other AZ0a-family model's ~7-9), autocorrelation 0.88-0.95 at lag 1, still 0.4-0.7 at lag
20, ~0 only by lag 100-200 — the classic slow-funnel signature. Root cause here wasn't the
z-prior's own parameterization (already non-centered) but the interaction with a NOISE-MIXTURE
likelihood: with a noise branch available to absorb outliers, the remaining "signal" residuals
shrink, and a scale parameter fit to those residuals gets pulled tight — coupling strongly to
~1000 per-cell signal/noise decisions simultaneously (confirmed `corr(sigma_plan, rho_P) =
0.43`), exactly the geometry NUTS struggles with.

**5b — Degenerate fixed-point collapse (a much more severe, qualitatively different failure)**:
a scale parameter for a distribution centred at a FIXED point (not a moving per-observation
target) can have UNBOUNDED density exactly at that point as its own scale shrinks — if a large
mass of real data sits exactly at that fixed point, the free parameter can collapse onto a
literal density singularity, not just a tight-but-finite region.
**Example**: AZ0's original 3-way mixture — the noise branch was `StudentT(mu=0, sigma=
sigma_noise)`, and 54.3%/28.2% of P_obs/E_obs cells are EXACTLY 0. `sigma_noise` had nothing
stopping it from shrinking toward 0, at which point its density at those exact-zero
observations diverges. Result: max r-hat 17.8, ~83% divergent draws — categorically worse than
5a's slow-funnel signature (which never produced ANY divergences throughout this whole
investigation).
**The fix that matters: a floor is not the same thing as a tighter prior scale, and only one
of them works.** `HalfNormal(small_sigma)` alone still has its MODE at exactly 0 regardless of
how small its own sigma parameter is — a "smaller prior" doesn't move the mode away from the
dangerous fixed point at all. What actually worked (AZ3, and later AZ4/AZ4b's `sigma_plan`/
`sigma_ben`): a **hard additive constant**, `sigma = floor + HalfNormal(excess_scale)` — this
makes the value at the fixed point structurally unreachable, independent of what the
`HalfNormal` excess component does. `sigma_noise_floor=25` (well above every other model's
converged ~7-9, so the noise branch stays unambiguously more tolerant than signal, not a
near-duplicate of it) and later `sigma_obs_floor=2.0` for `sigma_plan`/`sigma_ben` both used
this same construction, and both worked (AZ3's `sigma_plan` ESS 43→6086 after flooring).
**Diagnostic tell for 5b specifically**: check whether the collapsing parameter is the scale of
a distribution centered at a FIXED point (0, or any other constant), and whether a large mass
of real observations sits exactly at that point. If both are true, a floor is very likely
required — a tighter prior alone will not fix it, and may reproduce the collapse at a slightly
different, still-dangerous location.
**A follow-up nuance, found by actually checking rather than assuming the floor value is still
"working" the way it looks**: after flooring, `sigma_plan`/`sigma_ben` sat with tiny SD right
at 2.01/2.02 (essentially AT the floor), which looks alarming (still "collapsed", just capped)
— but a direct profile-log-likelihood sweep (hold `z`/`rho` fixed, scan candidate sigma values)
showed the true local optimum was ~2.5, barely above the floor, not near 0. **A parameter
sitting at its floor does not automatically mean the floor is doing dramatic work — check the
profile likelihood before concluding the floor is "actively binding."**

### Pattern 6 — Cross-parameter/cross-source coupling masquerading as an unrelated problem

**Signature**: a parameter or group of parameters shows convergence problems that, once a
neighbouring/related mechanism is removed, shrink dramatically MORE than that mechanism's own
share of the problem would predict.
**Mechanism**: two nominally-separate parameters (e.g. two data sources' own lag categories)
share a downstream dependency (a common `z`). One source's genuine ambiguity can inflate the
OTHER source's apparent instability purely through that shared path, without the other source
having any real ambiguity of its own.
**Example**: AZ1b's `lag_P` flagged 34 areas. After removing E's lag mechanism entirely (AZ1d),
only 16 areas remained flagged for `lag_P` — and critically, **13 of those 16 (81%) were
ALREADY flagged in AZ1b**, meaning the other 21 of AZ1b's original 34 "P-ambiguous" areas were
never really about P at all; they were E's ambiguity leaking into P's own convergence via
their shared `z`. The persisting 13-area core has median |D| = 410 (7.6x the dataset median) —
concentrated exactly where large real spikes make "which year" a meaningful question, unlike
the 21 resolved areas (median |D| = 170, closer to typical).
**How to check**: before concluding parameter A has its own convergence problem, check what
fraction of A's flagged instances survive after removing or fixing a plausibly-coupled
parameter B (same underlying group index, shared latent variable, or a Deterministic each
depends on). If most of A's problem disappears along with B's, A didn't really have the
problem in the first place.
**Practical implication**: this is also a legitimate, asymmetric FIX in its own right when one
side of a coupled pair is disproportionately responsible (see AZ1d) — remove the disproportionate
side's mechanism rather than constraining both sides symmetrically (contrast with Pattern 3,
where symmetric constraint failed twice).

### Pattern 7 — Composition leakage: combining validated pieces destabilizes previously-clean shared parameters

**Signature**: a combined model shows a *previously clean* shared/population parameter (fine
in each contributing piece tested alone) develop a serious ESS/r-hat problem once combined —
and the combined model's own top-line diagnostic table doesn't obviously flag WHY, because the
degradation isn't in the piece that changed the most.
**Mechanism**: a shared parameter that draws information from many groups (all 200 areas, or a
top-quartile subset) is only as stable as its LEAST stable informative groups. If a different
component of the combined model destabilizes a subset of those groups (e.g. per-area lag
ambiguity making some areas' z estimates unstable), and that subset heavily overlaps with the
shared parameter's informative set, the parameter inherits that instability wholesale — even
though, considered alone, that parameter's own construction never changed.
**Example**: AZ4 (AZ2b's top-boost z-prior + AZ1b's lag + AZ3's noise-mixture, combined).
`sigma_delta_top_boost` (ESS 605 in AZ2b alone) collapsed to ESS 40; `sigma_noise_E` (ESS>3600
in AZ3 alone) collapsed to ESS 33. Traced directly: 96% of areas with bad `z` r-hat also have
bad lag-weight r-hat in the SAME area (the lag mechanism's own known ambiguity), and 92% of the
top-quartile-|D| areas that feed `top_boost` are lag-ambiguous — nearly all of that shared
parameter's informative areas were unstable simultaneously. The asymmetry (E's
`sigma_noise_E`, not P's `sigma_noise_P`, collapsed) tracked directly onto which source's lag
was more genuinely ambiguous (Pattern 2/6) — the SAME root cause resurfacing through a new
channel, not an independent new problem.
**How to check**: for any shared parameter that degrades upon combination, identify its full
set of informative groups (every group whose likelihood term feeds it, directly or via a
Deterministic), and check what fraction of THOSE groups are independently flagged unstable by
some other component. A high overlap (as opposed to a random sample of groups) is the tell.
**Fix that failed**: hard-capping the coupling mechanism's own divergence (AZ4b, i.e. Pattern
3's fix applied to the leakage source) — made it worse, exactly as in isolation.
**Fix strongly implied but not yet built**: apply Pattern 6's asymmetric fix (remove the
disproportionately-ambiguous side's mechanism, i.e. rebuild the combination using AZ1d's design
rather than AZ1b's) — targets the actual leakage source rather than either accepting the
combined degradation or fighting it with a blunter constraint.

### Pattern 8 — Diagnostic tooling itself can hide the problem

Not a sampling pathology — a measurement pitfall, but one that changed the practical read of a
model's health more than any single sampling fix in this whole investigation.
**Signature**: a routine diagnostics table reports a small number of "bad" parameters; a full
cell-level check finds the true fraction is an order of magnitude larger.
**Mechanism**: `az.rhat`/`az.ess` (and this codebase's own `diagnostics_summary`) only check
whichever `var_names` are passed. Restricting to a model's ~10 named scalars is a deliberate
and reasonable default for speed (checking every per-cell Deterministic elementwise measured at
~220s for 2 small models) — but it silently excludes `z`, `delta`, `resp_noise_P/E`, and every
per-area lag-weight vector, which is exactly where architecturally-caused instability (Pattern
1, 2, 6, 7) actually lives.
**Example**: AZ4's routine `diagnose` run reported `n_bad_rhat=12`. A full per-cell check found
26% of `z` cells, 32-38% of lag-weight cells, and **85% of `resp_noise_E` cells** bad — a
completely different picture of the model's reliability than the summary table alone implied.
**Standing practice, not optional**: before concluding "this model is fine" or "this model is
about as bad as X" from a diagnostics table, rerun r-hat/ESS on every relevant array-shaped
variable at least once. Budget the ~1-4 minutes this costs per model; it has changed the
conclusion every time it's been done in this investigation.

## §4 — The log-likelihood-gap check, worked recipe

Used repeatedly above (Pattern 2) and worth having as a standalone recipe, since it isn't in
any of the standard PyMC reference material consulted for this guide:

```python
import arviz as az
import numpy as np

tr = az.from_netcdf(...)
lw = tr.posterior['some_per_group_simplex_or_category_var'].values  # (chain, draw, group, k)
loglik = tr.log_likelihood['some_observed_rv_name'].values           # (chain, draw, group, obs)

for group_idx in flagged_groups:  # from hierarchical_mode_summary or a manual r-hat scan
    chain_dominant_category = lw[:, :, group_idx, :].mean(axis=1).argmax(axis=1)  # (chain,)
    unique_cats = np.unique(chain_dominant_category)
    if len(unique_cats) < 2:
        continue  # chains agree on the mode despite flagged r-hat -- soft disagreement, not hard multimodality
    ll_by_cat = {
        cat: loglik[np.where(chain_dominant_category == cat)[0], :, group_idx, :].sum(axis=-1).mean()
        for cat in unique_cats
    }
    gap = max(ll_by_cat.values()) - min(ll_by_cat.values())
    # gap < ~2 nats  -> genuinely tied (Pattern 2a), report the ambiguity honestly
    # gap > ~10 nats -> some chains stuck in a decisively worse mode (Pattern 2b), a sampling
    #                   failure -- more chains / better init is the untested next lever
```

**Run this once per relevant observed likelihood, not just the one the flagged variable is
most obviously attached to** — see Pattern 2c's caveat above: `loglik` here is deliberately a
single named RV (`'some_observed_rv_name'`), and if the flagged variable has an indirect path
to a *different* observed RV via a shared latent, repeat this with that RV's `log_likelihood`
too (or sum both) before trusting a "genuinely tied" conclusion from the narrower version alone.

Requires the likelihood to be a plain observed RV (`pm.StudentT(..., observed=...)`, etc.), so
`tr.log_likelihood` is populated automatically. For `pm.Potential`-based marginalized mixtures
(no automatic per-observation log-likelihood), reconstruct the relevant logp terms in closed
form instead (see AZ3's E01002702 deep-dive in `az-family-work-plan.md` for a worked example:
because `sigma_delta` there was a fixed function of `D`, not sampled, the ZeroSumNormal prior
term and the mixture-likelihood terms were both computable exactly for a specific candidate
configuration, without needing this trace-based approach at all).

**This recipe, the within-chain switch count, `hierarchical_mode_summary`, and the validated
informed-init reseeding fix are now assembled into one reusable, tested pipeline** —
`housing_projections.multimodality` / `housing-projections check-multimodality` — rather than a
fresh ad hoc script per model. See `docs/multimodality-diagnostic-pipeline.md` for the full
walkthrough, category definitions, and worked example; reach for it directly instead of
re-deriving this recipe by hand next time Pattern 2 comes up.

## §5 — Quick-reference table

| Symptom | Likely pattern | Confirm with | Fix that worked | Fix that failed |
|---|---|---|---|---|
| r-hat won't budge with more chains; stable split ratio | 1 (architectural) | chain-mean clustering at 4 vs 8 chains | partial pooling / genuine per-group parameter | more chains alone, tighter prior |
| Residual multimodality in a MINORITY of groups after pooling | 2 | log-likelihood-gap check (§4, on EVERY downstream likelihood, not just the obvious one) + within-chain switch count | 2a: report honestly (`hierarchical_mode_summary`); 2b: more chains + seed `init_mean` toward the identified better mode (validated); 2c: chains round-trip — the pooled posterior mean is already a valid estimate, check its ESS | tighter prior, hard cap, unregularized horseshoe, marginalized/logsumexp likelihood (all Pattern 3) |
| Tightening a hierarchy's prior scale makes r-hat/ESS worse | 3 | check if posterior actually moved under the tighter prior | remove the disproportionate side's mechanism (Pattern 6) | more of the same tightening; per-group horseshoe; marginalizing the likelihood (each fails via a different mechanism — see Pattern 3) |
| Small persistent between-chain disagreement, clean autocorrelation, no clean mode split | 4 | look for a fixed 0/1 indicator/threshold feeding the parameter | smooth ramp replacing the hard cutoff | — |
| Scale parameter tight, high autocorrelation decaying slowly (lag 100-200) | 5a | autocorrelation profile | non-centered reparameterization; if already non-centered, check for a coupled mixture/likelihood interaction | — |
| Scale parameter collapsing toward 0, catastrophic (divergences, r-hat >10) | 5b | check for a fixed-point-centered branch + real data mass there | hard additive floor (`floor + HalfNormal(excess)`) | smaller prior sigma alone |
| Parameter A's flagged-group count shrinks a lot when unrelated parameter B is fixed | 6 | re-run A's diagnostics after fixing/removing B | remove the disproportionate side asymmetrically | symmetric constraint on both sides |
| Combining validated pieces destabilizes a previously-clean shared scalar | 7 | check overlap between the scalar's informative groups and another component's flagged groups | (in progress) asymmetric fix from the identified leakage source | hard-capping the leakage mechanism |
| Diagnostics table looks fine/mediocre but something still feels off | 8 | rerun r-hat/ESS on every array-shaped variable, not just named scalars | (measurement fix, not a model fix) | trusting the named-scalar summary alone |

## §6 — When to stop fixing and report instead

Not every bad r-hat is worth chasing further. Stop and report honestly (per Pattern 2a) when:
the log-likelihood-gap check shows genuine near-tied ambiguity; the total quantity of interest
(e.g. an area's total change, pinned by a hard constraint like a census total) is unaffected
regardless of which mode is "true"; and the ambiguity affects a small, identifiable, named
minority of groups rather than being pervasive. Keep pushing when: the gap check shows chains
stuck in a decisively worse mode (Pattern 2b — a real, likely-fixable sampling failure); a
structural discontinuity (Pattern 4) or degenerate fixed point (Pattern 5b) is identifiable and
untried; or a combination's leakage (Pattern 7) traces to a specific, already-understood
component problem with a known asymmetric fix (Pattern 6) not yet applied.

**Once you've concluded "stop and report" for genuine multimodality (Pattern 1 or 2a) and want
to go further than a chain-count vote share** — mapping every mode, estimating each one's
relative posterior mass with a validated method, and confirming the mode structure survives
prior-sensitivity and per-mode posterior-predictive checks — see
`docs/multimodality-characterization-guide.md`, which picks up exactly at this handoff point.
