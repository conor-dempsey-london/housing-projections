# Characterizing genuine multimodality

**Scope.** `docs/ess-rhat-diagnostic-guide.md` diagnoses *why* r-hat/ESS look bad and
classifies the mechanism (Patterns 1-8). Its own §2 decision procedure and §6 ("When to
stop fixing and report instead") both terminate, for a real subset of cases, at the same
conclusion: **the posterior has more than one genuinely separated mode, and no sampling
fix removes that** — Pattern 1 (architectural: AZ1a's `lambda_weights_E`), Pattern 2a
(local/per-group: AZ1b/AZ1d's ~13-area lag-category core), and AZ3's per-cell
`detect_z_multimodality` findings all land here. This document starts exactly where that
one stops: **you have already concluded, via the log-likelihood-gap check or an
architectural argument, that a posterior is genuinely multimodal — not a sampling failure,
not a labeling artifact (not yet checked — that's Step 1 below), not something a longer
run fixes. What do you do with that, beyond quoting a chain-count vote share?**

Every method below is a real technique with a real literature behind it, but they are not
equally worth their cost, and several of the "textbook" options either don't apply to this
codebase's models as shipped (no native PyMC support) or have documented failure modes of
their own. This guide orders them by cost and tells you which to skip.

**Automated first pass available**: `housing_projections.multimodality` /
`housing-projections check-multimodality` now runs Step 1's own within-chain switch-rate check
plus the log-likelihood-gap check automatically for every area a model flags, and classifies
each into `hard_genuine` / `stuck_fixable` / `round_tripping` / `mixed` / `needs_review` (see
`docs/multimodality-diagnostic-pipeline.md`) — including `resolve_stuck_areas`, the validated
informed-init fix for the `stuck_fixable` subset. Run that first; treat this document's manual
Steps 1-6 as what to do with whatever the pipeline puts in `needs_review`, or when a model's
scale differs enough from the AZ1 family that the pipeline's thresholds need re-deriving rather
than trusting as-is.

## Table of contents

- [§0 — Before you start: is this worth the full treatment?](#0-before-you-start)
- [§1 — The unified procedure, at a glance](#1-the-unified-procedure)
- §2 — Step-by-step in detail
  - [Step 1 — Cheap artifact checks first](#step-1-artifact-checks)
  - [Step 2 — Map the modes systematically](#step-2-map-modes)
  - [Step 3 — Sample and diagnose each mode individually](#step-3-per-mode-sampling)
  - [Step 4 — Estimate relative posterior mass across modes](#step-4-mass-estimation)
  - [Step 5 — Validate the mode structure isn't still an artifact](#step-5-validate)
  - [Step 6 — Report](#step-6-report)
- §3 — Mass-estimation method catalog
  - [3.1 Naive chain-fraction — screen only](#31-naive-chain-fraction)
  - [3.2 Laplace/BIC screen per mode — cheap sanity check](#32-laplace-bic)
  - [3.3 Bridge sampling — recommended default](#33-bridge-sampling)
  - [3.4 Stacking across per-mode sub-posteriors](#34-stacking)
  - [3.5 SMC (`pm.sample_smc`) — native but dimensionality-biased](#35-smc)
  - [3.6 Thermodynamic integration / stepping-stone — expensive](#36-thermodynamic-integration)
  - [3.7 Parallel/simulated tempering — not recommended here](#37-parallel-tempering)
  - [3.8 Nested sampling / pocoMC — the escalation path](#38-nested-sampling)
- §4 — Worked recipes
  - [4.1 Bridge sampling between two modes](#41-bridge-sampling-recipe)
  - [4.2 Cross-mode ensembling via `sensitivity.py`](#42-cross-mode-ensembling)
  - [4.3 Symmetry check for label-switching](#43-symmetry-check)
  - [4.4 Boundary-confined sampling for a clean per-mode read](#44-boundary-confined-sampling)
- [§5 — Quick-reference / cost table](#5-quick-reference)
- [§6 — Reporting template: the "mode card"](#6-mode-card)
- [§7 — Is this a persistent domain problem? Should the default tool change?](#7-persistent-problem)
- [§8 — When to stop](#8-when-to-stop)

---

<a id="0-before-you-start"></a>
## §0 — Before you start: is this worth the full treatment?

Cheap, and should always be true before investing in anything below:

1. **Re-run `ess-rhat-diagnostic-guide.md` §2 to completion first.** If you're looking at
   Pattern 2b (chains stuck in a decisively worse mode — log-likelihood gap > ~10 nats),
   this isn't genuine multimodality at all; more chains + `init_mean` seeding is the fix
   (already validated in this codebase: AZ1d's 13-area core, gaps up to 35 nats, resolved
   to 94-100% chain agreement after seeding). Don't run bridge sampling on a sampling bug.
2. **Check Pattern 8** — are you looking at the whole model (every array-shaped variable),
   or just named scalars? A "multimodal" named scalar can be an artifact of a routine
   `diagnose` script that never looked at `z` itself.
3. **Is the total quantity anyone actually cares about pinned regardless of which mode is
   true?** In this codebase, an area's total `z` sum is always exactly pinned to `D` by
   the zero-sum construction — only *which year* gets credit is ambiguous. If the
   consumer of the model only needs the pinned quantity, the honest answer may be "report
   the ambiguity in one sentence and stop" (§8) rather than running SMC.
4. **Rough cost budget.** Read §5's table before picking a method — the honest range here
   is "five minutes" (naive chain-fraction, already known unreliable) to "a full day of
   compute" (thermodynamic integration on a high-dimensional joint). Decide what the
   finding is worth before starting.

If all four check out and the ambiguity is worth characterizing precisely (a named,
recurring model behavior; a stakeholder-facing claim; a candidate for changing the
model's structure), proceed.

---

<a id="1-the-unified-procedure"></a>
## §1 — The unified procedure, at a glance

| Step | What | Cost | Detail |
|---|---|---|---|
| 0 | Gate: confirm it's worth it (§0) | minutes | above |
| 1 | Cheap artifact checks *before* mapping modes: label-switching/symmetry, reparameterization collapse | minutes-hours | §2.1 |
| 2 | Map modes: multi-path Pathfinder/ADVI restarts → dispersed chains → (only if low-dim) grid search | minutes-hours | §2.2 |
| 3 | Sample + diagnose each mode individually | ~1-2x one normal sampling run per mode | §2.3 |
| 4 | Estimate relative mass: naive fraction (screen only) → Laplace/BIC screen → bridge sampling or stacking → SMC → (rarely) TI/nested sampling | minutes to hours, see §3/§5 | §2.4, §3 |
| 5 | Validate genuineness: prior sensitivity, per-mode PPC, substantive plausibility, (optional) held-out check | hours | §2.5 |
| 6 | Report: one "mode card" per mode with a reliability tag on the mass estimate | — | §2.6 |
| 7 | If this recurs across models: consider a default-tool change (SMC/pocoMC), scoped narrowly | — | §7 |

Steps 1-2 are cheap and can invalidate the whole exercise (a relabeling artifact needs no
mode-mass estimate at all) — always run them before step 4's more expensive methods, even
though the user's original ordering put the "is it an artifact" check after mass
estimation. That ordering is worth deviating from: bridge sampling or SMC spent on two
modes that turn out to be the same mode under a parameter symmetry is wasted compute.

---

## §2 — Step-by-step in detail

<a id="step-1-artifact-checks"></a>
### Step 1 — Cheap artifact checks first

**1a. Label switching / exact symmetry.** Ask: is there a transformation `g` such that
`p(g(θ) | data) = p(θ | data)` for *every* θ — i.e., does swapping the two "modes"' values
leave the model's log-density exactly unchanged? This is a five-minute numerical check, not
a judgment call:

```python
logp_fn = model.compile_logp()
draw_a = {name: mode_a_draws[name] for name in model.free_RVs}
draw_b = swap_transform(draw_a)   # apply the candidate symmetry g
print(logp_fn(draw_a), logp_fn(draw_b))   # equal (to float tolerance) => literal relabeling
```

If `logp` is bit-identical under the swap, this is Stephens-style label switching (see
`pymc-modeling`'s `references/mixtures.md`: exchangeable mixture components create `K!`
equivalent modes with meaningless per-component summaries) — fix by imposing an ordering
constraint ex-ante or relabeling post-hoc (the "label.switching"-package/Stephens 2000
pivot algorithm the skill's own reference points to), and **stop — there is no genuine
multimodality left to characterize once relabeled.**

Important negative result to check for, not assume: **most of this codebase's own cases
are NOT literal label switching.** AZ1a's "E is same-year" vs "E lags by 1 year" are not
exchangeable under the likelihood — swapping which category is called "0" and which is
called "1" changes which observation-years get compared to which `z` values, so `logp`
changes under the swap. Run the check anyway; it's cheap and a wrong assumption here would
invalidate everything downstream.

**1b. Reparameterization collapse (a ridge or funnel masquerading as two modes).** Before
trusting a 1-D marginal's bimodality, look at the *joint* posterior of every parameter pair
implicated in the split, pooling across all chains (not per-chain):

```python
az.plot_pair(trace, var_names=[flagged_param_a, flagged_param_b], kind="scatter")
```

If what looks like two disconnected blobs in a 1-D marginal is actually one connected
banana/ridge in 2-D (e.g. `a·b ≈ constant`), that's Pattern 4/5 geometry from
`ess-rhat-diagnostic-guide.md` (a discontinuity or funnel), not real multimodality — a
reparameterization (work in `(a·b, a/b)` coordinates, or whatever the ridge's natural axis
is) may collapse it back to one mode entirely, the same way non-centering collapses a
funnel. This generalizes the pairwise-correlation check AZ2's investigation already used
(all |r| < 0.12 there, which is *why* that case turned out to be a genuine discontinuity,
Pattern 4, rather than a collapsible ridge).

Both checks together cost under an hour and can end the investigation outright. Only
proceed to Step 2 if the multimodality survives both.

<a id="step-2-map-modes"></a>
### Step 2 — Map the modes systematically

Three tools, cheapest first, escalate only as needed:

**2a. Multi-start optimization / multi-path Pathfinder (cheapest, first pass).**
`pm.fit(method="pathfinder", num_paths=8)` runs several independent quasi-Newton
optimization paths and is fast (seconds-minutes) — different paths from different starting
points can and do land near different modes, so this is a cheap way to survey *candidate*
mode locations before committing to full MCMC. **Caveat, stated directly in the
`pymc-modeling` skill**: "Approximate methods underestimate posterior uncertainty and may
miss multimodality. Always validate with MCMC when possible" — treat Pathfinder/ADVI output
here purely as a location survey, never as evidence of relative mass (mean-field ADVI in
particular assumes independence and will actively hide multimodality even when it exists;
full-rank ADVI still fits one unimodal Gaussian per run). `pmx.fit_laplace(model)` similarly
finds exactly one mode per call (per-pmc-extras skill: "Multimodal posterior: No — Laplace
finds only one mode") — useful for cheaply refining a candidate location found by
Pathfinder into a precise MAP + Hessian, not for discovery on its own.

**2b. Dispersed-chain launching (this codebase's existing default, keep using it, but
know its limit).** Already standard practice here (AZ1a/AZ1b's 4→8→16 chain escalations) —
launch many NUTS/nutpie chains from independently jittered inits and cluster their
post-warmup means (or, for a simplex/categorical quantity, dominant category) to find
distinct basins. `hierarchical_mode_summary` (`diagnostics.py`) already automates the
clustering step for a per-group simplex variable: chain purity ≥95% assigns a chain to a
mode, and it reports `n_modes`/`mode_chain_counts` directly. For a continuous (non-simplex)
multimodal quantity, the same idea generalizes via `sklearn.cluster.DBSCAN` (not k-means —
you don't know `k` in advance, and DBSCAN doesn't force a fixed cluster count the way
k-means does) on the vector of per-chain posterior means for the flagged parameters.

**Known limitation, already demonstrated in this codebase and worth restating explicitly
because it undermines a number people have already quoted**: dispersed-chain-fraction
counting is *not* a valid estimator of relative posterior mass unless chain initialization
is itself an unbiased draw from something proportional to the true posterior — which
`jitter+adapt_diag` around a fixed default point is not. AZ1a's own history is the direct
evidence: the 4-chain run reported an apparent 50/50 split; the 8-chain rerun (same model,
same data) found the true split was 37.5%/62.5% — a >10-point correction from doubling the
chain count alone, with no reason to think 8 chains has fully converged to the true ratio
either. **Treat any chain-fraction "mass estimate" as a rough, un-validated screen**, always
quoted with the `n_chains` it came from (per `hierarchical_mode_summary`'s own docstring),
never as the final number in a report without corroboration from §2.4/§3.

**2c. Grid search on suspected dimensions — narrow, situational, not a general tool.**
Only tractable when you have a *strong, specific hypothesis* about which 1-2 low-dimensional
quantities carry the multimodality (this codebase's own cases are exactly this shape: a
3-category simplex, a binary lag choice — not a general high-dimensional search). For those,
evaluate `model.compile_logp()` on a fine grid over the suspected 1-2 dimensions with
everything else held at each mode's own converged mean, and plot the resulting profile
log-density — this is a direct, cheap (no sampling at all) visual confirmation of two
separated peaks vs. one broad one, and costs a few seconds once the vectorized/batched logp
machinery from the moment-matching work (`model-progression-notes.md`'s "Moment matching for
pm.Potential-based models" recipe: `replace_rvs_by_values` + `vectorize_graph`) is reused.
**Do not attempt this beyond 2-3 dimensions** — it is a curse-of-dimensionality method, not
a scalable one; for anything higher-dimensional, rely on 2a/2b instead.

<a id="step-3-per-mode-sampling"></a>
### Step 3 — Sample and diagnose each mode individually

**First, check whether the modes are even trapped at all — this changes everything below.**
Before doing any of the resampling/subsetting work in this step, count within-chain switches
(how many times the dominant mode/category changes across a single chain's own draws). A
genuinely bimodal marginal (confirmed via Step 1's KDE peak check) can show up in two very
different ways: chains that get permanently stuck in one mode from early on (~0 switches —
the "hard multimodality" this whole document has mostly assumed), or chains that
successfully cross between both modes many times over their own run (tens+ switches, no
chain ever trapped). **Found concretely in AZ1d's own lag-category posterior**: in the exact
same model and run, one area's 8 chains showed zero switches each (hard-trapped) while two
other areas' 8 chains showed 26-108 switches each (round-tripping) — the same underlying
mechanism (a near-discrete softmax construction) producing both signatures depending on the
area's specific data. If chains are round-tripping, **the ordinary pooled posterior mean is
already a legitimate direct mass estimate** (check its ESS — 100+ is a reasonable bar; it
won't be as sharp as a hard-multimodality case's chain-count split, but it doesn't need one
either) — skip straight to a lighter version of Step 6, no per-mode resampling, no Step 4
machinery required. Only proceed with the rest of this step for genuinely trapped modes.

Once mode locations are known (Step 2) and confirmed trapped (not round-tripping), get a
trustworthy *within-mode* posterior for each:

**Cheapest: subset already-collected draws by mode**, valid *only* if each contributing
chain committed purely (≥95% purity, same threshold as `hierarchical_mode_summary`) to one
mode for the whole run — verify this explicitly per chain before subsetting, don't assume
it from a chain-mean scatter plot alone. This is free (no new sampling) but only works when
Step 2's dispersed chains already happened to separate cleanly.

**If chains don't separate cleanly, or you need more within-mode draws**: constrain
sampling to one mode directly. Two options, different strength guarantees:

**Option A — seed and hope (a soft nudge, already validated in this codebase)**: seed
**every** chain's `init_mean` toward the target mode's category (nutpie passthrough, already
wired via `DwellingModel.sample()`'s `_NUTPIE_PASSTHROUGH_KEYS`), then sample normally and
verify post-hoc that no chain wandered to the other mode. Cheap, no model change at all — but
it's only a starting-point nudge, not a guarantee: AZ1d's own seeding experiment left some
areas still split even after seeding (the genuinely-tied ones, where there's no "right" mode
to seed toward), and it does nothing for a round-tripping case (Pattern 2c) where chains
actively want to cross back regardless of where they started.

**Option B — boundary confinement (a hard guarantee, new to this guide)**: impose an actual
boundary in parameter space that the sampler cannot cross, so the run samples the *exact*
posterior conditional on being in that mode's region — zero cross-contamination, not just a
low probability of it. This is a real, named technique (truncated/constrained-region MCMC;
the more sophisticated version in the general HMC literature uses an actual momentum
*reflection* off the boundary within a leapfrog trajectory rather than a soft penalty — see
Neal 2003 §5.3 and Pakman & Paninski 2014, "Exact Hamiltonian Monte Carlo for Truncated
Multivariate Gaussians" — genuine reflection isn't natively available in PyMC/NUTS, so the
practical PyMC-native equivalent below approximates it via a steep potential wall, which NUTS's
gradient-based dynamics turn away from before crossing, rather than an exact geometric bounce).

Two ways to implement the boundary in PyMC, in order of preference:
1. **A genuinely truncated/bounded distribution** (`pm.Truncated`, or a custom bounded
   transform) when the mode-separating variable is a single scalar with a clean threshold —
   this is *exact*: the excluded region has literally zero density, not just a steep penalty,
   so there's no tuning and no divergence risk from an overly sharp wall.
2. **A steep `pm.Potential` penalty** when the boundary is a composite condition across
   several parameters (e.g., "which side of the softmax argmax is this raw_offset vector on"
   — not expressible as a simple bound on one variable). This is only a *soft* wall — finite,
   not infinite, so pick a penalty steep enough that crossing is astronomically unlikely (check
   post-hoc that no draw actually crossed) but not so steep that it creates a divergence-prone
   near-discontinuity; a scaled `pt.switch` with a penalty a few hundred nats deep is usually
   enough headroom without being numerically pathological. Worked recipe: §4.4.

**Critical validity requirement for either option**: the boundary must sit in the low-density
*valley* between the modes' peaks, not cut into either mode's own mass — otherwise you're not
isolating a mode, you're truncating it, and its mean/variance will be biased. Locate the valley
from Step 1's KDE peak-detection (the density minimum between two detected peaks) or Step 2's
cluster means (roughly their midpoint, refined against the actual density if the modes aren't
symmetric) — don't just guess a round number.

**This is the same mechanism as Pattern 3's failed hard cap on tau (`ess-rhat-diagnostic-
guide.md`) — but a completely different purpose, and that distinction matters.** Pattern 3's
cap was a *permanent model change* meant to stop an area from diverging into a second mode at
all, fighting genuine data-driven ambiguity — it failed because the tension didn't disappear,
it relocated to the population-level hyperparameters. Boundary confinement here is a
*temporary, diagnostic-only scaffold* applied to one sampling run, for an area/mode already
confirmed to exist (Step 2), for the sole purpose of getting that mode's own clean statistics
to report — it never becomes part of the model itself, and it isn't trying to suppress the
other mode's existence, only to sample this one without contamination while reporting. Using
this technique as a permanent model constraint instead of a one-off diagnostic run would
reproduce Pattern 3's exact failure — keep the confinement scoped to the specific sampling
call that produces the mode card's numbers, discard it afterward.

**Also useful for Pattern 2c (round-tripping) cases**, not just hard-trapped ones: the pooled
mean already gives a valid estimate of the *mixture's* mean, but if you want each mode's own
mean/variance separately (not just the blended pooled statistic), confinement is the way to
get it — round-tripping chains cross the valley freely, so subsetting by dominant category
(the free option used for hard-trapped modes) doesn't cleanly separate them the way it does
when chains never cross.

**Budget**: each mode needs its own ≥4 (ideally the codebase's usual 8) *confirmed-pure*
chains for a trustworthy within-mode r-hat/ESS read — so K modes costs roughly K× a single
normal run's chain budget, not 1×. For 2-3 modes this is usually affordable; for a
model with many small ambiguous groups each with their own 2-3 modes (like AZ1d's 13-area
core, if you tried to resolve every area's ambiguity individually rather than reporting it
in aggregate), this stops being practical — a reason in itself to prefer Step 6's aggregate
reporting over exhaustive per-area mode resolution for large numbers of small, low-stakes
ambiguities.

**Diagnose each mode's own subset exactly as normal**: r-hat/ESS restricted to that mode's
chains only, `check_chain_agreement`, autocorrelation. A mode that itself shows bad r-hat
or slow-decaying autocorrelation has its *own* unresolved problem (funnel, more sub-modes)
— run it back through `ess-rhat-diagnostic-guide.md`'s §2 procedure before trusting anything
built on top of it (mass estimate, PPC, report).

<a id="step-4-mass-estimation"></a>
### Step 4 — Estimate relative posterior mass across modes

Full method catalog with cost/reliability tradeoffs is §3 — this is the short version:

1. **Naive chain-fraction** (§2b) — free, but demonstrated unreliable at low chain counts
   in this codebase's own history. Use only as a first screen, always re-validated by one
   of the methods below before it goes in a report.
2. **Laplace/BIC-style screen at each mode's MAP** — cheap (one optimization + Hessian per
   mode, via `pmx.fit_laplace` run separately at each mode's own converged region), gives a
   fast closed-form relative-mass ratio. Valid only if each mode looks roughly Gaussian
   (check via Step 3's diagnostics first) — use as a sanity check on the more expensive
   methods below, not a final number, since it silently fails for boundary-pinned
   parameters (e.g. AZ1c's tau pinned at its cap) or skewed within-mode shapes.
3. **Bridge sampling between modes**, reusing this codebase's own already-built and
   verified unconstrained-transform/logp infrastructure — the recommended default when you
   already have clean per-mode draws from Step 3 and don't want to resample everything.
   Worked recipe: §4.1.
4. **Stacking across per-mode "sub-posteriors"** (Yao, Vehtari & Gelman 2022) — reuses
   `sensitivity.py`'s existing `compute_z_ensemble`/`compute_decomposed_uncertainty`
   machinery almost unchanged, and answers a genuinely different, often more useful
   question than a mass ratio (see §3.4). Cheap once per-mode traces exist.
5. **PyMC's native `pm.sample_smc`** — no new dependency, handles multimodality by
   design, gives mass fractions *and* a log-marginal-likelihood estimate in one run without
   needing Step 2/3's mode-mapping first. But has a documented, quantified bias toward the
   dominant mode as dimensionality grows (§3.3) — check this codebase's actual parameter
   count before trusting it uncorroborated.
6. **Thermodynamic integration / stepping-stone, parallel tempering, nested sampling /
   pocoMC** — high engineering and/or compute cost; reserve for when 3-5 disagree with each
   other, or when this becomes a persistent domain-wide problem (§7).

<a id="step-5-validate"></a>
### Step 5 — Validate the mode structure isn't still an artifact

(Beyond Step 1's cheap symmetry/ridge checks — these require Step 3's per-mode draws, so
they come after, not before, mode-mapping.)

**5a. Prior sensitivity.** Refit with 2-3 alternative, individually defensible priors
(2-3x wider/narrower scale on the relevant parameter, or a different family) and check
whether mode *count* and *locations* are stable, not just whether convergence improves —
this reuses the exact methodology `ess-rhat-diagnostic-guide.md` Pattern 3 already applies
for a different purpose (checking whether the *posterior* moved, not just the *prior*).
If a mode appears or vanishes under a mild, reasonable prior change, that mode is a
prior artifact, not a data-driven finding, and shouldn't be reported as one.

**5b. Posterior predictive check per mode.** Generate posterior-predictive draws from
each mode's own subset and compare specifically in the region where the modes differ (the
specific area/years a lag-category split disagrees on, say) — reuses this codebase's
existing `resp_noise`-colored plotting infrastructure and `plot_z_area_modes`/
`plot_spike_tracking_examples` machinery, which already renders per-mode bands. A PPC
discrepancy between modes is stronger, more interpretable evidence than the
log-likelihood-gap check alone, and is the *only* option when the likelihood is
`pm.Potential`-based (no automatic per-draw `log_likelihood` group) rather than a plain
observed RV.

**5c. Substantive plausibility.** For each candidate mode, can you state in one sentence a
real-world story it corresponds to ("BEN registration typically lags planning completion by
~12 months" / "registration is same-year for developments under X units")? A mode with no
sensible domain story, even if statistically clean, is a flag to look harder for an
artifact the other checks missed — not proof by itself, but informative in the same way
this codebase's E01002703 correction was ("uncorroborated single-source spike" is a
plausible story; a mode with no such story deserves more scrutiny before being reported
as real).

**5d. Held-out / robustness check (optional, most expensive, lowest priority).** Refit on
a perturbed dataset (drop the most recent year, subsample areas) and check the same mode
split reproduces. Full-joint refits are expensive for a 200-area hierarchical model — worth
doing only for a mode split that will drive a real downstream decision, not routinely.

<a id="step-6-report"></a>
### Step 6 — Report

See §6 for the full "mode card" template. The single most important lesson already on
record in this codebase for *why* this step matters, worth repeating verbatim: AZ3's
E01002702 investigation found the posterior **mean** for a genuinely bimodal `z` cell
(a spike at ~0 plus a broad hump at 30-150) sat in the **low-density valley between the two
modes, representing neither explanation** — and that a user-proposed alternative
configuration had *higher* log-posterior-density than the reported mean. **Never report a
bare posterior mean for a confirmed-multimodal quantity** — report per-mode summaries with
mass weights, or at minimum flag the cell/parameter as multimodal wherever its mean would
otherwise be quoted.

---

## §3 — Mass-estimation method catalog

Ordered roughly by cost; each entry states what it needs, what it gives you, and its
sharpest failure mode.

<a id="31-naive-chain-fraction"></a>
### 3.1 Naive chain-fraction (§2b) — screen only

**Needs**: nothing beyond Step 2/3's dispersed chains. **Gives**: a rough mass ratio.
**Failure mode**: biased by initialization, not a valid sampler of the true posterior's
mode-crossing frequency (chains never cross between modes by construction once trapped) —
demonstrated unreliable in this codebase (AZ1a: 50/50 at 4 chains → 37.5/62.5 at 8 chains).
Never the final number.

<a id="32-laplace-bic"></a>
### 3.2 Laplace/BIC screen per mode — cheap sanity check

**Needs**: `pmx.fit_laplace(model)` run once per mode (each call finds one MAP + Hessian).
Relative mass ≈ Laplace approximation to each mode's log marginal likelihood:
`log Ẑ_k ≈ log p(θ*_k | data) + (d/2)·log(2π) − ½·log|H_k|` (a BIC-flavored quantity, `d`
= number of free parameters, `H_k` the Hessian at the mode). **Gives**: a closed-form ratio
in seconds, no MCMC beyond what Step 3 already produced. **Failure mode**: only valid when
each mode is itself roughly Gaussian (verify via Step 3's own diagnostics — this is exactly
bridge sampling's own assumption below, so a failure here predicts a failure there too) and
breaks for a boundary-pinned parameter (Pattern 5b/AZ1c's tau-at-cap geometry) or a skewed
within-mode shape. Use to sanity-check 3.3/3.4/3.5 against each other, not standalone.

**A second, sharper failure mode, found by actually hitting it rather than just suspecting
it**: fitting the local Gaussian/volume term over TOO FEW dimensions relative to where the
mode's true height difference actually lives will badly overstate confidence, not just add
noise. Concretely (AZ1d's `E01002702`): a 2D local Laplace fit on `raw_offset` alone (the
variable mechanically closest to the flagged mode split) gave a sane-looking 38%/62% split
using only that variable's own directly-attached likelihood term. Extending the SAME 2D fit to
also credit a second likelihood term whose ~7-nat preference actually operates through a
*different*, unfitted 9-dimensional latent (`z`'s own zero-sum-constrained spread, not
`raw_offset`) produced a nonsensical 99.9%/0.1% split — the full height difference got
attributed to only 2 dimensions' worth of volume. **Before trusting a local Laplace/BIC number,
check that every likelihood term contributing to the height difference you're measuring
actually varies through the SAME dimensions you fit the Gaussian to** — if a meaningful part of
the height difference operates through a latent variable you held fixed at a point estimate,
either expand the fitted dimensions to include it (real engineering cost — see 3.3's transform
handling) or don't include that likelihood term's contribution at all and report the narrower
result as a documented lower/upper bound, not the full answer.

<a id="33-bridge-sampling"></a>
### 3.3 Bridge sampling — recommended default when you already have per-mode draws

**Needs**: posterior draws from each mode (Step 3) *and* the ability to evaluate the
model's unnormalized log-density (`model.compile_logp()`) at arbitrary points — this
codebase already built and verified exactly this (the moment-matching recipe in
`model-progression-notes.md`: `replace_rvs_by_values` + `vectorize_graph`, verified against
`model.compile_logp()` to ~1e-11 max abs error, ~1.7s per 6000-draw batch). **Key
simplification worth knowing before reaching for something heavier**: because all modes
live in the *same* model, same prior, same data — unlike a general Bayes-factor problem
between different models — you only need the *ratio* `Z_A/Z_B` of two modes' local
normalizing constants, not either one's absolute value. The classic Meng & Wong (1996)
iterative bridge estimator, using a fitted multivariate-normal-or-Student-t importance
density `g_k` per mode (fit to that mode's own unconstrained-space draws), computes exactly
this ratio directly — see the Gronau et al. (2017) tutorial ("A Tutorial on Bridge
Sampling", *Journal of Mathematical Psychology*) for the full iterative formula and its
`bridgesampling` R package as a reference implementation (no mature Python equivalent — the
`model-evaluation` skill's own read is blunt: "Bayes factors are difficult to compute
reliably... not recommended for routine model comparison — prefer LOO" — true for *general*
Bayes factors between different models, but the relative-mass-within-one-model case here is
a narrower, better-behaved problem, since the importance density only has to cover one
mode's local geometry, not an entire alternative model). **Gives**: a ratio with an
estimable standard error (via the tutorial's `error_measures`-style bootstrap over the
importance draws). **Failure mode**: variance blows up if a mode's own shape is strongly
non-elliptical (skewed, multi-lobed within itself) — Step 3's diagnostics should already
have flagged that as its own unresolved problem before you get here. Worked recipe: §4.1.

<a id="34-stacking"></a>
### 3.4 Stacking across per-mode sub-posteriors — different question, often the more useful one

Yao, Vehtari & Gelman, "Stacking for Non-mixing Bayesian Computations: The Curse and
Blessing of Multimodal Posteriors" (*JMLR* 23, 2022) proposes exactly this situation's
canonical fix: run separate chains/mode-finders to hit each mode, then combine the
resulting sub-posteriors via **stacking** (Yao, Vehtari, Simpson & Gelman, *Bayesian
Analysis* 13(3), 2018) rather than by estimating each mode's true probability mass. The
distinction matters: stacking weights are chosen to **optimize predictive accuracy** (an
LOO-based scoring rule), not to estimate the true relative posterior mass — a genuinely
different, and for most downstream uses (predicting `z`, reporting a point estimate) more
directly useful, quantity than a mass ratio. **Needs**: per-mode LOO/ELPD via
`az.compare`/`azs.compare`, exactly the model-evaluation skill's standard workflow, just
applied to per-mode traces instead of per-model traces. **Practical win specific to this
codebase**: `sensitivity.py`'s `compute_z_ensemble`/`compute_decomposed_uncertainty`
already implement LOO-stacking-weighted ensembling and within/between-uncertainty
decomposition — built for combining *models*, but the interface (`traces: dict[str,
az.InferenceData]`, an optional `comparison_df` of stacking weights) is architecturally
identical to combining *modes*: hand it `{"mode_a": trace_a, "mode_b": trace_b}` instead of
`{"AZ1b": trace, "AZ3": trace}` and the ensemble-z / within-vs-between-uncertainty
decomposition falls out with no new code. **Caveat, worth being explicit about**: whole-
model LOO comparison is currently *deferred* in this codebase's active work
(`az-family-work-plan.md`'s "Deferred" section) because of unresolved concerns about
PSIS-LOO reliability across structurally different models (AZ0b's own moment-matching
failure). Comparing *modes of the same model* is a narrower, better-behaved case of the
same machinery (same likelihood structure, same prior, same `log_likelihood` group shape)
— worth turning the LOO plumbing on for this specific, narrower purpose even while
whole-model comparison stays deferred more broadly.

<a id="35-smc"></a>
### 3.5 SMC (`pm.sample_smc`) — native, no new dependency, but documented dimensionality bias

PyMC ships `pm.sample_smc()`: tempers from prior (β=0) to posterior (β=1) over stages,
resampling and mutating a particle population at each stage — designed specifically to
handle multimodal posteriors (particles can independently discover different modes; the
resampling step keeps each mode's particle count roughly proportional to its true mass,
*if* the sampler is working correctly) and returns a log-marginal-likelihood estimate
alongside the posterior. **No mode-mapping (Step 2) required first** — run it directly on
the full model.

**Documented failure mode, checked directly against PyMC's own example gallery, not
assumed**: the official `SMC2_gaussians` notebook (bimodal Gaussian mixture, known true
minority-mode weight = 0.1) reports the *recovered* weight as **0.907** at 4 dimensions —
already badly biased toward the dominant mode — and **0.991** at 80 dimensions, i.e. nearly
all the minority mode's mass gets lost as dimensionality grows. The notebook's own stated
mitigation is to increase `draws`, tune `p_acc_rate`, and monitor `n_steps` — real levers,
but this codebase's models have per-cell latents (`z`: 200 areas × 10 years, plus per-area
lag simplices) numbering in the thousands, several orders of magnitude past the 80-dim case
PyMC's own example already shows failing. **Practical implication**: don't run
`pm.sample_smc` on the full joint model and trust its particle fractions uncorroborated —
either (a) use it only for a **reduced representation** restricted to the specific
low-dimensional quantity that's actually multimodal (the simplex/category, holding
everything else fixed at a representative posterior draw — the same reduction Step 2c's
grid search uses), where SMC's own literature-documented strengths actually apply, or
(b) treat its output as one more corroborating estimate alongside 3.2-3.4, not a standalone
answer. **Cost**: comparable to running several times as many mutation steps as a single
NUTS run — moderate, cheaper than 3.6 below.

<a id="36-thermodynamic-integration"></a>
### 3.6 Thermodynamic integration / stepping-stone sampling — expensive, absolute evidence

Lartillot & Philippe (2006, thermodynamic integration) and Xie et al. (2011, "stepping
stone" — a lower-variance telescoping-importance-ratio improvement on TI) construct a full
temperature ladder (`β: 0 → 1`, typically 10-30 rungs) and run converged MCMC at *each*
rung, then integrate/telescope to an absolute log-evidence. To get a *relative* mass
between two modes specifically (rather than the whole model's evidence), you'd need to
restrict each rung's sampling to one mode's basin (a hard boundary, feasible for a discrete
category like a lag choice, messier for a continuous multimodal geometry) and run the whole
ladder separately per mode. **Cost**: the most expensive option on this list by a wide
margin — effectively 10-30x a single converged sampling run, *per mode*. **When it's worth
it**: only if 3.3's bridge-sampling importance-density assumption is clearly violated (a
strongly non-elliptical per-mode shape) or an independently-wanted *absolute* evidence
number is needed for something beyond relative mass (e.g. feeding into a genuine
cross-model Bayes factor later). Not recommended as a first move for this codebase's
sampling costs (models already take minutes; 10-30x that per mode is hours, easily a full
day across several modes).

<a id="37-parallel-tempering"></a>
### 3.7 Parallel/simulated tempering — theoretically the cleanest, practically not recommended here

Geyer (1991) / Earl & Deem (2005, "Parallel tempering: theory, applications, and new
perspectives") run multiple chains at a temperature ladder simultaneously with periodic
Metropolis-Hastings swap proposals between adjacent rungs — at high temperature the
posterior flattens toward the prior and chains can cross energy barriers, and (unlike naive
dispersed-chain counting, §3.1) a well-tuned PT chain's β=1 draws genuinely sample the true
joint posterior with correct relative mode weighting, checkable via a "round-trip" count
(how many times a chain fully traverses low→high→low temperature). **Not natively available
in PyMC** — no general parallel-tempering NUTS implementation ships with it; would need an
external package (`ptemcee`, or BlackJAX's tempered-SMC in the JAX ecosystem) and real
engineering to wire a PyMC model's logp into it, plus non-trivial tuning (adjacent-rung swap
acceptance ~20-40%, a ladder fine enough to bridge each gap). **Recommendation**: prefer
3.5 (SMC, already native) or 3.8 (pocoMC, purpose-built) over hand-rolling PT for this
codebase — PT's theoretical cleanliness doesn't offset the engineering cost when off-the-
shelf alternatives already exist that target the same problem.

<a id="38-nested-sampling"></a>
### 3.8 Nested sampling / pocoMC — the escalation path if 3.5 isn't good enough

Nested sampling (Skilling 2004/2006) samples via a sequence of constrained-likelihood
shells rather than MCMC transitions, and is specifically well-regarded in the literature
(astrophysics/cosmology, where it originated) for robustness to multimodality — dynesty
(Speagle 2020, *MNRAS* 493) and UltraNest use multi-ellipsoidal or "reactive" bounding
methods explicitly designed to track multiple modes separately, and both give a principled
evidence uncertainty as a byproduct. **Engineering cost, honestly stated**: no PyMC
integration ships out of the box — needs the model's logp *and* an invertible prior
transform (inverse CDF) exposed as plain Python callables, real work similar in kind to
this codebase's already-solved moment-matching unconstrained-transform problem
(`LogTransform`, `ZeroSumTransform`/`extend_axis_rev`, etc.) but for the *prior*, not just
the posterior density. **pocoMC** (Karamanis, Nabergoj, Seljak & Handley, 2022, arXiv:
2207.05660) is a newer, purpose-built alternative worth flagging specifically: it combines
sequential Monte Carlo with a normalizing-flow preconditioner, is explicitly benchmarked
against both vanilla MCMC and nested sampling on multimodal *and* high-dimensional
problems (reported 25-50x faster than nested sampling on comparable astrophysics
benchmarks), gives evidence estimates, and — like dynesty/UltraNest — takes plain
logp/logprior callables rather than requiring a PyMC-native integration. **Recommendation**:
don't reach for this on a one-off diagnosis. It's the right tool if §7's "persistent
domain problem" trigger fires and 3.5's SMC has already been tried and shown the same
dimensionality bias its own documentation warns about.

---

## §4 — Worked recipes

<a id="41-bridge-sampling-recipe"></a>
### 4.1 Bridge sampling between two modes (reusing existing infrastructure)

```python
import numpy as np
from scipy import stats

# 1. Per-mode unconstrained draws (reuse the moment-matching recipe's
#    forward-transform step from model-progression-notes.md)
draws_a_unconstrained = to_unconstrained(mode_a_trace)   # (n_draws_a, n_dim)
draws_b_unconstrained = to_unconstrained(mode_b_trace)

# 2. Fit an importance density to each mode (multivariate normal is the
#    simplest starting point; use Student-t if Step 3 diagnostics show
#    heavier-than-Gaussian tails within the mode)
g_a = stats.multivariate_normal(draws_a_unconstrained.mean(0),
                                 np.cov(draws_a_unconstrained.T))
g_b = stats.multivariate_normal(draws_b_unconstrained.mean(0),
                                 np.cov(draws_b_unconstrained.T))

logp_fn = model.compile_logp()  # unnormalized log posterior density, unconstrained space

# 3. Meng & Wong (1996) iterative bridge estimator for log(Z_a / Z_b):
#    draws from BOTH the posterior samples and BOTH importance densities
#    are needed; iterate to convergence (see Gronau et al. 2017 §3 for
#    the full update equation and stopping rule) rather than a single
#    plug-in ratio, which has much higher variance.
log_ratio, se = bridge_sampling_ratio(
    draws_a_unconstrained, draws_b_unconstrained,
    g_a, g_b, logp_fn,
)
mass_a = 1 / (1 + np.exp(-log_ratio))   # since mass_a / mass_b = exp(log_ratio)
```

Report `se` alongside `mass_a` — a wide SE (importance density not covering the target
posterior well) is itself informative and should downgrade confidence in the quoted split,
not be silently dropped.

<a id="42-cross-mode-ensembling"></a>
### 4.2 Cross-mode ensembling via existing `sensitivity.py` machinery

```python
from housing_projections.sensitivity import compute_z_ensemble, compute_decomposed_uncertainty
from housing_projections.analysis import compute_model_comparison

mode_traces = {"same_year": trace_mode_a, "one_year_lag": trace_mode_b}
# az.compare across modes of the SAME model — a narrower, better-behaved use of the LOO
# machinery than the currently-deferred whole-model comparison; needs log_likelihood
# populated per mode (pm.compute_log_likelihood, already done by DwellingModel.sample()).
comparison_df = compute_model_comparison(mode_traces)

z_ensemble = compute_z_ensemble(mode_traces, comparison_df)             # stacking-weighted z
uncertainty_df = compute_decomposed_uncertainty(mode_traces, comparison_df)
# uncertainty_df['z_between_uncertainty'] is now literally "how much does z change
# depending on which mode is true" -- the between-MODEL column, repurposed as
# between-MODE, with no code changes required.
```

<a id="43-symmetry-check"></a>
### 4.3 Symmetry check for label-switching (Step 1a)

```python
logp_fn = model.compile_logp()

def swap(point, param_name, i, j):
    p = dict(point)
    v = p[param_name].copy()
    v[..., [i, j]] = v[..., [j, i]]
    p[param_name] = v
    return p

point = {name: mode_a_draws[name][0] for name in model.free_RVs}  # one representative draw
swapped = swap(point, "flagged_param", i=0, j=1)
print(logp_fn(point), logp_fn(swapped))  # equal => literal relabeling, not real ambiguity
```

<a id="44-boundary-confined-sampling"></a>
### 4.4 Boundary-confined sampling for a clean per-mode read (Step 3, Option B)

**4.4a — exact truncation**, when the mode-separating variable is one scalar with a clean
threshold (found in Step 1/2 — the valley between the KDE-detected peaks, or the midpoint of
the two cluster means):

```python
import pymc as pm

valley = 0.5 * (mode_a_cluster_mean + mode_b_cluster_mean)  # refine against the actual KDE
                                                              # valley if the modes are skewed

with pm.Model(coords=...) as confined_model:
    # ... everything else built exactly as in the original model ...
    raw_offset = pm.Truncated(
        "raw_offset", pm.Normal.dist(mu=0, sigma=1),
        lower=valley,       # confines to mode B's side; use upper=valley for mode A
    )
    # ... rest of the model unchanged, same likelihoods ...
    idata_mode_b = pm.sample(chains=8, cores=8, random_seed=42)
# idata_mode_b['posterior'] is now an EXACT sample from the posterior conditional on
# raw_offset > valley -- report its mean/sd/HDI directly as mode B's own statistics.
```

**4.4b — steep potential wall**, when the boundary is a composite condition (e.g. "which
softmax category currently dominates") not expressible as a bound on one variable:

```python
import pytensor.tensor as pt

with pm.Model(coords=...) as confined_model:
    # ... build raw_offset, lambda_weights, etc. exactly as in the original model ...
    dominant = pt.argmax(lambda_weights[area_idx, :])
    pm.Potential(
        "stay_in_mode_b",
        pt.switch(pt.eq(dominant, target_category), 0.0, -300.0),
    )
    # ... rest of the model unchanged ...
    idata_mode_b = pm.sample(chains=8, cores=8, random_seed=42)

# Verify the wall actually held before trusting the result:
crossed = (idata_mode_b.posterior['lag_P_lambda_weights']
           .values[:, :, area_idx, :].argmax(axis=-1) != target_category)
assert not crossed.any(), "penalty too shallow -- some draws crossed the boundary"
```

-300 nats is deep enough that crossing is astronomically unlikely without being a literal
`-inf` (which can break gradient evaluation right at the boundary and cause divergences) —
tune the exact depth per-model if the assertion above ever fails, and always run the assertion
rather than assuming the wall held.

**What this buys over the pooled/subsetted draws already used in earlier steps**: an exact,
zero-contamination sample from that one mode's own conditional posterior — a materially
cleaner mean/HDI than either "hope the chains happened to separate" (Step 3's default) or "seed
and verify no wandering" (Option A above), at the cost of one extra full sampling run per mode
you want this treatment for. Reserve it for the specific mode(s) whose own statistics will
actually be quoted in the report (§6's mode card), not routinely for every flagged group.

---

<a id="5-quick-reference"></a>
## §5 — Quick-reference / cost table

| Method | New dependency? | Cost | What it gives you | Sharpest caveat |
|---|---|---|---|---|
| Naive chain-fraction | none | minutes | rough screen only | biased by init; AZ1a: 50/50→37.5/62.5 going 4→8 chains |
| Laplace/BIC per mode | `pmx.fit_laplace` (already in this codebase) | minutes | closed-form ratio | only valid if each mode is ~Gaussian; fails at boundary pins |
| Bridge sampling | none (reuse existing logp infra) | tens of min | ratio + SE | needs a good-fitting importance density per mode |
| Stacking (per-mode ensemble) | none (`sensitivity.py` already built) | minutes once LOO exists | predictive-optimal weights, not true mass | answers a different question than "true ratio" — often the more useful one |
| SMC (`pm.sample_smc`) | none (native PyMC) | ~3-10x one NUTS run | mass fractions + log-evidence | documented dominant-mode bias growing with dimensionality (PyMC's own example: 0.1→0.907 true→recovered at 4 dims) |
| Thermodynamic integration / stepping-stone | none (hand-rolled ladder) | ~10-30x per mode | absolute evidence, most rigorous | very expensive; only for elliptical-assumption failures or absolute-evidence needs |
| Parallel tempering | `ptemcee`/BlackJAX | high engineering + compute | correct joint-posterior mode weighting | no native PyMC support; prefer SMC/pocoMC instead |
| Nested sampling (dynesty/UltraNest) / pocoMC | new dependency | high engineering, moderate-high compute | evidence + robust multimodal sampling | needs custom logp/prior-transform wrapper; pocoMC is the more modern, higher-dimension-tolerant choice of the two |

---

<a id="6-mode-card"></a>
## §6 — Reporting template: the "mode card"

For every mode reported, include all of the following — omitting the mass-estimate method
or its uncertainty is the single most common way this kind of finding gets over-quoted
later:

```
Mode: <one-line label, e.g. "E lags planning completion by 1 year">
Location: posterior mean/median + 90% HDI for the differentiating parameter(s)
Mass: <e.g. "62% [bridge sampling, SE=0.03 log-units, n_draws=4000/mode]">
      -- NEVER a bare percentage; always the method + its own uncertainty
      -- if only a naive chain-fraction screen was run, say so explicitly and
         flag the number as unvalidated, per §3.1
Substantive story: one sentence a domain reader can evaluate independently
PPC support: does this mode's own posterior predictive fit the differentiating
             observations better/worse/about the same as the other mode(s)?
Caveats: prior sensitivity result (stable / shifted under alternate priors),
         any residual within-mode convergence issues from Step 3
```

**Downstream guidance to attach whenever a multimodal quantity feeds a report or plot**:
report per-mode summaries, not a blended mean — repeat the E01002702 lesson (§2 Step 6)
inline if the audience is likely to reach for "just give me one number" regardless.

---

<a id="7-persistent-problem"></a>
## §7 — Is this a persistent domain problem? Should the default tool change?

**Trigger checklist** — don't act on a single case. Consider a default-tool change only if
*most* of these are true across **independent** model families (not just repeated
symptoms of the same one architectural cause, e.g. AZ1a/AZ1b/AZ1d are one story about
lag-category ambiguity, not three):

- Confirmed genuine multimodality (survived §0's gate and Step 1's artifact checks) recurs
  in ≥2 structurally unrelated parts of the model space, not just the one already-diagnosed
  lag/simplex mechanism.
- The Pattern-2a "report honestly" treatment from `ess-rhat-diagnostic-guide.md` is being
  reached for routinely enough that per-model manual mode-characterization (this whole
  document, run by hand each time) is becoming the bottleneck in iteration speed.
- A stakeholder-facing deliverable specifically needs a trustworthy, quoted mass split (not
  just "this year's attribution is uncertain, total is fine") often enough that the
  naive-chain-fraction screen's known unreliability (§3.1) is no longer an acceptable
  answer.

**If triggered — practical migration path for this codebase specifically:**

1. **Don't switch the default sampler wholesale.** Most models here (AZ0a, AZ2b, the
   floored AZ3) converge cleanly under nutpie/NUTS and gain nothing from SMC's extra cost.
   Scope any change to the specific model(s)/family showing the recurring problem, the same
   way AZ1b/AZ1c already override `sample_kwargs` for chain count rather than changing
   `DEFAULT_SAMPLE_KWARGS` globally.
2. **`pm.sample_smc()` is the first thing to try, not a rewrite** — `DwellingModel.sample()`
   (`base.py`) would need a new branch alongside its existing `use_nutpie` flag (SMC doesn't
   take nutpie's `draws`/`tune`/`chains` semantics the same way — `draws` doubles as the
   particle count, and PyMC runs its own small number of "SMC chains" for diagnostics
   distinct from `chains` as normally understood). Re-verify
   `_attach_pointwise_log_likelihood` still works afterward — SMC's internal sampling
   mechanism differs enough from NUTS that the `pm.Potential`-based pointwise-log-likelihood
   attachment (already a source of one silent bug in this codebase's history, per
   `model-progression-notes.md`'s DataTree note) deserves a fresh regression check, not an
   assumption it still works.
3. **Respect §3.5's dimensionality caveat before trusting it.** This codebase's models have
   thousands of latent per-cell values (`z`: 200×10, plus lag simplices) — well past the
   80-dimension case PyMC's own example gallery already shows badly biased. Prototype SMC
   first on a **reduced representation**: fix or marginalize out the well-behaved majority
   of `z`/areas and run SMC only on the specific ambiguous low-dimensional quantity (the
   simplex/category itself). This is the same reduction M16's marginalization work
   (`model-progression-notes.md` §3) already had to solve for a related reason — reuse that
   experience (the `CustomDist` + `pymc_extras.marginalize()` obstacles documented there:
   `pm.Potential` isn't traceable, `z` can't stay a `Deterministic`, `area` must be a batch
   dimension) as the template for building the reduced model, rather than re-deriving it.
4. **If SMC's reduced-representation version still isn't good enough**, pocoMC (§3.8) is
   the next step up — a new dependency, but purpose-built for exactly "multimodal AND
   higher-dimensional than nested sampling tolerates well," and it accepts a plain
   logp/logprior callable interface similar in shape to what the moment-matching and (if
   built) nested-sampling wrappers already require. Don't skip straight to it without
   trying the reduced-SMC route first — it's more engineering cost for a problem the
   cheaper route might already solve.
5. **Budget accordingly** (§5's table) — this migration is worth scoping as its own
   iteration, not a same-session addition once the trigger fires.

---

<a id="8-when-to-stop"></a>
## §8 — When to stop

Mirrors `ess-rhat-diagnostic-guide.md` §6, extended for this document's deeper checks. Stop
and report (§6) once: Step 1's artifact checks come back clean (no symmetry, no collapsible
ridge), Step 5's prior-sensitivity and PPC checks support each mode as substantively real,
and at least one mass-estimation method beyond the naive chain-fraction screen (§3.2-3.5)
has been run. Keep going only if: two independent mass-estimation methods disagree
materially (bridge sampling and SMC give substantially different splits — investigate the
disagreement itself before quoting either), a mode's own within-mode diagnostics (Step 3)
are themselves unresolved, or §7's persistent-problem trigger has fired and a scoped
tool migration is the actual next step rather than another round of manual
characterization.
