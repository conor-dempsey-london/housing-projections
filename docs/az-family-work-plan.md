# AZ family iteration — active work plan

**Status doc for the current round of work.** Unlike `model-progression-notes.md`
(the retrospective doc, updated when an iteration *concludes*), this is a live
checklist — check back here at the start of every new step, and update it
immediately when a step's status changes. When this round of work concludes,
fold the durable findings back into `model-progression-notes.md` and this file
can be retired/archived.

**See also `docs/az-ess-diagnosis.md`**: a focused pass diagnosing and attempting a fix for
every AZ-family model with a documented ESS/convergence problem (AZ1a, AZ1b, AZ2) — separate
from AZ3's own ESS story below, which stays here. Produced `AZ1c` (tau-capped hierarchical
lag — rejected, made AZ1b's multimodality worse), `AZ2b` (smooth top-boost ramp — real fix for
`sigma_delta_top_boost`'s own ESS, 47→800+, but a later full whole-model scan found the same
underlying pathology had resurfaced on `sigma_plan`/`sigma_ben` instead, not resolved — see
`az-ess-diagnosis.md`'s AZ2b follow-up section; branch closed with `sigma_ben` fixed via
`target_accept=0.97` and `sigma_plan`'s marginal ESS accepted as a disclosed limitation, not
fixable by sampler tuning), `AZ1e` (unregularized horseshoe per-area tau — rejected,
`local_scale` exploded to 3.09M causing divergences), and `AZ1g` (regularized/slab-capped
horseshoe — the follow-through AZ1e's own docstring flagged but didn't build; a real,
mechanistically-confirmed fix for AZ1d's flagged-minority-contaminates-shared-tau leakage
on two independently-sampled 200-area datasets, though not a full fix for per-area hard
multimodality on the harder of the two samples tested — see that doc's AZ1g section for
detail before using it as AZ1d's default replacement). `AZ1h` (AZ1g moved onto the textbook-
canonical regularized-horseshoe recipe: a sampled slab and a sparsity-calibrated `global_tau`
prior instead of AZ1g's hand-fixed versions of both) — **rejected**, both changes: a
mis-transferred sparsity formula caused a prior-data conflict, and sampling the slab introduced
a new shared-slab/per-area ridge that made whole-model convergence far worse (max r-hat
1.02→1.96 in the isolated test) despite the named scalars looking fine in isolation. AZ1g's
simpler, fixed-slab design stands confirmed, not just unchallenged.

Baseline going into this round: **AZ0a** (zero-sum census-anchored z prior,
plain same-year StudentT likelihoods, no lag/reallocation) is the current
best-validated model — max r-hat 1.006, min ESS 3180, 0 divergences. AZ0b
(discrete same-year/prior-year mixture) was rejected: worse convergence, and
a full moment-matching PSIS-LOO check found its apparent LOO win over AZ0a
was an artefact (0/561 bad-k points fixed).

## Reference areas (plotting)

`plot_spike_tracking_examples` (Phase 0 tool) now always includes these
LSOAs as extra panels, on top of its auto-selected ones, so plots stay
directly comparable across models rather than each one auto-selecting a
different set. Mirrored in code as `REFERENCE_AREAS` in
`src/housing_projections/plots/core.py` — the two are not auto-synced,
update both when a new area gets singled out by name in a report.

| LSOA | Why it's tracked |
|---|---|
| E01033491 | under-tracked: huge P spike (762), tiny D (126) |
| E01001774 | under-tracked: extreme mismatch (P_sum=460 vs D=18) |
| E01033711 | AZ0a/AZ1a's worst-missed spike; AZ1b/AZ2's biggest win (D=634) |
| E01002703 | single-source spike, NOT cross-source agreement (corrected -- see Phase 3 notes: 2013 P=274, E=8 same year; E peaks separately at 162 in 2012 when P=0; D=501) |
| E01002794 | P/E disagree on spike year (2020 vs 2016) |
| E01033700 | high Pareto-k, mostly-quiet years (D=556) |
| E01035656 | under-tracked spike (498), D=412 |
| E01002702 | AZ1b mode-summary example: chains split on E's lag category |

Panel titles now show the actual LSOA21CD code, not a positional index.

## Ground rules for this round

- One specific addition at a time. Don't combine unvalidated pieces.
- After every new model is sampled, generate `plot_spike_tracking_examples`
  (Phase 0 tool) and show it before drawing conclusions — aggregate
  diagnostics (r-hat, elpd) can look fine while the model quietly smooths
  away exactly the spikes stakeholders care about (this is literally how
  AZ0a's failure modes were found).
- Stakeholder requirement, non-negotiable: the model must be able to
  attribute large P/E observations to specific years/areas, not just
  average them away. Every step should be checked against this, not just
  convergence stats.
- LOO/moment-matching/k-fold comparison is explicitly **deferred** (user's
  call) until there's a real finalist worth evaluating carefully. Use
  convergence diagnostics + the spike-tracking plot to judge each step in
  the meantime.

## Phase status

### Phase 0 — spike-tracking diagnostic tool — DONE
`select_spike_tracking_areas` / `plot_spike_tracking_examples` added to
`src/housing_projections/plots/core.py`, exported via `plots/__init__.py`.
Auto-selects: worst under-tracked spikes, worst LOO Pareto-k area (if
available), biggest P/E spike-year disagreement, a well-tracked spike for
contrast. Verified against AZ0a — reproduced the manually-found failure
cases (E01033491, E01033711, E01002703) plus new ones.

### Phase 1a — AZ1a: fully-pooled continuous lag convolution — DONE, mixed result
Built by reusing existing, already-validated machinery
(`_build_lag`/`_build_pre_inference`, the same builders M1 used) rather than
new code — both P and E get their own **separate** Dirichlet lag-weight
vector (`lambda_weights_P`, `lambda_weights_E`), each **fully pooled**
(one shared kernel across all 200 areas). `max_lag=2`, matching M13/M14's
empirically-validated window. No discrete latent anywhere (unlike AZ0b), so
no analogous component-identity instability expected.

**Result**: converged worse than AZ0a, not better — max r-hat 1.74 (8 params
over threshold), min ESS 6, year-allocation confidence collapsed to 89%
low-confidence (vs AZ0a's 4%). Root cause diagnosed directly from the trace,
not assumed: `lambda_weights_E` is genuinely **bimodal** across chains —

```
chain 0: [0.08, 0.88, 0.04]   <- "E lags true completion by 1 year"
chain 1: [0.89, 0.06, 0.05]   <- "E is same-year"
chain 2: [0.89, 0.06, 0.05]   <- same as chain 1
chain 3: [0.08, 0.87, 0.04]   <- same as chain 0
```

A single shared kernel has no one right answer when different areas'
true lag patterns genuinely differ — this reproduces a failure mode already
documented in this codebase's history for a *different* model (the old
M-family's M11: "shared lambda_weights... the real problem was the
architecture, not the parameterization"). Spike-tracking plot showed AZ1a
pulling some signal into previously-missed spikes (e.g. E01033711's 2013
BEN spike, completely missed by AZ0a, now partially tracked) but also
spreading small spurious bumps into years that shouldn't have any —
consistent with a single kernel being forced to compromise across
heterogeneous areas.

Plots saved: `results/scratch/az0a_spike_diagnostic.png`,
`results/scratch/az1a_spike_diagnostic.png`.

### Phase 1b — area-level (not borough) grouping for lag weights — DONE, clear improvement
**Revised from the original plan.** Borough-level partial pooling was the
original idea, but rejected before building it: no strong mechanistic
reason to expect a lag caused by site/development-level completion-to
-registration timing to align with administrative borough boundaries — the
user's call, and correct on reflection. "Delays are caused by things
operating below Borough level."

Two candidate designs, not mutually exclusive:

1. **Area-level hierarchical pooling** (favoured starting point) — 200
   groups (one per area), each area's lag-weight vector shrunk toward a
   shared global kernel via an estimated between-area variance `tau`.
   `tau -> 0` recovers AZ1a's fully-pooled answer; `tau` large lets an
   area's data pull it away from the global kernel. NOT the same as literal
   unpooled (avoids the identifiability risk flagged for M12-style
   independent-per-area fits) — the shrinkage prior itself regularizes it.
   Needs new builder code (`_build_lag`'s Dirichlet-only interface doesn't
   support this) — likely a softmax-over-logits construction:
   `area_logit[a,k] = mu_logit[k] + offset[a,k]`,
   `offset[a,k] = raw[a,k] * tau[k]` (non-centered),
   `lambda_weights[a,:] = softmax(concat([0, area_logit[a,:]]))`.
   Same mechanic already used in this codebase for per-area hierarchies
   (M9's `sigma_slab`), just applied to a simplex-valued quantity instead
   of a scale.
2. **Empirical-lag clustering** (documented fallback, not first choice) —
   group areas by their own observed cross-source lag pattern (reusing the
   original cross-source lag-matching analysis), pool within cluster.
   Real risk: same data used to define groups AND fit the hierarchical
   model within them risks circularity/overfitting-flavoured optimism.
   Worth trying only if (1) doesn't resolve the bimodality, and worth
   validating carefully (e.g. some kind of held-out check) rather than
   trusting convergence alone.

**Built**: `_build_hierarchical_lag` in `models.py` (softmax-over-logits,
non-centered `offset[a,k] = raw[a,k] * tau[k]`, `AZ1b` model class). Reused
`_build_lag`'s Dirichlet(lag_alpha) prior mean, translated into logit space
(`prior_logit = log(lag_alpha[1:]/lag_alpha[0])`), so the prior's "prefer
short lags" bias carried over rather than silently resetting to uniform.

**Result — clear improvement over AZ1a, close to AZ0a**:

| | max r-hat | min ESS | low year-confidence |
|---|---|---|---|
| AZ0a (baseline) | 1.006 | 3180 | 4.0% |
| AZ1a (fully pooled) | 1.744 | 6 | 89.0% |
| **AZ1b (area-hierarchical)** | **1.123** | **23** | **14.5%** |

Confirmed directly (not assumed) that hierarchical pooling fixed the
bimodality: `lag_P_mu_logit`/`lag_E_mu_logit` now agree closely across all
4 chains (vs AZ1a's clean 2-vs-2 split on `lambda_weights_E`), and both
`tau` posteriors are large (P: 2.6-3.7, E: 5.1-5.9) — areas are genuinely
using their individual freedom rather than the population splitting into
two camps.

Spike-tracking plot (`results/scratch/az1b_spike_diagnostic.png`) showed a
dramatic, concrete win: LSOA E01033711 (D=634) — the area AZ0a/AZ1a most
badly mis-tracked, missing its 2013 BEN spike (764) entirely — is now
tracked closely across three separate years (2014, 2016, 2018), correctly
attributing credit to whichever source peaked in each.

**Residual issue, localized and understood**: max r-hat is still above the
1.01 threshold, but isolated almost entirely to the per-area
`lag_P_raw_offset`/`lag_E_raw_offset` parameters (37/400 and 47/400 bad,
~10-12%) — the population-level `mu_logit`/`tau` are fine to borderline.

**Correction**: this was initially (wrongly) written off as an acceptable
residual issue. R-hat 1.12 and ESS 23 both fail the standard bar (r-hat <
1.01, ESS > 400) and should not have been waved off. Investigated properly:

- First hypothesis (non-centered parameterization mismatched to a large
  `tau`) turned out to be the wrong diagnosis. Checked directly: for the
  worst-offending areas (e.g. E01002702), **each chain spent all 1500
  draws entirely in one of two disconnected modes and never crossed over**
  (e.g. chain 1: 0/1500 draws in mode A, chains 0/2/3: 1500/1500). That is
  genuine hard multimodality, not slow mixing -- no amount of extra
  draws or reparameterization-for-geometry fixes that.
- Mechanism: with only ~10 observations/area/source, some areas' data
  can't cleanly distinguish between two candidate lag years each explaining
  a spike about equally well -- blending between them (a smooth simplex
  point) fits *worse* than committing fully to either one, so the
  likelihood surface itself has two separated peaks for those areas. This
  is the same underlying pathology AZ0b's discrete mixture had, just
  emerging locally, per-area, from a nominally continuous construction
  instead of model-wide from an explicit discrete latent.
- Tried tightening the shrinkage (`tau_sigma`: 1.5 -> 0.5) on the
  hypothesis that less per-area freedom would keep areas out of these
  disconnected modes. **Tested, not assumed -- and it made things worse**:
  max r-hat 1.12 -> 1.24, min ESS 23 -> 12. Checked why: the posterior
  `tau` barely moved (e.g. lag_E_tau ~2.7-3.2 under the tighter prior vs
  ~5.1-5.9 before) -- the likelihood's pull toward per-area divergence is
  strong enough to mostly override a 3x tighter prior, so the softer prior
  did nothing useful and merely made the geometry harder to sample.
  Reverted to `tau_sigma=1.5` (the better of the two tested values) --
  current saved trace is this reverted version.

**Status: open, unresolved.** Real options going forward, not yet chosen:
1. Much more aggressive/near-fixed shrinkage (e.g. tau_sigma << 0.5, or
   fixing tau near-constant) -- untested how small it would need to be to
   actually suppress the multimodality, and each step down trades away
   more of the per-area flexibility that made AZ1b's E01033711 win
   possible in the first place. Real risk of reverting most areas toward
   AZ1a's behaviour to fix ~10-15% of them.
2. Coarser grouping than per-area (e.g. D-band, similar to Phase 2's
   sigma_delta idea) -- fewer, larger groups might have enough aggregate
   data per group to avoid the sharp/peaked per-group solutions that cause
   this, without losing as much flexibility as full pooling.
3. Marginalize the near-discrete choice analytically instead of letting
   NUTS sample it -- the principled fix for this class of problem
   (precedented in this codebase: M16 marginalized an explicit discrete
   profile_k to fix M14/M15's Gibbs-sampling instability), but AZ1b's
   near-discreteness isn't from an explicit discrete RV, it emerges from
   the continuous softmax construction's likelihood shape for
   sharply-informative areas -- would need a different, more involved
   construction, not a direct reuse of M16's approach.
4. Accept and report genuine per-area ambiguity rather than forcing it
   away -- philosophically defensible (some areas' sparse data really
   can't distinguish two lag explanations) but practically unworkable with
   only 4 chains: r-hat/ESS become uninterpretable under real
   multimodality, and 4 chains split 3-vs-1 or 2-vs-2 is much too noisy an
   estimate of each mode's true relative posterior mass to trust.

**Chosen: option 4, with chains raised 4 -> 8 (cores=8) specifically to
make the reported mode split trustworthy** (`AZ1b.sample_kwargs`). Direct
result of raising chain count, not assumed:

- r-hat/ESS did NOT improve (as expected/correct under this approach --
  r-hat measures chain agreement, and well-mixing chains on a genuinely
  multimodal posterior *should* disagree; forcing r-hat->1 here would mean
  suppressing real ambiguity). max r-hat 1.12 -> 1.19, min ESS 23 -> 29,
  n_bad_rhat 8 -> 10 for lag_P (expected: more chains can surface more
  distinct modes that 4 chains never happened to visit).
- Built `hierarchical_mode_summary` (`diagnostics.py`) specifically to
  characterize this properly instead of relying on a single scalar
  max-r-hat number: for each flagged area, which lag category each chain
  concentrated on (>=95% of that chain's draws), grouped into modes, with
  an estimated relative posterior mass per mode from the fraction of
  chains landing there.
- With 8 chains, the picture is more nuanced than "everything is cleanly
  bimodal": roughly half of flagged areas (16/34 for lag_P, 9/31 for
  lag_E) are "stable" -- every chain cleanly commits to one mode, giving a
  trustworthy mass estimate (e.g. area 112/E01033711's E-lag: 6/8 chains
  in "same-year", 2/8 in "1yr-lag", i.e. ~75/25). The other half show
  chains that don't concentrate >=95% on any single category ("mixed") --
  a genuinely different, milder form of uncertainty (broad/diffuse rather
  than cleanly disconnected) that 4 chains couldn't distinguish from hard
  multimodality.
- Noted structural insight worth flagging for a future iteration: a single
  lag-weight vector per area assumes ALL of that area's years share one
  lag profile. Area 112/E01033711 -- the model's best spike-tracking win --
  has a genuinely split E-lag posture (~75/25 same-year vs 1yr-lag), which
  may reflect that its *different* spikes (2014, 2016, 2018) each have
  their own true lag, not one shared area-level lag. If so, no amount of
  per-area (as opposed to per-record) flexibility fully resolves this --
  worth keeping in mind if this class of ambiguity persists after Phase 2.
- Spike-tracking quality confirmed preserved with 8 chains (re-generated
  the diagnostic plot; E01033711's multi-spike tracking win is unchanged).

**Status: resolved for this round** under the chosen approach -- AZ1b's
remaining r-hat/ESS elevation is now understood, characterized, and
reported rather than either hidden or blocking progress. Treat AZ1b as:
total-per-area change (the z sum) fully reliable; year-by-year attribution
reliable for ~86% of areas (the non-flagged + "stable" flagged ones, whose
mode split can be quoted with a defensible n=8-chains-backed estimate);
genuinely uncertain (not just unconverged) for the remainder, use
`hierarchical_mode_summary` to report per-area specifics rather than a
single point estimate for those.

Plots: `results/scratch/az1b_spike_diagnostic.png` (4-chain, tau_sigma=1.5),
`results/scratch/az1b_8chain_spike_diagnostic.png` (final 8-chain version).

### Phase 2 — top-D-quartile sigma_delta boost — DONE, resolved via simplification
Merges the original Step 3 (heavier-tailed/adaptive deviation scale) and
Step 5 (partial pooling by D-magnitude band, user's explicit preference
over per-area) from the earlier next-steps discussion. Bin areas into a
handful of D-magnitude tiers (quartiles of |D|, n_bands=4), each with its
own `sigma_delta_band` pooled toward a population mean, replacing the
fixed `floor + k*|D|` formula. Independent, single-change branch off AZ0a
(not layered on the lag work), built as `AZ2`.

**Attempt 1 (dropped floor entirely, pure per-band multiplier) —
severe regression, caught before being treated as done.** Real-data run:
`frac_flat_despite_active` jumped from AZ0a's 11.5% to **70%** — the exact
core pathology this whole model family exists to avoid. Diagnosed directly:
`sigma_delta_band` collapsed to ~0.2-0.8 for the bottom 3 of 4 bands
(150/200 areas), flattening z almost everywhere. Root cause: bands are
defined by |D| (net census change), a poor proxy for volatility (a
canceling-out +100/-95 area has tiny net D but needs large sigma_delta) --
binning purely on |D| lets the hierarchy shrink a band's scale toward
whatever fits that band's typically-quiet majority, even when some
individual areas in it need much more room. Mirrors an already-documented
failure in this codebase (M9's per-area sigma_slab collapsing to ~0.16
across all 200 areas) -- same shrinkage-toward-the-majority pathology,
different parameter.

**Attempt 2 (reinstated a shared, fixed floor=3.0 as an unconditional
additive minimum, same value as AZ0a) — big improvement, not fully
resolved.** `frac_flat_despite_active` 70% -> 29% (still worse than AZ0a's
11.5%), max r-hat 1.56 -> 1.07, min ESS 7 -> 39. Investigated the
remaining gap rather than accept it: `sigma_delta_band`'s "excess over
floor" component is STILL collapsing to ~0 for bands 0-2 (only band 3, the
top quartile, retains real hierarchical flexibility, ~62 excess). Checked
directly against flatness by band:

| band (quartile of \|D\|) | frac_flat_despite_active |
|---|---|
| 0 (smallest) | 45.7% |
| 1 | 43.4% |
| 2 | 27.5% |
| 3 (largest) | 0.0% |

So band 3 -- which got real flexibility -- has ZERO flatness pathology,
even better than AZ0a's baseline. Bands 0-2 collapsed to an
*undifferentiated* floor-only fit (sigma_delta ~3.0-3.1 for all three,
barely distinguishable from each other) and lost the graduated `k*|D|`
scaling AZ0a's original formula gave those same areas -- a real design
mistake, not just a missing floor: `_build_zero_sum_z_prior_banded`
deliberately dropped the within-band linear |D| term ("band membership
already encodes magnitude tier"), which turns out false for the bottom
75% of areas -- there's real, useful |D|-dependence WITHIN those bands
that a single per-band constant doesn't capture, creating a threshold
discontinuity right at each band edge.

**User chose option 2** ("global formula + one extra top-tier boost")
over reinstating a per-band slope (option 1) or trying more/different
bands (option 3) -- go with the simplification the evidence already
pointed at, rather than add back complexity to patch a 4-way hierarchy
that had already shown 3 of its 4 groups added nothing.

**Attempt 3 (`_build_zero_sum_z_prior_top_boost`): keep AZ0a's original
floor + k*|D| formula COMPLETELY UNCHANGED for every area; add exactly
one new sampled scalar (`sigma_delta_top_boost ~ HalfNormal(40)`) that
only applies additively to the top quartile of |D| — RESOLVED, better
than AZ0a on the metric that matters most.**

| | max r-hat | min ESS | frac_flat_despite_active |
|---|---|---|---|
| AZ0a (baseline) | 1.006 | 3180 | 11.5% |
| AZ2 attempt 1 (pure 4-band, no floor) | 1.56 | 7 | 70.0% |
| AZ2 attempt 2 (4-band + floor) | 1.07 | 39 | 29.0% |
| **AZ2 attempt 3 (top-boost, final)** | **1.05** | **47** | **9.0%** |

`sigma_delta_top_boost` converged cleanly to a well-identified value
(mean 29.35, sd 2.10, range [22, 37] -- much tighter than the 4-band
version's top-tier "excess" of ~62 with no comparably clean uncertainty
estimate, since that number was fighting for identification against 3
other collapsing bands in the same hierarchy). Confirmed by direct test
(`test_only_top_quartile_gets_boost`) that non-top areas get EXACTLY
AZ0a's original formula (floor + k*|D|, unchanged), and top-quartile
areas get exactly that plus the sampled boost -- no silent behaviour
change for the 150 areas that were never the problem.

Spike-tracking plot (`results/scratch/az2_spike_diagnostic.png`) confirms
the practical win is real, not just a diagnostics-table number: E01033711
(D=634, AZ0a/AZ1a's worst-missed case) and E01002703 (the AZ0a "well-
tracked" reference case) both track cleanly.

Residual, minor, not chased further this round: max r-hat 1.05 (3 bad
params) and year-allocation confidence (87.5% vs AZ0a's 96%) are both
slightly worse than AZ0a's baseline -- plausibly just the ordinary cost of
one new parameter, not a sign of a deeper problem the way the earlier
attempts' 70%/29% flatness was. Worth a quick look if this gets combined
with other phases later, not worth blocking on now.

**Lesson for later phases**: this is the second time in this round
(after AZ1b) that the "obvious" richer/more granular construction turned
out to be the wrong level of complexity, and a simpler, more targeted
version -- built only after directly diagnosing WHAT the data actually
needed -- won cleanly. Worth defaulting to "smallest change that targets
the diagnosed need" before reaching for a full hierarchy, in Phase 3 and
beyond.

**Follow-up investigation: the residual low bulk ESS (min_ess=47,
`sigma_delta_top_boost`).** Requested specifically: look deeper than the
diagnose-table summary number, and check bulk vs tail ESS separately
(previously only a single blended number was reported).

- Bulk ESS is much worse than tail ESS across all three scalars
  (`sigma_plan`: 308 vs 2389; `sigma_ben`: 71 vs 271;
  `sigma_delta_top_boost`: 47 vs 185) -- the central/bulk region is harder
  for the chains to agree on than the tails.
- Ruled out ordinary slow within-chain mixing: autocorrelation at lags up
  to 100 is near zero for every chain on both `sigma_ben` and
  `sigma_delta_top_boost`, and each chain's quarter-by-quarter mean is
  stable (no drift/trend within a chain).
- It IS a small, persistent BETWEEN-chain disagreement: e.g.
  `sigma_delta_top_boost`'s 4 chain means range 28.5-30.3 and stay there
  for the whole run, never converging toward each other or drifting
  further apart. This is what depresses rank-normalized bulk ESS even
  with good local mixing -- a much milder, non-catastrophic cousin of
  AZ1b's genuine multimodality (one stretched/shallow basin, not
  disconnected modes).
- Checked for a ridge/tradeoff between `sigma_plan`, `sigma_ben`, and
  `sigma_delta_top_boost` directly (the natural candidates, since they all
  compete to explain the same top-quartile areas' spikiness) -- no
  meaningful pairwise correlation found (all |r| < 0.12). Also found a
  real, moderate (r=0.48) correlation between `top_boost` and the actual
  realized |delta| magnitude in top-quartile areas -- an expected,
  structural "prior scale correlates with what it governs" relationship,
  not obviously a problem on its own.
- **Tested the "just needs more draws" hypothesis directly (draws
  1500->4000, tune 500->1500) -- result is genuinely mixed, not a clean
  fix.** `sigma_delta_top_boost` improved a lot (bulk ESS 47->194, r-hat
  1.05->1.02). But `sigma_plan` got WORSE (bulk ESS 308->94, r-hat
  1.015->1.029). Re-checked pairwise correlations in the longer run too --
  still no meaningful pairwise relationship among the three named scalars
  (all |r| < 0.1). So the disagreement doesn't reduce to a simple 2-3
  variable ridge among the named scalars; it plausibly involves the much
  higher-dimensional per-area `z`/`delta` space (200 areas x 10 years),
  which hasn't been (and isn't practical to) fully explored by hand.

**Status: open, low-priority.** r-hat stays mild (1.02-1.05, never above
that), zero divergences, chains are stable not multimodal, and AZ2's
*practical* behaviour is separately validated as good (9.0%
frac_flat_despite_active, better than AZ0a's 11.5%; spike-tracking plot
confirmed). Given the mixed/inconclusive result from the direct test and
the low practical stakes, not recommended to chase further right now --
flagged here for visibility if it resurfaces or matters more once AZ2 is
combined with other phases (Phase 4).

### Phase 3 — floored outlier/noise branch + automatic flagging — BUILT, verdict still open
Built `_build_noise_mixture_likelihood` (per-cell 2-way mixture: obs
explained by z, or by a noise branch centred at 0 with a HARD floor on
its scale, `sigma_noise = 25 + HalfNormal(...)` — the floor is an
unconditional additive constant, not just a small prior scale, learning
directly from AZ0's collapse where a HalfNormal ALONE still has its mode
at 0 regardless of its own sigma). `AZ3` = AZ0a + this mixture on both P
and E, independent branch, `rho ~ Beta(8,2)` (prior mean 0.8, expecting a
small noise minority).

**Result**: `frac_flat_despite_active` improved a lot (11.5% -> 2.5%).
`rho_P`/`rho_E` converged to ~0.63-0.66 (not close to the prior's mean
0.8), `sigma_noise_P`/`sigma_noise_E` converged to almost exactly the
floor (25.06/25.13). Checked the mechanism against its own motivating
case, E01001774 (D=18, P_sum=460): works correctly -- `resp_noise_P`
cleanly separates quiet years (0.04-0.21) from the genuinely extreme ones
(0.99-1.0).

**Correction, logged so the mistake isn't silently dropped**: this
section originally also claimed a "regression" at E01002703, described as
"the clean, cross-source-corroborated ~270 spike every other AZ model
tracks well," on the basis that `resp_noise` there was high (0.77-0.84).
**That claim was wrong and unverified.** Checked against the actual raw
data only after the user directly disputed it: 2013 P_obs=274 but
E_obs=8 the SAME year; E's own peak (162) is in 2012, when P_obs=0. P and
E never agree at this LSOA, in either magnitude or timing -- it is a
single uncorroborated source spike, not the clean agreed case claimed.
AZ3 assigning it meaningful noise-probability is therefore not obviously
wrong, and REFERENCE_AREAS'/this doc's description of it has been
corrected. The general "does resp_noise track magnitude reasonably"
question was re-checked properly afterward: 70% of cells with
`|P_obs|>50` get `resp_noise_P>0.5`, which is also not obviously wrong
given how much of this dataset's large single-cell values are known,
verified magnitude mismatches (E01033491: P=762 vs D=126; E01001774:
P=398 vs D=18) rather than genuine spikes.

**Status: open, but the case for "a real problem" is much weaker than
first claimed.** What's left, still true and not dependent on the
retracted example: `rho` converged well below its prior mean (0.63-0.66
vs 0.8), `sigma_noise` sits at the floor, and year-allocation confidence
dropped to 52% low-confidence (vs AZ0a's 4%, worse than AZ1b's 13.5% or
AZ2's 12.5%). Whether that combination represents miscalibration or an
accurate reflection of how noisy/disagreeing this dataset genuinely is
(175/200 areas already found to disagree on WHICH year has a source's
spike) is not yet resolved -- needs either a principled read on the
prior/data tension, or a held-out-style check, rather than another single
illustrative example treated as decisive. Lesson: verify a specific
per-area claim against raw data BEFORE using it as evidence, not after
being challenged on it.

**Follow-up: noise-marking visualisation, cross-area pattern check, and a
resolved deep-dive into E01002702 -- net verdict now positive.**

1. **Plotting**: `plot_z_area`/`plot_spike_tracking_examples` now colour-code
   P_obs/E_obs markers by posterior mean `resp_noise` (green=signal,
   red=noise, shared colourbar) whenever a trace has the noise-mixture
   variables, so which specific points a model is discounting is visible
   directly on the plot. Applies automatically to any current/future model
   with this mechanism, not just AZ3.

2. **Cross-area pattern check -- the noise-marking is well-behaved,
   reversing the earlier negative read**:
   - `resp_noise_P` correlates sensibly with cross-source corroboration:
     mean 0.68 for P-active-but-E-silent cells, dropping monotonically to
     0.45 for cells where E closely matches or exceeds P (corr=-0.25).
   - Independent validation against `outliers.py`'s raw-data threshold
     flagging (a method with zero knowledge of the model): flagged cells
     get mean resp_noise=0.83 vs 0.36 for everything else. Visual check
     (`results/scratch/az3_noise_pattern_examples.png`) confirms this
     cleanly: EVERY one of the 5 independently-flagged outlier spikes in
     this 200-area sample renders deep red (resp_noise~1.0), and z
     visibly ignores every one of them in favour of a more plausible
     path. Reverses the earlier "real problem" framing -- that framing
     leaned on the retracted E01002703 example; the properly-checked
     aggregate pattern does not support it.

3. **E01002702 deep-dive (2013/2014 "why not z=0" question) -- resolved,
   and it explains an open puzzle from earlier too.** Constructed the
   user's proposed alternative (z=0 in 2013 AND 2014, freed mass
   redistributed proportionally into 2016/2019/2021) and compared its
   exact log-posterior-density against the current posterior-MEAN
   configuration directly (closed-form: AZ3's sigma_delta is fixed, not
   sampled, so the ZeroSumNormal prior term and the mixture-likelihood
   terms are both computable exactly, without needing MCMC). **The user's
   alternative has HIGHER total logp (+6.17 nats, prior -2.09 against it
   but both P_like and E_like clearly for it, +4.44 and +3.82) than the
   posterior mean.** This is not a case of the model preferring a worse
   solution -- checking the actual per-draw distribution (not just its
   mean) showed why: z's marginal posterior for 2013/2014 is genuinely
   bimodal (histogram: `results/scratch/az3_E01002702_z_bimodality.png`)
   -- a sharp spike at ~0 (the SINGLE MOST COMMON individual draw, ~50%
   of all draws for 2013) plus a broad secondary hump from ~30-150. The
   posterior MEAN (34.5) sits in the low-density valley BETWEEN these,
   representing neither explanation well -- exactly consistent with the
   alternative-config point beating the mean-config point on logp.
   Checked per-chain (not just pooled): all 4 chains show the same
   ~48-56% near-zero fraction, so this is NOT AZ1b-style hard multimodality
   (chains splitting onto different answers) -- it's a single, genuinely
   bimodal marginal distribution that every chain samples correctly and
   consistently. r-hat/ESS look fine here for exactly that reason; the
   problem is that the MEAN (used throughout this investigation's plots
   and tables as the summary z value) is a poor, misleading statistic for
   a bimodal quantity, not that sampling failed.

   **Root cause, and it retroactively explains the `rho` puzzle above**:
   `sigma_plan`/`sigma_ben` converged to ~0.58/0.99 -- dramatically
   tighter than every other AZ0a-family model (~7-9). Checked, not
   assumed. A signal branch this tight makes ANY mismatch between z and
   an observation extremely costly, effectively forcing a near-discrete
   choice per cell ("z matches almost exactly" vs "call it noise") --
   the same discrete-choice-flavoured geometry that produced hard
   multimodality in AZ1b's lag weights, now showing up as SOFT (within-
   chain) multimodality in z itself, because the zero-sum constraint
   couples each cell's resolution to how the area's OTHER ambiguous cells
   resolve. This also explains why `rho` sits at 0.63-0.66 rather than
   the prior's 0.8: with sigma_plan/sigma_ben this tight, far more cells
   fail to fit the signal branch acceptably than would under a normal-
   sized sigma_plan, pushing more mass to the noise branch by construction,
   not because the noise branch is mis-specified.

   **Practical implication**: for areas/years like this, report the
   *shape* of z's marginal (or at least flag genuine multimodality),
   not just its mean -- the current spike-tracking plots' point-and-CI
   summary can visually understate how resolved a specific year's z
   actually is. -- **built, see next item.**

**Status: net positive.** The noise mechanism appears well-calibrated in
its ranking/direction (corroboration, independent validation) even though
its overall level (`rho`, `sigma_plan`/`sigma_ben`) is unusual --
and that unusual level now has a concrete, understood cause rather than
being an unexplained miscalibration.

4. **How widespread is z-multimodality, and a new visualization for it --
   built and validated.** Built `diagnostics.detect_z_multimodality`: a
   KDE-based mode count per (area, year) cell (`scipy.stats.gaussian_kde`
   + `scipy.signal.find_peaks`, with a prominence threshold to avoid
   counting KDE ripples as genuine modes). Validated against E01002702's
   already-known cells before trusting it at scale: correctly finds 1
   mode for its confident years (2012, 2017) and 2 for its genuinely
   bimodal ones (2013, 2014) -- also revealed 2016/2019/2021 (the years
   with big, mostly-noise-flagged E spikes) are themselves multimodal too
   (2019 even trimodal).

   **Scanned all 2000 LSOA-years in AZ3**: **34.6% (692/2000) have >=2
   modes, and 89% of areas (178/200) have at least one multimodal year**
   -- this is not a rare edge case, it's pervasive. Some areas show it in
   8-9 of their 10 years (e.g. E01035709, E01033706, E01002769: 9/10).
   This mechanism is expected to be specific to models with a tight-
   signal-branch mixture likelihood (AZ3), not a general AZ-family
   property, though the scan itself works on any model's z.

   **Built `plots.plot_z_area_modes`** as the replacement visualization:
   clusters entire posterior DRAWS (the full per-draw n_years vector for
   an area, not each year independently -- deliberate, since a "mode" is
   a coherent whole-row story, e.g. "2013 and 2014 both ~0, extra
   concentrated in 2016/2019/2021" is ONE scenario, not an independent
   per-year coin flip) via `scipy.cluster.vq.kmeans2` on per-year-
   standardized draws, then plots each cluster's own mean+CI band
   labelled with its share of the posterior, instead of one blended
   band. Each cluster's mean still sums to exactly D (every draw does,
   individually, by construction, so every cluster mean inherits that).

   Compared old vs new directly on real traces
   (`results/scratch/az3_mode_decomposition_compare.png`):
   - E01002702: cleanly separates into a ~53%/47% split matching the
     qualitative story already found by hand.
   - E01035709 (9/10 multimodal years, the most extreme case in the
     scan): FIRST presented as "two coherent, confident scenarios" --
     **this claim was wrong and made without checking, and the user
     correctly called it out.** See the correction immediately below
     (item 4b) before trusting anything about this area's decomposition.

   **4b. Correction + a real fix to the tool, prompted by the user
   directly disputing the E01035709 claim.** Checked E01035709's raw
   data first: literally P_obs=E_obs=0 in 9 of 10 years (one year has a
   trivial 5/5). With essentially zero data, "two coherent scenarios" was
   never a claim that should have been made without verification. Checked
   what's actually driving the apparent 72%/28% k-means split: computed,
   for every posterior draw, which single year has the highest z --
   found this is spread almost UNIFORMLY across all 10 years (~0.10 each,
   close to pure exchangeability), not concentrated on 2 candidates. The
   "2 scenarios" were an artefact of forcing k=2 on diffuse, near-
   exchangeable uncertainty (unsurprising -- with zero data and a
   symmetric prior, no year is distinguished from any other), not 2 real
   stories. Silhouette score did not catch this (E01035709's was a
   middling 0.17) -- separately scanned all 200 areas and found
   `corr(silhouette, n_active_P/E_cells) = 0.085`, i.e. essentially no
   relationship between how well-"separated" a k-means split looks and
   how much real data supports it (one area with ZERO active cells had
   silhouette 0.56, better-separated-looking than most areas with real
   data).

   Fixed `plot_z_area_modes` itself rather than just noting this in
   prose: added a "top-n_clusters-year concentration" figure to the
   title (fraction of draws whose single highest-z year falls among the
   n_clusters most commonly-highest years) -- this is what actually
   caught the E01035709 problem, unlike silhouette. Below 50% now prints
   an explicit "LOW: scenarios may not be real" warning on the plot
   itself, not just in a docstring someone has to go read.

   Re-selected examples properly this time, screening on this
   concentration metric (not silhouette) before presenting anything:
   E01002739 is a genuine, well-concentrated 2-scenario case (86%
   concentration, real divergence around 2017/2020). Two other high-
   silhouette candidates (E01002695, E01001738) turned out to collapse to
   ONE dominant scenario once tiny clusters were filtered (91%/87%
   concentration but only 1 surviving cluster) -- i.e. per-cell
   multimodality flagged by `detect_z_multimodality` doesn't always mean
   the WHOLE area is multi-scenario; sometimes it's one clear shape with
   local per-cell noise. E01035709 now correctly renders with the "LOW"
   warning (concentration=32%). All four in
   `results/scratch/az3_mode_examples_v2.png`.

   **Lesson, stated plainly since it recurred**: check a specific
   per-area claim against the actual numbers before presenting it with
   confidence, especially favourable-looking ones (a clean-looking
   72%/28% split, a high silhouette score) -- both looked like evidence
   of real structure and were not, in exactly the way an unverified
   "P and E agree at E01002703" claim wasn't earlier in this phase.

   Not yet added to the standard `plot_spike_tracking_examples` grid
   (which still uses the single-band `plot_z_area`) -- worth wiring in,
   e.g. auto-switching to the mode-decomposed version (with its
   concentration check) for areas `detect_z_multimodality` flags, once
   there's a settled view on whether Phase 3 continues in its current
   form (see ESS item below).

5. **AZ3's low min ESS, investigated on request -- a different mechanism
   than the z-multimodality above, not the same thing.** `sigma_plan` is
   the worst offender (rhat=1.074, ESS bulk=43, ESS tail=55), followed by
   `rho_P` (rhat=1.034, ESS bulk=97, tail=713); `sigma_ben`/`rho_E` are
   milder; `sigma_noise_P`/`sigma_noise_E` are excellent (ESS>3600).
   Checked autocorrelation directly for `sigma_plan`: high and slow-
   decaying (lag1~0.88-0.95, still 0.4-0.7 at lag20, ~0 by lag100-200) --
   the opposite signature from z's cells (near-zero autocorrelation,
   clean within-chain switching). This is classic small-scale funnel
   geometry, not multimodality: `sigma_plan` converged to ~0.58, an order
   of magnitude below every other AZ0a-family model's ~7-9 (the same
   collapse-toward-small-values dynamic already implicated in the `rho`
   puzzle), and a scale parameter converging that tight couples strongly
   to ~1000 P-cells' signal/noise decisions simultaneously, which is
   exactly the geometry NUTS struggles with. Confirmed direct coupling:
   `corr(sigma_plan, rho_P) = 0.43`.

   Bounded, not catastrophic: no divergences, all 4 chains agree on the
   same rough region (means 0.52-0.64, not split like AZ1b's hard
   multimodality), autocorrelation does eventually decay. Read: the
   qualitative finding (signal branch collapsed unusually tight) is
   robust; the precise numeric values of `sigma_plan`/`rho_P` carry more
   uncertainty than the draw count suggests and shouldn't be trusted
   past rough magnitude.

   **Tried: floor on `sigma_plan`/`sigma_ben`, `sigma_obs_floor=2.0` +
   `HalfNormal(3)` excess, exactly mirroring `sigma_noise_floor`.**
   Implemented as `sigma_plan = 2.0 + sigma_plan_excess`,
   `sigma_plan_excess ~ HalfNormal(3)` (same for `sigma_ben`). Resampled
   on the real 200-area dataset and re-ran `diagnose`, `detect_z_multimodality`
   -- checked empirically, not assumed:

   | | max r-hat | min ESS | frac_flat_despite_active | low year-confidence | LSOA-years multimodal | areas w/ any multimodal cell |
   |---|---|---|---|---|---|---|
   | AZ3 unfloored | 1.074 (sigma_plan) | 43 (bulk) | 2.5% | 52% | 34.6% (692/2000) | 89% (178/200) |
   | AZ3 floored | 1.0039 | 1627 | 3.5% | 24.5% | 20.3% (406/2000) | 60.5% (121/200) |
   | AZ0a (baseline) | 1.006 | 3180 | 11.5% | 4.0% | -- | -- |

   Per-scalar detail on the floored run: `sigma_plan` r-hat=1.00, ESS
   bulk=6086/tail=3540 (was bulk=43); `sigma_ben` bulk=4998/tail=3124;
   `rho_P`/`rho_E` bulk=1691/1627 (worst of the six, still comfortably
   over 400) -- `rho_P`/`rho_E` also moved closer to their Beta(8,2)
   prior mean of 0.8 (0.7385/0.7048, up from 0.63-0.66 unfloored, though
   still not fully there); `sigma_noise_P`/`sigma_noise_E` unchanged
   (~25.1/25.4, at their own floor as before).

   **Read: the ESS problem is fixed, and it dragged three other things
   with it (year-allocation confidence, `rho` positioning, multimodality
   prevalence) as predicted, consistent with them sharing one root
   mechanism.** `frac_flat_despite_active` also got slightly WORSE than
   the unfloored version (2.5% -> 3.5%, though still much better than
   AZ0a's 11.5%) -- a minor tradeoff.

   **Correction, from a deeper follow-up investigation (full detail in
   the `az3-floor-followup` artifact) -- the "actively binding floor,
   still collapsing toward zero" framing immediately above was written
   from the sd alone and turned out to be too alarmist; checked
   properly with a profile-log-likelihood computation and it does not
   hold up.** Holding z and rho at their converged posterior means and
   sweeping candidate signal-branch sigma over a grid, the model's own
   implied log-likelihood is maximized at ~2.5 for both P and E -- just
   above the 2.0 floor, not down near 0. The floor is barely doing any
   work given the current fit; `sigma_plan`/`sigma_ben` landing at
   2.01/2.02 with sd~0.014 looks to be close to genuinely optimal, not
   wall-pinned. (Caveat carried over from the artifact: this profile
   check is LOCAL -- it can't rule out a completely different
   equilibrium existing elsewhere in the space, since z was itself
   fitted self-consistently under a small sigma_plan.)

   What the deeper check found instead: standardized residuals for
   cells nominally classified "signal" (`resp_noise<0.5`) have std~3.26,
   about 2.3x wider than a properly-calibrated StudentT(nu=4, scale=2.0)
   predicts (theoretical std 1.41) -- even though the near-zero core of
   the residual distribution matches the StudentT(4) reference closely
   (35.7% expected within +/-0.5 sigma vs 33.8%/32.8% observed). Read:
   the "signal vs noise" split is not cleanly separating two
   populations -- there's a large ambiguous, boundary-straddling middle
   (visible as a second shoulder around -2 to -3 sigma in the residual
   histograms), consistent with rho landing at 0.70-0.74 rather than
   near 0 or 1, and this same per-cell ambiguity is what's generating
   z's multimodality (see below) -- not a broken floor.

   **The multimodality-prevalence drop (89% -> 60.5% of areas), dug
   into further per explicit request -- it's a real but partial and
   mechanistic effect, not evidence the ambiguity went away.** Matched
   each pre-floor multimodal cell to its floored-model outcome:
   331/692 (48%) resolved to unimodal, 361/692 (52%) are still
   multimodal, and a further 45 cells that were unimodal before became
   newly multimodal (net: 692 -> 406 multimodal cells, 34.6% -> 20.3%).
   Resolved cells had a much smaller pre-floor mode gap (median 10.0)
   than retained cells (median 32.1) -- Mann-Whitney p~6.5e-57. Read:
   widening the signal-branch scale ~3.5x (0.58 -> 2.0) smooths over
   NEARBY competing explanations (gap <~15) into one blended estimate,
   but doesn't touch genuinely far-apart alternatives (gap >~30) -- it's
   a resolution/blurring effect, not a resolution-of-disagreement
   effect. Whether blending a 10-unit-gap bimodality into one number is
   correct (the disagreement wasn't big enough to matter) or a loss of
   real information is a judgement call, not yet settled.

   **Bigger finding, and a correction to this round's own methodology:
   cross-validated the mode-decomposed plot's whole-draw k-means split
   against the per-cell KDE scan directly, and the two frequently
   disagree.** Of 115 areas the per-cell scan flags as multimodal
   somewhere, only 27 (23%) have a whole-draw 2-cluster split whose
   per-year gap actually matches the KDE-detected mode gap in every
   flagged year; 49 (43%) partially match; 39 (34%) don't match at all.
   Worse, checked the 79 areas the per-cell scan finds are unimodal in
   EVERY single year, and 31 of them (39%) still produce a
   high-confidence (>70%) "2-scenario" split under `plot_z_area_modes`'
   argmax-year concentration check -- **including E01002739, this
   round's own earlier "validated genuine 2-scenario" example**, which
   had real per-cell bimodality unfloored (5/10 years, gaps 3.8-12,
   squarely in the "resolved" range above) and is now fully unimodal
   per-cell, yet still renders as a crisp, confident-looking 91%
   -concentration two-line plot. Read: the concentration check (built
   two turns ago specifically to rule out diffuse/exchangeable false
   positives like E01035709) does that job correctly, but was never
   sufficient on its own to prove genuine multimodality -- a
   zero-sum-constrained, "peaky" (few dominant years) but fully unimodal
   joint posterior can still produce a confident-looking 2-cluster split
   by chance, because k-means is forced to produce 2 clusters
   regardless. **`plot_z_area_modes` output should only be trusted for
   an area once `detect_z_multimodality` independently confirms at
   least one of its years is actually multimodal AND the cluster-mean
   gap in that year roughly matches the KDE-detected mode gap** -- not
   from the concentration score alone. Not yet fixed in the plotting
   function itself; the cross-validation check currently only exists as
   a one-off script in `az3_floor_investigation.py`, not wired into
   `plots.core`.

### Phase 4 — combine validated pieces — BUILT (AZ4), verdict mixed
Full diagnosis in `docs/az-ess-diagnosis.md` (Phase 4 section). `AZ4` = AZ0a +
AZ2b's smooth top-boost z-prior + AZ1b's area-hierarchical lag (both sources)
+ AZ3's floored noise-mixture likelihood, applied to the lag-convolved
`P_mean`/`E_mean`. Sampled on real 200-area data, 8 chains.

**Best-in-family on the metric that matters most (`frac_flat_despite_active`
= 2.0%, beating every single-piece model including AZ3's 3.5%), but also
worst-in-family on year-allocation confidence (38.0% low-confidence, worse
than any single component) and BEN coverage (`ben_cov_90=0.796`).** The
composition-risk flagged before building it (AZ1b's lag ambiguity and AZ3's
signal/noise ambiguity compounding) was checked directly and partly
confirmed: `sigma_noise_E` collapsed from AZ3's ESS>3600 to 33, and
`sigma_delta_top_boost` collapsed from AZ2b's 605 to 40 — real regressions
in pieces that were individually fixed, concentrated specifically in E (the
source with the more genuinely ambiguous lag category throughout this
round), not a random/symmetric effect. Zero divergences; the degraded
scalars show the same small-persistent-disagreement (shallow basin)
signature as AZ2's/AZ3's original problems, not AZ1b's hard multimodality
recurring. Spike-tracking spot-check: real wins (E01033711's two genuine
spikes) preserved and even more confidently tracked, but with new spurious-
looking dips in previously-quiet years.

**Not yet a recommended finalist as-is.** Four follow-up options on the
table, none chosen yet: investigate `sigma_noise_E` specifically; try a
wider `sigma_noise_floor` or revisit `top_boost`'s prior for AZ4
specifically; let the still-deferred LOO comparison be the actual
tie-breaker between AZ4 and its simpler components; or report AZ4 as-is with
an explicit "reliable for total change and which spikes are real, not for
which year" caveat, mirroring AZ1b's and AZ3's own individual caveats.

### Phase 5 — AZ5: AZ1g + AZ3, a narrower two-piece combination — SAMPLED, verdict mixed (not a finalist)
Deliberately narrower than Phase 4's `AZ4`/`AZ4b` (three pieces): combines exactly the two
independently-validated branches this step was asked to combine — `AZ1g` (the best AZ1-branch
model: P-only regularized/slab-capped horseshoe hierarchical lag,
`_build_hierarchical_lag_regularized_horseshoe`; E stays same-year, per AZ1d's original
rationale that E's lag category was disproportionately unstable throughout this round) and
`AZ3` (floored noise/outlier mixture likelihood, `sigma_obs_floor=2.0 + HalfNormal(3)` excess
on `sigma_plan`/`sigma_ben`, `sigma_noise_floor=25.0`). AZ2b's smooth top-boost z-prior is
deliberately NOT included — per this round's "combine at most two at once" rule, so any
regression is attributable to the AZ1g/AZ3 interaction alone, not a third simultaneous change.

Composition, mirroring AZ4's precedent: AZ3's noise mixture applies to the lag-convolved
`P_mean` for P (the piece that has one), and to raw `z` directly for E (which has no lag
mechanism here to convolve). `sigma_plan`/`sigma_ben` use AZ3's floored construction rather
than AZ1g's original plain `HalfNormal(2)` — AZ1g predates AZ3's own diagnosis of the
collapse-to-near-zero funnel, so carrying the fix forward avoids reintroducing a known,
already-fixed failure mode. `target_accept=0.98`/8 chains/8 cores carried forward from AZ1g
unchanged (not relaxed to AZ4's 0.95), since the horseshoe geometry that required them is
still present.

**Composition risk flagged going in**: unlike AZ4 (where the lag/noise-mixture leakage ran
through a third piece, `sigma_delta_top_boost`, with no analogue in AZ5), here the regularized
horseshoe's per-area escape valve and the noise mixture's per-cell escape valve compete
directly for the SAME ambiguous large-|D| P-cells — whether that competition helps (each
absorbs a distinct part of the ambiguity) or compounds (the same cells become doubly-flexible
and unidentifiably split between "different lag" and "just noise") was the open question this
sampling run was built to answer. Structure-only tests pass (`TestAZ5Structure`,
`tests/test_models.py`): both pieces' named variables present, P's `lambda_weights` are valid
simplices, no E-lag machinery, `sigma_plan`/`sigma_ben`/`sigma_noise_P`/`sigma_noise_E` never
below their floors, `z` sums to `D` exactly.

**Sampled on the same real 200-area (Islington-centred) dataset used throughout this round, 8
chains, `target_accept=0.98`.** Checked via `diagnose --models AZ5,AZ1g,AZ3,AZ0a`:

| | max r-hat | min ESS | frac_flat_despite_active | low year-confidence |
|---|---|---|---|---|
| AZ0a (baseline) | 1.006 | 3180 | 11.5% | 4.0% |
| AZ1g | 1.021 | 437 | 7.5% | 8.0% |
| AZ3 | 1.004 | 1627 | 3.5% | 24.5% |
| **AZ5 (AZ1g + AZ3)** | **1.107** | **50** | **1.5%** | **35.5%** |

**Best-in-family on the metric that matters most** (`frac_flat_despite_active` = 1.5%, beating
both individual parents), **but the composition risk materialised on every other axis**: max
r-hat and min ESS are both worse than EITHER parent alone (not just worse than the better one),
and year-allocation confidence (35.5% low-confidence) is worse than both AZ1g's 8.0% and AZ3's
24.5% — i.e. combining these two specific pieces didn't split the difference, it compounded the
convergence cost on both fronts simultaneously, more directly and immediately than AZ4's
three-piece combination did.

**Ran `check-multimodality --models AZ5 --lag-var lag_P_lambda_weights` before drawing any
r-hat conclusion, per the standing pipeline** — 62 area/cell findings across `lag_P_lambda_weights`
plus AZ5's 8 named scalars: 2 `hard_genuine`, 0 `stuck_fixable`, 26 `round_tripping` (benign,
report pooled posterior as-is), 0 `mixed`, 29 `needs_review` (mostly "fewer than 2 pure
chain-groups" — diffuse rather than cleanly split), and 5 scalar cells (`lag_P_mu_logit[0]`,
`lag_P_mu_logit[1]`, `lag_P_global_tau[0]`, `rho_P`, `rho_E`) classified `not_multimodal`.
**The headline max r-hat (1.107) belongs to `lag_P_mu_logit[1]`, one of the `not_multimodal`
cells** — best-case r-hat/ESS (excluding every attributable-to-multimodality finding) comes
back identical to raw (1.107/50), confirming this isn't a multimodality story at all for the
worst offender: it needs `docs/ess-rhat-diagnostic-guide.md`'s own procedure, not this
pipeline, and that diagnosis has not yet been done. With 29 findings still `needs_review`,
none of AZ5's r-hat/ESS numbers should be treated as a finished diagnosis either way.

**Spike-tracking plot** (`results/scratch/az5_spike_diagnostic.png`) generated and inspected
before writing any of the above: reference cases behave as expected from AZ3's already-validated
noise-flagging (E01033491, E01001774, E01002703 all show their known spurious spikes rendered
deep red/high-`resp_noise`, z correctly declining to chase them), and E01033711 (D=634,
AZ0a/AZ1a's worst-missed case, AZ1b's flagship win) is still tracked at both its 2014 and 2018
BEN spikes despite AZ5 having no E-lag mechanism at all — suggesting the same-year E likelihood
alone, combined with the noise mixture's tolerance, is sufficient for this specific area,
though this has only been checked visually here, not via the same per-area verification rigor
`hierarchical_mode_summary` gave AZ1b's win. No detailed per-panel resp_noise values were read
off the plot for this write-up (per this round's "verify a specific claim before publishing"
lesson) — only the pattern-level read above.

**Status: not a recommended finalist.** Real, honest tension: this is the best single-metric
result in the family on the metric the stakeholder cares about most (frac_flat_despite_active),
achieved by a genuinely narrower, more attributable combination than AZ4 (two pieces, not
three) — but it bought that at a bigger year-allocation-confidence cost than AZ4 itself paid
(35.5% vs AZ4's 38.0%, so comparable, not better) while ALSO leaving an undiagnosed,
not-multimodality-attributable r-hat problem that neither AZ1g nor AZ3 had on their own.
**Next steps, not yet chosen**: (1) run `docs/ess-rhat-diagnostic-guide.md`'s procedure on
`lag_P_mu_logit`/`lag_P_global_tau`/`rho_P`/`rho_E` specifically, since these are now confirmed
NOT multimodality and were never diagnosed that way before being combined; (2) work through the
29 `needs_review` `lag_P_lambda_weights` areas properly before trusting the adjusted r-hat
picture; (3) treat this as one more data point (alongside AZ4/AZ4b) that this dataset's
per-area ambiguity resists being cleanly absorbed by combining escape-valve mechanisms, and
consider whether the deferred LOO/k-fold comparison should be brought forward specifically to
adjudicate between AZ3 (simplest, cleanest single-piece win), AZ5, and AZ4/AZ4b now that there
are several honestly-flawed combination candidates rather than one obvious next step.

### Phase 6 — redundancy-hypothesis check, model comparison, full-dataset AZ3 validation
**Hypothesis check, requested before trusting it**: AZ5's docstring hypothesised that P's two
escape valves (regularized-horseshoe lag reallocation, noise-mixture flagging) compete for the
same ambiguous large-|D| cells, causing the combined convergence/year-confidence regression.
Checked directly on the AZ5 trace, not assumed — **refuted, not confirmed**: area-level
correlation between lag ambiguity (`1 - max(lambda_P)`) and mean `resp_noise_P` on that area's
active P cells is **negative** (r=-0.328, p=3.6e-6) and the scalar-level correlations
(`rho_P`/`rho_E` vs `lag_P_global_tau`/`lag_P_mu_logit`) are all |r|<0.15 — the opposite of what
a genuine competing-explanations ridge would show. High-lag-ambiguity areas actually have LOWER
noise-flagging, not higher. The mechanism behind AZ5's regression remains undiagnosed; per user
instruction this was not chased further this round (see the still-open `ess-rhat-diagnostic-
guide.md` next step logged in Phase 5).

**Fixed `compute_model_comparison`'s hardcoded `var_name='P_like'`** (`analysis.py`,
`_joint_pe_loglik`) — it was silently scoring P only, ignoring E entirely, a gap already flagged
in `docs/model-evaluation-methods.md`. Fixing it surfaced a second, more serious bug on the way:
naively summing `P_like + E_like` via xarray's own `+` silently broadcast a **179 GiB cross-join**
for traces (e.g. AZ0a) where P_like/E_like were attached with each variable's own auto-generated
dim names instead of shared `('area','year')` dims — xarray's dimension-name alignment treats
differently-named same-length dims as genuinely different axes rather than raising. Fixed by
summing via raw `.values` and rewrapping into `P_like`'s own coords, bypassing xarray's
dim-name-based alignment entirely. Both bugs covered by new regression tests
(`tests/test_diagnostics.py::TestComputeModelComparison`, a new `mock_traces_with_pe_ll` fixture
with deliberately mismatched P/E dim names, matching the real AZ0a case).

**First-pass LOO comparison (PSIS, joint P+E) across all 8 sampled AZ-family models, same
200-area sample**:

| rank | model | elpd (P+E) | elpd_diff | dse | frac_flat | max r-hat | min ESS | low-year-conf | % Pareto-k>0.7 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | AZ4b | -15176 | 0 | — | 1.0% | 1.459 | 15 | 36.0% | 30.4% |
| 1 | AZ4 | -15227 | -50 | 29 | 2.0% | 1.168 | 31 | 38.0% | 36.8% |
| 2 | AZ5 | -15711 | -500 | 53 | 1.5% | 1.107 | 50 | 35.5% | 17.4% (best) |
| 3 | AZ3 | -16954 | -1800 | 64 | 3.5% | 1.004 (best) | 1627 (best) | 24.5% (best) | 25.1% |
| 4-7 | AZ1g, AZ2, AZ2b, AZ0a | -17515 to -19511 | -2300 to -4300 | — | — | — | — | — | 19.7-40.4% |

**Two things temper this ranking rather than settle it**: (1) AZ4 vs AZ4b are statistically
indistinguishable (|elpd_diff/dse|=1.7 < 2) yet AZ4b has markedly WORSE r-hat/ESS than AZ4 (1.459
vs 1.168, ESS 15 vs 31) — its tau cap bought a fragile, noise-level LOO edge at a real
convergence cost, the same pattern AZ1c already showed on its own turf. (2) **every model shows
17-40% of Pareto-k > 0.7** on the joint P+E likelihood, a materially worse reliability picture
than this codebase's earlier P-only Pareto-k checks (e.g. AZ0a: 5% bad-k on P alone vs 28.9%
once E is scored too) — standard PSIS-LOO is not trustworthy at face value for ANY current
family member, not just the discrete-mixture ones `model-evaluation-methods.md` flagged as
higher-risk going in. Per that doc's own decision tree this points toward K-fold CV as the
principled next step, but **user's call: don't build the K-fold `SamplingWrapper` yet** — treat
this comparison as directional only, revisit once there's a smaller finalist set.

**User selected AZ3 for a full-dataset (4987-area, all London LSOAs minus the 7 hard outliers)
validation run**, given its clean convergence and best year-confidence among the four real
candidates. **Found and worked around a CLI gap on the way**: `cmd_run_models` (`cli.py`)
silently defaults to `n_areas=200` whenever `--n-areas` is omitted (`args.n_areas or 200`) —
there is no flag to request the full dataset explicitly, and a first attempt without
`--n-areas` silently re-subsampled to 200 areas rather than running on everything. Worked around
by passing `--n-areas 4987` (`select_spatial_sample`'s `nsmallest(n)` just returns everything
when `n` exceeds the available count); worth a proper CLI fix later (e.g. a `--full` flag or
making the missing-value behaviour "all areas" instead of a silent 200).

**Result: AZ3 generalises cleanly from the 200-area development sample to the full dataset** —
convergence is essentially as good, proportionally even slightly better:

| | 200 areas | 4987 areas (full) |
|---|---|---|
| frac_flat_despite_active | 3.5% | 4.7% |
| max r-hat | 1.004 | 1.011 |
| min ESS | 1627 | 606 |
| divergences | 0 | 0 |
| plan_cov_90 / ben_cov_90 | 0.912 / 0.849 | 0.921 / 0.903 |
| low year-confidence | 24.5% | 23.3% |

Spike-tracking plot (`results/scratch/az3_full_spike_diagnostic.png`) confirms the same
qualitative behaviour holds at full scale: reference outlier cases (E01033491, E01001774) are
still correctly noise-flagged, and E01033711 (D=634, the flagship spike-tracking case) is still
tracked. Saved to a separate `results/traces_full/` directory, not overwriting the 200-area
traces used for the comparison above.

**Status: AZ3 is the strongest evidence-backed finalist so far** — best convergence, best
year-confidence, validated at full scale, and its LOO/frac_flat shortfall relative to AZ4/AZ4b
needs to be weighed against those two models' own reliability problems (AZ4b's poor r-hat, and
the now-family-wide Pareto-k caution) rather than taken at face value. Not yet a final decision:
AZ4/AZ5 haven't been run at full scale, and K-fold remains the properly deferred tie-breaker.

### Deferred (explicit user instruction, not forgotten)
- Rigorous K-fold CV evaluation of the finalist(s) — bring back once there's a smaller
  finalist set; the Phase 6 first-pass PSIS-LOO comparison is directional only given the
  17-40% Pareto-k>0.7 rate found across every current family member.
- Fixing the flat `mu_area = D/n_years` baseline shape (original Step 2) —
  revisit after Phases 1b-2 to see how much of the "lumpy quiet-year" bad
  Pareto-k problem the D-band scale already closes without touching the
  mean assumption.

## Side thread (not part of this modelling plan, tracked separately)
The marimo trace-browser notebook (`notebooks/7.0-sd-trace_browser.py`)
isn't picking up new trace files automatically in the user's live session,
despite `mo.watch.directory`'s polling mechanism being verified correct by
reading marimo's source and confirmed via a fresh static export. Live
session (PID 2248 at time of writing) predates AZ1a.nc's creation by ~30
minutes and never picked it up — diagnosis pending a kernel restart to
determine whether this specific session got stuck vs. a deeper issue.
