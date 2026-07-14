# Multimodality diagnostic pipeline

A manually-run pipeline (`housing_projections.multimodality`, exposed via
`housing-projections check-multimodality`) for one specific, recurring question from the AZ1
family's ESS/r-hat investigation (`az-ess-diagnosis.md`, `ess-rhat-diagnostic-guide.md` Pattern
2, `multimodality-characterization-guide.md`): **when a model has bad r-hat/ESS, how much of
that is genuine, expected multimodality (not a defect) versus a real, fixable sampling problem
versus something that still needs a manual deep-dive — without re-running a bespoke
investigation script for every model, every time?**

Originally built for one specific shape of that question — a per-area hierarchical lag-category
simplex (`*_lambda_weights`) — and later generalized to answer it for ANY flagged
variable/cell in a model's diagnostic scope (any scalar or vector in `var_names`, not just a
simplex with known categories; see §1b/§2c). The two mechanisms differ (known categories +
per-source log-likelihood for `*_lambda_weights`; discovered chain-level clusters + total logp
for everything else) but report through the same five-plus-one categories and the same
raw/adjusted/best_case figures, so a model's per-area AND per-hyperparameter multimodality
picture come out of one combined report.

Run it whenever a model's `pixi run diagnose` output shows elevated r-hat/ESS (not for every
model — see §0 below).

## Table of contents

- [Why this exists, briefly](#why-this-exists-briefly)
- [§0 — Is this worth running?](#0-is-this-worth-running)
- [§1 — The five categories](#1-the-five-categories)
- [§1b — The sixth outcome: `not_multimodal` (scalars/vectors only)](#1b-the-sixth-outcome-not_multimodal-scalarsvectors-only)
- [§2 — The full workflow, step by step](#2-the-full-workflow-step-by-step)
- [§2a — Running it](#2a-running-it)
  - [Worked example (AZ1d, real data, this session)](#worked-example-az1d-real-data-this-session)
- [§2b — Why `check-multimodality` checks every lag var by default (AZ4b, real data)](#2b-why-check-multimodality-checks-every-lag-var-by-default-az4b-real-data)
- [§2c — Generalizing beyond `*_lambda_weights`: scalars and vectors (AZ4b, real data)](#2c-generalizing-beyond-_lambda_weights-scalars-and-vectors-az4b-real-data)
  - [Worked result: `lag_E_mu_logit`, resolved (AZ4b, real data)](#worked-result-lag_e_mu_logit-resolved-az4b-real-data)
  - [`--resolve` in practice: verification catches a real edge case, not just a formality](#--resolve-in-practice-verification-catches-a-real-edge-case-not-just-a-formality)
  - [Step 4 in detail: `diagnose --adjust-for-multimodality`](#step-4-in-detail-diagnose---adjust-for-multimodality)
- [§3 — What "within-chain ESS" is checking, and why it's a separate number](#3-what-within-chain-ess-is-checking-and-why-its-a-separate-number)
- [§4 — When the "adjusted" figures actually move](#4-when-the-adjusted-figures-actually-move)
- [§5 — Limitations, stated plainly](#5-limitations-stated-plainly)

## Why this exists, briefly

Diagnosing AZ1d's residual r-hat/ESS problems this session required, by hand, in sequence:
`hierarchical_mode_summary` (chain purity), a bespoke log-likelihood-gap script, a bespoke
within-chain switch-rate check (which caught that AZ1d's `E01035649` — reported via purity alone
as a clean "7 vs 1 chains" split — was actually every one of 8 chains visiting both categories
26-55 times across their run, not a real separation), and finally a bespoke informed-init
reseeding script to test which flagged areas were actually fixable. Each of those was a fresh,
one-off script. This module assembles all four into one reusable, tested pipeline, and adds the
one thing none of them did on their own: an automatic classification and an "adjusted"
diagnostics figure that excludes cells whose bad r-hat is *expected*, so the raw model
diagnostics table stops conflating "the sampler is unhealthy" with "the posterior genuinely
disagrees."

**Calibration caveat, upfront and not buried**: every threshold below (purity ≥0.95, switch
rate >0.02, gap <2/>10 nats) was derived empirically from the AZ1 family this session, not from
a universal statistical law. They are a documented starting point. Re-validate them, don't just
trust them, against a model with a materially different observation-noise scale, different
observations-per-area, or different lag structure.

## §0 — Is this worth running?

Cheap checks before reaching for the full pipeline (mirrors
`ess-rhat-diagnostic-guide.md` §2/§6, `multimodality-characterization-guide.md` §0):

1. Is the model's bad r-hat/ESS on a `*_lambda_weights` simplex, or on a scalar/vector in the
   model's own `var_names`? This pipeline now covers both (§1b/§2c) — but a discontinuity or
   funnel in a variable OUTSIDE that scope (e.g. `z`, `delta`, or anything else deliberately
   excluded from `var_names` for performance reasons — see `diagnostics_summary`'s own
   docstring) still isn't covered; go to `ess-rhat-diagnostic-guide.md`'s decision procedure
   for those.
2. Has `pixi run diagnose` already been run and shown elevated max r-hat / low min ESS? Don't
   run this speculatively on a model that's already converging cleanly.
3. Do you actually need the per-area breakdown, or does the consumer of this model only care
   about each area's *total* change? The census zero-sum constraint always pins that exactly —
   only *which year* gets credit is ever ambiguous. If nobody downstream needs the year-by-year
   split, "genuine ambiguity exists in a documented minority of areas" (one sentence, from a
   plain `hierarchical_mode_summary` call) may already be the right stopping point.

## §1 — The five categories

| Category | Signature | What it means | What to do |
|---|---|---|---|
| `hard_genuine` | chains cleanly separate (switch rate ~0), every non-best pure group tied with the best (gap <2 nats) | Real, irreducible epistemic ambiguity — the data can't distinguish two lag categories, no amount of sampling changes that | Report via `hierarchical_mode_summary`; don't try to fix |
| `stuck_fixable` | chains cleanly separate, every non-best pure group decisively worse (gap >10 nats) | Some chains are stuck in a mode that's actually worse — a sampling failure, not genuine ambiguity | `resolve_stuck_areas`: reseed and resample |
| `round_tripping` | switch rate high (>0.02) — chains cross between categories repeatedly, none is "stuck" | A shallow, overlapping ambiguity; the ordinary pooled posterior already reflects it reasonably | Report as-is; don't attempt to isolate/reseed "modes" that aren't really separate chain populations |
| `mixed` | switch rate low, but SOME non-best groups are tied with the best while at least one OTHER is decisively worse | A genuine tie between two explanations, plus an unrelated straggler chain stuck in a clearly worse third | **Not** auto-resolved — `nutpie`'s `init_mean` seeds the whole area for every chain, so "fixing" the straggler would also force the genuinely-tied chains toward one target, destroying real ambiguity worth keeping. Needs a human call. |
| `needs_review` | doesn't cleanly fit any of the above (fewer than 2 pure chain-groups, or a gap in the ambiguous 2-10 nat band) | The evidence doesn't clearly support any label | Manual inspection |

Full mechanism/derivation for each category, and the worked real examples (`E01002702` =
hard_genuine, `E01033711`/`E01035656` = stuck_fixable, `E01035649`/`E01035646` = round_tripping,
`E01035708` = mixed — a genuine tie between two lag categories plus one lone chain stuck in a
clearly worse third) are in `az-ess-diagnosis.md` and the `az1d-multimodal-deepdive` artifact.

## §1b — The sixth outcome: `not_multimodal` (scalars/vectors only)

`classify_multimodality`'s five categories all presuppose that a flagged area's chains are
disagreeing about something with a KNOWN shape — which of K discrete lag categories fits best.
A generic scalar or vector hyperparameter (`lag_E_mu_logit`, `sigma_plan`, `rho_P`, …) has no
such known shape, so `classify_scalar_multimodality` (§2c) has to DISCOVER whether chain-level
clusters exist at all before it can even ask which of the five categories applies. When it
finds fewer than two clusters, that's `not_multimodal`: the cell's bad r-hat/ESS is a REAL,
unexplained problem — slow mixing, a funnel, whatever — not multimodality of any kind.

| | |
|---|---|
| Signature | Fewer than 2 chain-level clusters discovered (see §2c) |
| What it means | The bad r-hat/ESS is NOT attributable to multimodality at all |
| What to do | `ess-rhat-diagnostic-guide.md`'s decision procedure, not this pipeline |

This can only occur on the generic (non-`*_lambda_weights`) path — a categorical simplex's
"modes" are given, not discovered, so it's structurally always multimodality-shaped once
flagged. `not_multimodal` cells are NEVER excluded from `adjusted`/`best_case` (excluding them
would misrepresent a real, still-open problem as "already accounted for"), and are counted
separately (`n_not_multimodal`) so they're never confused with `needs_review` ("we don't know
which explanation yet" — a fundamentally different status from "we're confident there ISN'T a
mode-separation explanation").

## §2 — The full workflow, step by step

This is a four-step loop between two commands, not two independent checks. Do them in order:

1. **`pixi run diagnose`** (plain, no flags). Surfaces which models have elevated max r-hat /
   low min ESS. This tells you *that* something looks bad, not *why* — bad r-hat can equally
   mean a genuine sampling failure or expected, benign multimodality.
2. **`housing-projections check-multimodality --models <Name>`**, for each model step 1
   flagged. Checks the model's full diagnostic scope (§0) — every `*_lambda_weights` simplex
   AND every scalar/vector in the model's own `var_names` — for anything genuinely outside
   that scope, go to `ess-rhat-diagnostic-guide.md` instead. It classifies every flagged
   area/cell into one of the six §1/§1b categories and reports THREE r-hat/ESS figures, all
   computed over that SAME combined scope: raw, "adjusted" (drops `hard_genuine`/
   `round_tripping` cells, whose bad r-hat is expected rather than broken — §4 explains when
   this will and won't visibly move), and "best case" (drops every flagged area/cell EXCEPT
   `not_multimodal` ones — a preview of the rest of the model once everything attributable to
   multimodality has been triaged; see the worked example below for why this is a different
   question from "adjusted", not a stricter version of it).
3. **`check-multimodality --models <Name> --resolve --lag-var <one variable>`** — optional, and
   only for models where step 2 found `stuck_fixable` areas. Unlike step 2's report,
   `--resolve` requires `--lag-var` to name EXACTLY ONE variable: resolving means resampling the
   model once, for one lag hierarchy at a time, so it can't default to "every lag var" the way
   the report itself does. Reseeds and resamples that one hierarchy's `stuck_fixable` subset,
   saving `{Name}_resolved.nc` alongside the original trace (which is left untouched). `mixed`
   areas are deliberately never auto-resolved this way (§5).
4. **`housing-projections diagnose --adjust-for-multimodality`** — run once you've done
   steps 2–3 for every model step 1 flagged and are satisfied with the per-area triage. This
   folds the same adjustment into the routine multi-model `diagnose` table instead of you
   eyeballing each model's `check-multimodality` output separately; see the dedicated
   subsection below for exactly how it differs from step 2's own adjusted figure.

Steps 2 and 4 both produce an "adjusted" r-hat/ESS, and it's easy to conflate them: step 2's
adjustment is per-model — one `check-multimodality` call at a time, useful while you're still
deciding how to triage each flagged area — while step 4's is the aggregate, run once triage is
done across every model step 1 flagged. Both now auto-detect every `*_lambda_weights` var a
trace has by default (see §2b for why that matters), so they're no longer scoped differently;
step 4 is *stricter* in a different sense (see "Two behavioural differences" below) — it
excludes more categories by default, not more variables.

## §2a — Running it

```bash
# classify only (fast, safe to run anytime a model's r-hat/ESS looks concerning)
housing-projections check-multimodality --models AZ1d

# also attempt the validated fix for the resolvable (stuck_fixable) subset — this
# RESAMPLES the model (slow), and saves the result as AZ1d_resolved.nc alongside the
# original trace, which is left untouched
housing-projections check-multimodality --models AZ1d --resolve --resolve-chains 16
```

By default this checks **every** `*_lambda_weights` variable the model's trace has (planning
and BEN independently, if both exist) PLUS every scalar/vector in that model's own `var_names`
(the same list `diagnose` itself uses — resolved automatically from the registered model
class, same as `diagnose`), and combines them into one report —
`housing_projections.multimodality.multimodality_report` under the hood. Pass `--lag-var
lag_P_lambda_weights,lag_E_lambda_weights` (comma-separated) to restrict the LAG-VAR side to a
subset (the scalar side isn't separately restrictable via a flag — it always follows the
model's own `var_names`); the matching log-likelihood variable for each lag var is derived
automatically by naming convention (`derive_loglik_var`: `lag_P_lambda_weights` → `P_like`), so
there's no `--loglik-var` flag to set by hand. Other flags: `--rhat-threshold` (default 1.01,
same convention as `diagnose`).

### Worked example (AZ1d, real data, this session)

```
Flagged areas: 16/200
  hard_genuine (irreducible tie, no fix):      1
  stuck_fixable (fixable via reseeding):       2
  round_tripping (shallow/overlapping, benign): 4
  mixed (tie + stuck straggler, needs a call): 1
  needs_review (doesn't cleanly classify):     8

Raw        max r-hat=1.7017  min ESS=12
Adjusted   max r-hat=1.7017  min ESS=12  (excludes hard_genuine/round_tripping cells)
Best case  max r-hat=1.0090  min ESS=612  (excludes ALL 16 flagged areas)
```

AZ1d has only one lag hierarchy (planning), so `--lag-var` defaulting to "every `*_lambda_weights`
var found" checks exactly the one variable this worked example already shows — nothing here
changes for a single-lag-var model. See the AZ4b example below (§2b) for where checking every lag
var by default actually changes the answer.

This reproduces, in one command, essentially the same structure that took a multi-script manual
investigation to establish earlier this session — `E01002702` as the clean hard_genuine tie,
`E01033711`/`E01035656` as the resolvable core, `E01035649`/`E01035646` correctly separated out
as round-tripping rather than lumped in with the "stuck" areas, and `E01035708` correctly
flagged as `mixed` (a genuine tie between two categories plus one lone stuck chain) rather than
mischaracterized as a simple 2-way split, which is what a plain chain-fraction read had
originally done for it.

Note here that the adjusted max r-hat/min ESS didn't move — that's a legitimate outcome, not a
bug: `stuck_fixable`/`mixed`/`needs_review` areas are deliberately *not* excluded from it (their
bad r-hat isn't yet established as "expected"), and in this run one of them happened to carry the
same extreme value as the excluded `hard_genuine` area. The adjusted figure only ever *removes*
values, it doesn't manufacture an improvement that isn't earned — see §4 for when to expect it
to actually move.

**"Best case" is a third, distinct figure, not a stricter version of "adjusted".** Where
`adjusted_max_rhat`/`min_ess` only excludes cells already established as benign
(`hard_genuine`/`round_tripping`), `best_case_max_rhat`/`min_ess` excludes **every** flagged
area — `stuck_fixable`, `mixed`, `needs_review`, resolved or not — regardless of whether its
status has actually been confirmed. That's deliberate: it answers a different question — not
"is there still a sampling problem right now" (that's what `adjusted` is for) but "once I've
triaged every one of these 16 areas — fixed the fixable ones, accepted the genuine ties, made
the manual calls — what does the rest of the model look like." Treat it as a preview, not a
verdict: it's honest about excluding areas nothing has actually proven fine, so don't quote it as
"the model's real r-hat" while `n_needs_deep_dive` is still above zero for that model — check the
per-area table (below) and work through it first. If a model's flagged areas are a small fraction
of the total (rare enough that excluding them can't hide a real, separate problem elsewhere), the
gap between `adjusted` and `best_case` mainly tells you how much of the model's *current* headline
r-hat/ESS is being dragged down by multimodality specifically, versus something else.

## §2b — Why `check-multimodality` checks every lag var by default (AZ4b, real data)

`best_case_max_rhat`/`min_ess` is *supposed* to answer one question precisely: "what would
r-hat/ESS look like if every flagged area were excluded?" For a model with two independent lag
hierarchies (planning and BEN, e.g. AZ4b's `lag_P_lambda_weights`/`lag_E_lambda_weights`), that
question is only actually answered if BOTH hierarchies' flagged areas are excluded — checking
only one and calling the result "best case" quietly answers a narrower question while claiming
to answer the broad one.

This module used to default `check-multimodality` to a single `--lag-var` (`lag_P_lambda_weights`),
while `diagnose --adjust-for-multimodality` auto-detected every `*_lambda_weights` var. Running
both against the same AZ4b trace surfaced the gap directly:

```
check-multimodality --models AZ4b                          -> best_case: r-hat=1.0096  ESS=1190
diagnose --models AZ4b --adjust-for-multimodality           -> best_case: r-hat=1.4594  ESS=16
```

Two separate problems compounded into that mismatch:

1. **Scope.** `check-multimodality` was silently only checking `lag_P_lambda_weights` (85
   flagged areas) and never looking at `lag_E_lambda_weights` at all (a further 182 flagged
   areas, including 3 `hard_genuine` ones with a real r-hat of 1.73). `diagnose` saw both.
2. **Contamination.** `diagnose`'s own best_case, at the time, additionally (and wrongly)
   included every scalar hyperparameter in the model's `var_names` unfiltered — and AZ4b's
   `lag_E_mu_logit` (the hierarchical mean-logit for E's lag-category prior, NOT a per-area
   cell) had its own real r-hat/ESS problem (1.4594 / 15.5) that no amount of area exclusion
   could ever touch.

Both are fixed now, not worked around (at this point in the story): `check-multimodality` calls
`multimodality.multimodality_report`, which auto-detects and combines every `*_lambda_weights`
var a trace has — the same auto-detection `adjusted_diagnostics_summary` already used — so both
commands check the identical set of lag vars. Separately, `best_case_max_rhat`/`min_ess` in
`adjusted_diagnostics_summary` was made scoped strictly to `*_lambda_weights` cells, never scalar
hyperparameters, matching what `adjusted_diagnostics_report`'s own best_case did. Both commands
agreed on AZ4b at this point (lag vars only, scalars not yet in scope on either side):

```
check-multimodality --models AZ4b                          -> best_case: r-hat=1.0098  ESS=867
diagnose --models AZ4b --adjust-for-multimodality           -> best_case: r-hat=1.0098  ESS=867
```

**This intermediate fix was NOT the end state — §2c below supersedes these exact numbers.**
Excluding scalars from `best_case` entirely was a way of not lying about scope, not a way of
actually covering the model's full diagnostic surface — `lag_E_mu_logit`'s own real problem
was still sitting there, just outside `best_case`'s reach rather than wrongly counted inside
it. §2c brings scalars INTO scope properly (classified, not just excluded-or-ignored), which
moves AZ4b's `best_case` again, this time for real: see its own worked result for the current
numbers.

**`lag_E_mu_logit`'s problem didn't go away — it's just no longer mislabeled, and no longer
invisible to this pipeline either.** At the time this was written, it was still visible only
via plain `diagnose`'s ordinary `max_rhat`/`min_ess`, and its cause was an open hypothesis
("plausibly caused by the per-area multimodality — a hierarchical mean being pulled around by
182 unstable per-area posteriors feeding into it — but that's a hypothesis, not confirmed").
§2c below closes that gap: `classify_scalar_multimodality` now checks `lag_E_mu_logit` (and
every other scalar in the model's `var_names`) directly, and the hypothesis turned out to be
**wrong** — see §2c's own worked result.

## §2c — Generalizing beyond `*_lambda_weights`: scalars and vectors (AZ4b, real data)

A `*_lambda_weights` simplex has a built-in advantage this pipeline originally leaned on
entirely: the "modes" (lag categories) are KNOWN in advance, so classification is just
"argmax which category each chain favours, then compare per-source log-likelihood between
groups." A scalar or vector hyperparameter (`lag_E_mu_logit`, `sigma_plan`, `rho_P`, …) has no
such known categories — classifying it requires DISCOVERING whether chains have split into
distinct clusters at all, before any of the existing switch-rate/purity/gap machinery can run.

Two new mechanisms make that possible, both deliberately validated against synthetic failure
cases before being trusted (see `_cluster_chain_means`'s own docstring in `multimodality.py`
for the exact cases checked, not just described):

- **Mode discovery — standardized-gap clustering on chain means, NOT KDE.** A first attempt at
  KDE peak-finding on each chain's own mean (reusing the same `_detect_modes` machinery already
  validated for `z`) was tried and rejected: scipy's default bandwidth over-smoothed an obvious
  cluster split into one lump; a hand-tuned smaller bandwidth that DID separate it also
  false-positived on pure unimodal noise AND still missed a minority 1-vs-N chain split. The
  adopted mechanism instead sorts chain means and starts a new cluster whenever the gap to the
  previous one exceeds `sigma_threshold` (default 3.0) times the pooled within-chain DRAW
  standard deviation — the same sigma-relative-to-noise principle `check_chain_agreement`
  already uses for its own trapped-chain check, just applied to means instead of per-draw logp.
- **The gap check — sigma-normalized total logp, not a named log-likelihood.** A generic
  scalar has no dedicated `P_like`/`E_like`-style log-likelihood group to compare between chain
  groups. `compute_logp_gap` uses `sample_stats.logp`/`.lp` instead (already available for any
  sampled model — the same value `check_chain_agreement` reads), with the gap expressed in
  units of pooled within-chain logp standard deviation rather than absolute nats — total model
  logp sums the density of every parameter and observation in the model, not just the ones
  relevant to the cell being classified, so the `*_lambda_weights` pipeline's fixed 2/10-nat
  tied/decisive thresholds would be meaningless applied to it directly (they're the SAME
  numeric thresholds here, just now measured in sigma units instead of nats — re-validate
  before trusting that reuse beyond what's been checked so far).

**Calibration caveat, exactly as blunt as the one at the top of this doc:** `sigma_threshold`
and the sigma-unit tied/decisive thresholds are checked against the synthetic cases above and
the one real worked example below — a documented starting point, not a proven-robust default.

### Worked result: `lag_E_mu_logit`, resolved (AZ4b, real data)

Running the generalized pipeline against AZ4b (`housing-projections check-multimodality
--models AZ4b`, now covering `var_names` by default) classified every one of its 11 flagged
scalar cells — including `lag_E_mu_logit[1]`, r-hat 1.4594, the exact cell §2b's speculative
hypothesis was about — as `not_multimodal`: fewer than 2 chain-level clusters were found at
all. **The hypothesis was wrong.** `lag_E_mu_logit`'s poor mixing is NOT a downstream symptom
of the per-area lag-category multimodality — it's a genuinely separate sampling problem (likely
slow mixing or a funnel-like geometry with only 8 chains), and belongs in
`ess-rhat-diagnostic-guide.md`'s decision procedure, not this one. Guessing ahead of running
the check would have kept treating it as "probably explained already" indefinitely.

With scalars now included in the SAME scope as `raw`, the two headline figures are finally the
honest before/after comparison this pipeline was always supposed to provide:

```
Raw        max r-hat=1.7267  min ESS=12   (lag vars + 11 scalars, unfiltered)
Best case  max r-hat=1.4594  min ESS=16   (same scope, every multimodal cell excluded)
```

`best_case` moved (1.73 → 1.46), but not all the way to "healthy" — because `lag_E_mu_logit`'s
`not_multimodal` cell is correctly still counted. That gap between "best case, multimodality
resolved" and "actually healthy" IS the finding: it's exactly the amount of AZ4b's r-hat/ESS
problem that this pipeline can't explain, told honestly rather than hidden inside a
same-scope-mismatched or scalar-blind number.

### `--resolve` in practice: verification catches a real edge case, not just a formality

Running `--resolve` on this same trace reseeded `E01033711` and `E01035656` with 16 chains and
verified afterward:

```
E01033711: 93.75% chain agreement -> resolved
E01035656: 75.00% chain agreement -> resolved
```

Two things worth knowing from this specific run, not just in the abstract:

1. **The default `resolved` threshold matters and was recalibrated from experience, not
   guessed.** `verify_resolution`'s `purity_threshold` defaults to 0.75, not 0.95 — with a
   realistic chain count (16), 0.95 effectively demands total unanimity (15/16 = 93.75% already
   falls short of it), stricter than this session's own established precedent for "resolved"
   (`E01033700` at 69%, `E01035656` at 81% in the original investigation were both reported as
   genuine successes). The raw `chain_agreement_frac` is always returned alongside the boolean
   specifically so this can be judged directly rather than trusting one cutoff blindly.
2. **`E01035656` reseeded toward the "wrong" category and the run corrected for it anyway** —
   `classify_multimodality`'s `best_category` for this area, computed from the 8-chain baseline
   trace, was category 1; `resolve_stuck_areas` seeded toward it; but 75% of the 16 reseeded
   chains drifted to category 2 regardless. This means the baseline classification's
   `best_category` for this one area wasn't fully robust (plausibly an 8-chain point estimate
   noisy enough to get the comparison backwards) — `init_mean` is a soft nudge, not a hard
   constraint, so chains can and do still find a genuinely better mode even when seeded
   elsewhere. This is exactly the kind of thing `verify_resolution` exists to catch: don't
   trust a seeded run "worked" just because it ran and produced a majority — check what that
   majority actually converged to.

### Step 4 in detail: `diagnose --adjust-for-multimodality`

This is step 4 from §2 — run it once every model §1's loop flagged has been through
`check-multimodality` (and `--resolve` where applicable):

```bash
housing-projections diagnose --adjust-for-multimodality
```

It auto-detects every `*_lambda_weights` var in each trace (same as step 2's own default now —
see §2b) and automatically picks up any `{model}_resolved.nc` already saved by
`check-multimodality --resolve` in `--traces-dir`.

Two behavioural differences from `check-multimodality`'s own adjusted figure, both deliberate:

1. **`--exclude-reviewed` (default on) is more aggressive than `check-multimodality`'s default.**
   By this point you've already seen the `mixed`/`needs_review`/still-unresolved-`stuck_fixable`
   areas in `check-multimodality`'s per-area table — nothing about them is hidden — so the
   headline max r-hat/min ESS also excludes them here, on top of the always-excluded
   `hard_genuine`/`round_tripping` cells. Pass `--no-exclude-reviewed` for the more conservative
   reading that only credits areas as PROVEN benign or PROVEN fixed.
2. **`n_needs_deep_dive` counts an unresolved `stuck_fixable` area as needing a look, even if
   `--resolve` was never run for that model.** `check-multimodality`'s own report assumes an
   un-attempted `stuck_fixable` area is "known and fixable, just not fixed yet" and doesn't
   count it; `diagnose --adjust-for-multimodality` is stricter, since by the time you're running
   this you're asking "is there anything left to do", not "do I understand the picture".

The `diagnose` table gains seven columns: `best_case_max_rhat`, `best_case_min_ess`,
`n_lambda_weights_vars`, `n_flagged_multimodal`, `n_resolved`, `n_needs_deep_dive`,
`n_not_multimodal` — plus a summary line under the usual r-hat/divergence/chain callouts. A
model with no `*_lambda_weights` var and no matching `var_names` entry is reported completely
unchanged from plain `diagnose` (0 in every new column); the same is true for a
`*_lambda_weights` var that has no matching `_like` entry in `trace.log_likelihood` to classify
against — its cells are still counted, just unadjusted, rather than silently dropped from the
aggregate.

`best_case_max_rhat`/`best_case_min_ess` are the SAME number `check-multimodality` reports
(§2a, §2b, §2c) — every flagged area/cell excluded across every `*_lambda_weights` var AND
every scalar in `var_names` (except `not_multimodal` ones), regardless of category or
`--exclude-reviewed` — computed once per model here instead of once per `check-multimodality
--models <Name>` call. This table's own `max_rhat`/`min_ess` (governed by `--exclude-reviewed`)
and `best_case_max_rhat`/`min_ess` are now computed over the IDENTICAL scope (§2c), so with the
default `--exclude-reviewed=True` the two are numerically equal *unless* a `not_multimodal`
scalar cell is present — that's the one case where the model still has a real, open problem
`best_case` correctly refuses to paper over (see §2c's own `lag_E_mu_logit` result). Check
`n_not_multimodal` first if the two disagree.

## §3 — What "within-chain ESS" is checking, and why it's a separate number

For every `hard_genuine`/`round_tripping` area, the report includes each chain's own ESS,
computed in isolation (not the usual cross-chain rank-normalized ESS). The distinction matters:
standard r-hat/ESS conflates "did this chain mix well on its own" with "do independent chains
agree with each other" — for a genuinely multimodal area, chains correctly landing in different
places isn't a mixing failure, so the ordinary cross-chain number is expected to look bad and
tells you nothing about whether the sampler itself is doing its job. Within-chain ESS answers
that second question directly (mirrors `ess-rhat-diagnostic-guide.md` §2 step 1's per-chain
autocorrelation check, generalized into a reusable number).

## §4 — When the "adjusted" figures actually move

The adjusted max r-hat/min ESS will differ from the raw figures whenever the model's *worst*
r-hat/ESS values are concentrated in `hard_genuine`/`round_tripping` cells specifically — i.e.
when the model's real, fixable problems (if any) are milder than its genuine, irreducible
ambiguity. If a `stuck_fixable` or `mixed` or `needs_review` area happens to carry an equally or
more extreme value (as in the AZ1d example above, where a stuck area shared the same enormous
value as the genuine tie), the adjusted figure won't move — and shouldn't, since that area's
problem hasn't been established as benign yet. Don't read "the adjusted number didn't change" as
"the exclusion logic didn't work"; check `n_needs_deep_dive` and the per-area category table for
the real signal instead.

## §5 — Limitations, stated plainly

- **Thresholds are heuristics from one model family.** Re-validate before trusting them
  elsewhere (see the calibration caveat at the top).
- **`resolve_stuck_areas` only ever touches `stuck_fixable` `*_lambda_weights` areas — never
  scalars.** There is no automated resolution path for a generic scalar's `stuck_fixable`
  cell (no equivalent of `raw_offset` to seed); `mixed` areas of either kind can't be safely
  auto-resolved either, because `init_mean` seeds the whole area for every chain at once —
  there's no way to nudge only the stuck straggler while leaving the genuinely-tied chains
  alone.
- **Purity/switch-rate checks (and the generic mode-discovery/gap mechanism, §2c) haven't been
  stress-tested at scale.** They're validated against this session's own worked examples (real
  for `*_lambda_weights`; synthetic PLUS one real worked result — `lag_E_mu_logit` — for the
  generic path), not proven robust across arbitrary future models.
- **`classify_scalar_multimodality` checks each cell of a vector variable independently.** A
  2-element `lag_P_mu_logit` gets classified element-by-element, not as one joint 2-D
  question — unlike `*_lambda_weights`, where an area's whole category simplex is judged
  together. This loses some potential joint-explanatory power (e.g. if two logits of the same
  source are correlated across chains) in exchange for a much simpler, generic implementation.
- **A classifier output is a screening aid, not a verdict.** The whole reason this pipeline
  exists is that a shallow, single-signal read (chain purity alone) mischaracterized
  `E01035649` and would have quietly gotten `E01035708` wrong too. Treat `needs_review` results,
  and any `resolved` boolean sitting near its threshold, as an instruction to look closer — not
  as license to stop looking.
