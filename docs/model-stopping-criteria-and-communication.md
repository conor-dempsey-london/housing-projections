# Model stopping criteria and stakeholder communication

**Purpose**: answer two linked questions for this project — (1) at what point do we say a
dwelling-change model is "good enough" to stop iterating on, and (2) how do we actually
produce and explain the (uncertain) year-by-year area-level estimates to stakeholders who
are not statisticians. Written as Task 3 of `docs/model-finalization-work-plan.md`, after
Task 1 (full-scale AZ3 characterization) and Task 2 (K-fold CV of AZ3 vs AZ0a vs M5) —
both feed the argument here directly.

## Table of contents

- [1. What "good enough" means for a Bayesian model, in general](#1-what-good-enough-means-for-a-bayesian-model-in-general)
- [2. Concrete thresholds for this project](#2-concrete-thresholds-for-this-project)
  - [2a. Why per-cell z r-hat is treated differently from every other convergence metric](#2a-why-per-cell-z-r-hat-is-treated-differently-from-every-other-convergence-metric)
- [3. Applying this to AZ3 specifically](#3-applying-this-to-az3-specifically)
- [4. Producing and communicating year-by-year estimates](#4-producing-and-communicating-year-by-year-estimates)
  - [Tier 1 — Confident (majority of area-years)](#tier-1-confident-majority-of-area-years)
  - [Tier 2 — Ambiguous, but characterizably so](#tier-2-ambiguous-but-characterizably-so)
  - [Tier 3 — Genuinely diffuse / low-data](#tier-3-genuinely-diffuse-low-data)
  - [Practical reporting output](#practical-reporting-output)
  - [What NOT to do](#what-not-to-do)
- [5. Decision: is AZ3 good enough to stop iterating on?](#5-decision-is-az3-good-enough-to-stop-iterating-on)

## 1. What "good enough" means for a Bayesian model, in general

Standard Bayesian model-development practice (see e.g. this repo's own
`ess-rhat-diagnostic-guide.md`/`multimodality-characterization-guide.md`, and the
`pymc-modeling`/`model-evaluation` skill references) treats "is this model done" as several
independent questions, each necessary but none individually sufficient:

1. **Does it sample correctly?** (Convergence — a precondition, not a merit.) r-hat, ESS,
   divergences all pass. A model that fails this tells you nothing trustworthy about any of
   the questions below — convergence failure isn't evidence of model quality one way or the
   other, it's a sampling-quality gate that has to pass before the other checks mean anything.
2. **Is it internally honest about its own uncertainty?** (Calibration.) Does a nominal 90%
   credible interval actually contain the true value ~90% of the time? Checked in-sample via
   posterior-predictive coverage (already in `diagnostics.py`'s `_check_calibration`) and,
   more rigorously, out-of-sample via K-fold-refit calibration (Task 2) or LOO-PIT.
   In-sample calibration can look good on an overfit model; out-of-sample calibration cannot
   be gamed the same way.
3. **Does it predict better than simpler alternatives, by enough to matter?** (Predictive
   accuracy / parsimony.) K-fold or LOO ELPD comparison, with the standard decision rule:
   `|elpd_diff| < ~4` (or `|elpd_diff / dse| < 2`) means the models are practically
   indistinguishable — prefer the simpler one. A model only earns its added complexity if it
   clears this bar.
4. **Does it solve the actual problem, not just the statistical proxy for it?** (Domain
   fitness.) This project's stakeholder requirement — attributing large P/E observations to
   specific years/areas, not averaging them away — is not automatically implied by 1-3. A
   model can converge cleanly, be well-calibrated, and beat simpler alternatives on ELPD while
   still smoothing away exactly the spikes stakeholders care about (this is literally how
   AZ0a's own failure mode was first found — see `az-family-work-plan.md`'s ground rules).
   `plot_spike_tracking_examples` and `frac_flat_despite_active` exist specifically to check
   this axis, and neither substitutes for 1-3.
5. **Are the returns to further iteration still worth the cost?** (Diminishing returns.) Not
   a statistical test — a judgement call, but one that can be made with evidence: has the
   AZ-family's own iteration history (Phase 0 through Phase 6) shown each new piece buying a
   shrinking marginal improvement, or trading one metric's gain for another's loss (as AZ4/AZ5
   did — better `frac_flat_despite_active` at the cost of worse year-allocation confidence)?

A model is "good enough to ship" when 1-4 all pass and 5 suggests further work would trade
one axis for another rather than improve all of them together — not when any single number
crosses a threshold in isolation.

## 2. Concrete thresholds for this project

Tied to diagnostics already built in this codebase, not invented fresh:

| Axis | Metric | Threshold | Where computed |
|---|---|---|---|
| Convergence | max r-hat (named scalars) | < 1.01 | `diagnostics.diagnostics_summary` |
| Convergence | min ESS (bulk & tail) | > 400 | `diagnostics.diagnostics_summary` |
| Convergence | divergences | 0 (or < 0.1%, scattered) | `diagnostics._check_divergences` |
| Convergence (z-specific) | per-cell z r-hat elevated | expected & reportable via `z_identifiability_summary`/`hierarchical_mode_summary`, NOT required to hit 1.01 — see §2a | `diagnostics.z_identifiability_summary` |
| Calibration | posterior-predictive coverage (90% nominal) | 0.85-0.95 | `diagnostics._check_calibration` |
| Calibration (out-of-sample) | K-fold-refit PIT histogram | roughly uniform, no strong U/inverted-U/skew | Task 2's `pit_records.csv` |
| Predictive accuracy | K-fold ELPD vs simpler alternative | `|elpd_diff| > 4` AND `|elpd_diff/dse| > 2` to justify added complexity | Task 2 |
| Domain fitness | `frac_flat_despite_active` | lower is better; judge relative to alternatives, not an absolute cutoff (AZ0a's 200-area baseline: 11.5%; AZ3's: 3.5%/4.7% at 200/4987-area scale) | `diagnostics.z_flatness_summary` |
| Domain fitness | spike-tracking plot | qualitative — every reference area (`REFERENCE_AREAS`) tracked plausibly, no new spurious dips/spikes introduced | `plots.plot_spike_tracking_examples` |
| Honesty about residual ambiguity | fraction of areas with ≥1 low-confidence year | report as a first-class number, not hidden — Task 1: 23.3% of areas at full scale | `diagnostics.z_identifiability_summary` |

### 2a. Why per-cell z r-hat is treated differently from every other convergence metric

This project's own diagnostic pipeline (`docs/ess-rhat-diagnostic-guide.md`,
`multimodality-diagnostic-pipeline.md`) already establishes that elevated r-hat/low ESS on
individual `z[area, year]` cells is frequently genuine, irreducible epistemic ambiguity (which
YEAR absorbed a change), not a sampling defect — the census constraint pins the area's TOTAL
change reliably even when its year-by-year breakdown is genuinely underdetermined by sparse
per-year data. Forcing r-hat → 1 on these cells would mean suppressing real uncertainty, not
fixing a bug (see AZ1b's documented case in `az-family-work-plan.md`). The stopping criterion
here is therefore: named scalar hyperparameters must hit the standard bar (1.01/400/0); z's
per-cell r-hat is CHARACTERIZED (via `z_identifiability_summary`, and
`check-multimodality`/`detect_z_multimodality` where relevant) and REPORTED, not forced to
converge — see §4 below for how this gets communicated rather than hidden.

## 3. Applying this to AZ3 specifically

*(Task 1's full-scale findings; Task 2's comparison numbers below)*

- Convergence: named scalars converge extremely tightly at full scale (`sigma_plan`,
  `sigma_ben`, `rho_P`, `rho_E`, `sigma_noise_P`, `sigma_noise_E` — posterior sd 0.0003-0.02).
  Zero divergences. Per-cell z r-hat reaches 2.20 for genuinely ambiguous areas — expected,
  not a defect (§2a).
- Calibration: plan/BEN 90% coverage 0.921/0.903 at full scale — inside the 0.85-0.95 band.
- Domain fitness: `frac_flat_despite_active` 4.65% at full scale (vs AZ0a's 200-area baseline
  of 11.5%) — the core pathology this whole model family exists to avoid is substantially
  reduced. 42.8% of areas have ≥1 genuinely multimodal `z` year — a real, sizeable minority
  where the year-by-year breakdown should be reported as multi-scenario, not a point estimate
  (§4).
- Predictive accuracy / parsimony (Task 2): **AZ3 wins clearly on genuine out-of-sample
  K-fold CV** — elpd -15077.0 (se 135.8) vs AZ0a's -16485.5 (se 152.4, diff/se=-6.9) and M5's
  -17156.9 (se 151.5, diff/se=-10.2). Both differences are far past the `|diff/se| > 2` bar.
  This *inverts* the earlier PSIS-LOO-based ranking (which had M5 ahead of AZ0a) — exactly
  the failure mode `model-evaluation-methods.md` flagged as a live risk given every AZ-family
  member's 17-40% bad-Pareto-k rate. K-fold, not PSIS-LOO, is the trustworthy ranking here.
  Held-out calibration agrees rather than contradicts: AZ3's mean LOO-PIT (0.450/0.460 for
  P/E) is the best-centered of the three, M5's (0.370/0.423) the most skewed — the
  best-predicting model is also the best-calibrated one, a mutually reinforcing result rather
  than one metric overriding another.

## 4. Producing and communicating year-by-year estimates

The project's core deliverable ([[project_goal_year_by_year_inference]]: the intercensal
total is already trusted, the annual pattern is the real ask) means every area/year cell
needs a reported number — but not every cell has the same *kind* of uncertainty behind it,
and treating them identically would misrepresent the ones that are genuinely ambiguous.
Three-tier reporting scheme, built entirely from existing tooling:

### Tier 1 — Confident (majority of area-years)
Point estimate = posterior mean of `z[area, year]`, with a 90% credible interval. Reportable
as a single number + band, the standard Bayesian summary. Identify via
`z_identifiability_summary` (`confident == True`) and `detect_z_multimodality`
(`n_modes == 1`).

### Tier 2 — Ambiguous, but characterizably so
For areas/years where `z_identifiability_summary` flags low year-confidence AND
`detect_z_multimodality`/`plot_z_area_modes` confirms genuine multimodality (not a diffuse
unimodal spread mistaken for two scenarios — see the concentration-check safeguard built into
`plot_z_area_modes` after the AZ3 Phase 3 false-positive lesson): report as **N labelled
scenarios with relative posterior mass**, e.g. "53% chance the change concentrated in 2016;
47% chance it concentrated in 2019" — not a misleading single mean sitting in the low-density
valley between two real modes (the exact AZ3 E01002702 lesson). `hierarchical_mode_summary`
provides the same per-chain-based mass estimate for hierarchical lag-weight cases.

### Tier 3 — Genuinely diffuse / low-data
Areas with near-zero P/E activity across the whole decade (e.g. E01035709's 9/10
all-zero years) where the year-by-year breakdown is close to exchangeable — no real
information distinguishes one year from another. Report as: "total change of D over the
decade, no informative year-level breakdown available" — explicitly, not as a confident-
looking but spurious 2-scenario split (the corrected failure mode from
`plot_z_area_modes`' own concentration-check fix).

### Practical reporting output
- Area-level: the existing `.{model}.identifiability.csv` (per `report`'s CLI docs) already
  carries the per-area confidence tier; extend its consumption (dashboards, stakeholder
  exports) to branch on tier rather than always rendering a single point-and-CI line.
- Spike-tracking plots (`plot_spike_tracking_examples`) remain the qualitative complement —
  useful for spot-checking specific areas stakeholders ask about, not a replacement for the
  systematic tier classification above.
- Aggregate honesty: always report the TOTAL-across-London number (or borough-level total)
  alongside any area-level breakdown — this project currently treats the census figure itself
  as exact (a simplifying modelling choice for this round, not a claim that the census has no
  error of its own; revisitable later), and *given that choice* the model adds no further
  uncertainty of its own to the borough/London total, unlike the year allocation. Stakeholders
  should see both numbers side by side so the area-level uncertainty doesn't read as uncertainty
  the model is introducing into the total, which it isn't — while still being told, at least
  once, that the total's own exactness is itself an assumption rather than a proven fact.

### What NOT to do
- Don't publish a single point estimate for a Tier 2/3 area/year without its uncertainty
  qualifier — this is the specific, already-documented failure mode (AZ3's E01002702 mean
  sitting in a low-density valley between two real explanations).
- Don't force r-hat-driven "fixes" onto genuinely multimodal z cells to make the report look
  cleaner (§2a) — report the ambiguity, don't suppress it.
- Don't present `frac_flat_despite_active`/coverage/ELPD numbers to stakeholders directly —
  translate them into the tiered practical statement above; the raw diagnostic numbers are
  for the modelling team's own stopping-criterion judgement (§1-3), not the stakeholder-facing
  artifact.

## 5. Decision: is AZ3 good enough to stop iterating on?

Applying the five-axis checklist from §1 directly:

1. **Samples correctly?** Yes. Named scalars converge tightly at full scale (posterior sd
   0.0003-0.02), zero divergences. Per-cell z r-hat elevation is understood and expected
   (§2a), not an open convergence question.
2. **Internally honest about its own uncertainty?** Yes, on both axes checked. In-sample
   coverage (0.921/0.903 plan/BEN at full scale) sits inside the healthy 0.85-0.95 band.
   Out-of-sample K-fold-refit calibration (Task 2) is the best-centered of the three models
   compared, not just adequate in isolation.
3. **Predicts better than simpler alternatives, by enough to matter?** Yes, decisively.
   AZ3 beats both AZ0a (its own simpler parent) and M5 (the best M-class alternative) by
   `|diff/se|` of 6.9 and 10.2 respectively on genuine K-fold CV — nowhere near the
   practically-indistinguishable zone (`|elpd_diff| < 4` / `|diff/se| < 2`).
4. **Solves the actual domain problem?** Yes, on the metric that matters most to
   stakeholders: `frac_flat_despite_active` at full scale (4.65%) is dramatically better than
   the AZ0a baseline (11.5% at 200-area scale) — the core pathology this whole model family
   exists to avoid is substantially, not marginally, reduced. Spike-tracking checks
   (`az-family-work-plan.md` Phase 3) already confirmed this isn't just an aggregate-metric
   artifact — specific known spike cases (E01033711, E01033491, E01001774) are tracked/
   flagged correctly.
5. **Diminishing returns from further iteration?** The AZ-family's own history says yes:
   AZ4/AZ4b/AZ5 (combining AZ3 with additional pieces) bought a marginal, noise-level
   `frac_flat_despite_active` improvement at a clear cost elsewhere (worse r-hat/ESS, worse
   year-allocation confidence) — the pattern this whole round has repeatedly found (AZ1b,
   AZ2's own history) is that the simplest construction that directly targets the diagnosed
   need wins, and combining validated pieces has consistently cost more than it bought. There
   is no evidence a further increment would improve every axis simultaneously rather than
   trade one for another.

**Decision: AZ3 clears all five criteria and is the recommended model to stop iterating on
and move toward production reporting.** Two honest caveats to carry forward into the
stakeholder-facing artifact (§4), not reasons to keep iterating on the model itself:

- 23.3% of areas have at least one low-confidence year, and 42.8% of areas have at least one
  genuinely multimodal year — both are properties of the DATA (sparse per-year signal, real
  ambiguity about which year absorbed a change), not defects in AZ3 specifically, and both
  are exactly what the tiered reporting scheme in §4 is built to communicate honestly rather
  than paper over.
- The K-fold comparison (Task 2) covers AZ3 vs its own baseline and the best simpler
  M-family alternative on the 200-area development sample — it does not re-litigate AZ4/AZ4b/
  AZ5 (the more complex combined models), which were already set aside on r-hat/year-
  confidence grounds in `az-family-work-plan.md` Phase 6 rather than on K-fold evidence. If
  those are ever revisited, K-fold (not PSIS-LOO) should be the tie-breaker, per the
  discrepancy this round already found between the two methods.
