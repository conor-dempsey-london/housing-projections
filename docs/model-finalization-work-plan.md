# Model finalization work plan — full-dataset characterization, CV, stopping criterion

**Status doc for this round of work.** Like `az-family-work-plan.md`, this is a live
checklist, not a finished record — check back here at the start of each task, update it
immediately when a task's status changes, and fold durable findings into
`model-progression-notes.md` once this round concludes. Three related tasks, described
below, run in the order given for the dependency reasons stated in each task's rationale.

## Table of contents

- [Context this plan builds on](#context-this-plan-builds-on)
- [Ordering and rationale](#ordering-and-rationale)
- [Task 1 — AZ3 full-dataset characterization](#task-1-az3-full-dataset-characterization)
- [Task 2 — Proper CV: AZ3 vs AZ0a vs M5](#task-2-proper-cv-az3-vs-az0a-vs-m5)
- [Task 3 — Stopping criterion + stakeholder communication method doc](#task-3-stopping-criterion-stakeholder-communication-method-doc)
- [Ground rules for this round](#ground-rules-for-this-round)

## Context this plan builds on

- `AZ3` (floored noise/outlier mixture likelihood on top of `AZ0a`) is the strongest
  evidence-backed finalist from the AZ-family round (`az-family-work-plan.md`, Phase 6):
  best convergence and year-allocation confidence among the real candidates, validated at
  full scale (4987 areas, all London LSOAs minus 7 hard outliers) — trace saved at
  `results/traces_full/AZ3.nc` (24 GB). `AZ4`/`AZ4b`/`AZ5` beat it on a first-pass PSIS-LOO
  but carry worse r-hat/ESS or an undiagnosed convergence problem, and every family member
  showed 17-40% Pareto-k > 0.7 on the joint P+E likelihood — the existing LOO comparison is
  directional only, not a finished answer.
- `model-evaluation-methods.md` is the standing plan for a "real" comparison once there's a
  finalist worth evaluating carefully (grouped K-fold CV, held-out calibration, and a
  census-anchored held-out-decade check) — written but not yet built. `compute_model_comparison`'s
  `P_like`-only bug and its xarray dim-mismatch cross-join bug were already fixed in Phase 6.
- Best M-class model per the existing (P+E, PSIS-LOO, 200-area) comparison is **M5**
  (elpd -17100, clearly behind AZ3's -16954 at dse=83) — this is the M-family comparator for
  Task 2, not a new search.
- Cached comparison lives in `results/traces/comparison.csv` / `comparison_meta.json`.

## Ordering and rationale

**Task 1 → Task 2 → Task 3.**

- Task 1 needs no new sampling (the full-scale AZ3 trace already exists) and its
  spatial/multimodality findings should inform how folds are constructed in Task 2 — AZ3 has
  neither a temporal-lag mechanism nor a borough hierarchy, so `model-evaluation-methods.md`'s
  default heuristic (leave-year-out / leave-borough-out) doesn't map onto it cleanly; better
  to look at where the real ambiguity/correlation structure lives before choosing.
- Task 2 produces the load-bearing evidence Task 3 needs — "AZ3 beats the alternatives by X,
  defensibly" requires the K-fold/calibration numbers, not the still-directional PSIS-LOO pass.
- Task 3 is the synthesis: convergence + comparison + calibration + spike-tracking together
  become the stopping-criterion argument and the stakeholder communication plan.

---

## Task 1 — AZ3 full-dataset characterization

**Status: DONE.**

Built `scripts/full_dataset_characterization.py` (heavy, trace-reading pass) +
`results/artifacts/az3_full_characterization/build_report.py` (CSV/PNG → HTML, no trace
access — the reproduction script) + `docs/az3-full-dataset-report-method.md` (the reusable
method write-up). Ran against the full 4987-area `results/traces_full/AZ3.nc`:

- **Convergence/behaviour**: `frac_flat_despite_active` 4.65% (vs the 200-area dev sample's
  3.5%, still far better than AZ0a's 200-area baseline of 11.5%); plan/BEN 90% coverage
  0.921/0.903; census constraint satisfied to ~1e-13 (exact, as designed). Named scalar
  hyperparameters (`sigma_plan`, `sigma_ben`, `rho_P`, `rho_E`, `sigma_noise_P`,
  `sigma_noise_E`) all converge extremely tightly (posterior sd 0.0003-0.02) — rock solid at
  full scale. Per-cell `z` r-hat is a different story: max 2.20, and 23.3% of areas have at
  least one low-confidence year — expected and already-documented (Phase 6's own
  "low-year-confidence 23.3%" figure), not a new problem; it reflects genuine per-area
  year-allocation ambiguity, not a sampling failure of the named parameters.
- **Multimodality at scale**: 42.8% of areas have at least one genuinely multimodal `z` year
  (12.6% of all area-years), using a thinned (every 4th draw) KDE scan for tractability at
  49,870 cells — see the method doc's documented resolution/speed tradeoff.
- **Spatial**: Moran's I on BEN residuals is small but significant (I=0.108, p=0.001) —
  some real spatial clustering in BEN under/over-prediction; planning residuals show
  negligible spatial structure (I=-0.014, p=0.04).
- Deliverables live in `results/artifacts/az3_full_characterization/`: `area_summary.csv`,
  `borough_summary.csv`, `multimodal_cells.csv`, `morans_i_resp_noise_by_year.csv`,
  `example_areas.csv`/`example_areas_timeseries.csv`, `scalar_summary.json`, 16 pre-rendered
  mode-decomposition PNGs for deep-dive example areas, and `report.html`.

**Goal**: a full characterization artifact of the AZ3 full-dataset run, plus a reusable
method doc so any future finalist can get the same report cheaply.

Steps:
1. Load `results/traces_full/AZ3.nc` (24 GB — needs chunked/subset access, not a naive
   in-memory load; document the approach used in the method doc).
2. Summary statistics at full scale (4987 areas): `frac_flat_despite_active`, `rho_P`/`rho_E`,
   `resp_noise`, posterior-predictive coverage, broken out by borough.
3. Spatial analysis: map `resp_noise` / flatness / multimodality onto the GDF, Moran's I on
   residuals (`spatial.py`), look for spatial clustering of ambiguity.
4. Multimodality at scale: run `detect_z_multimodality` / `check-multimodality` across all
   ~50k LSOA-years (not just the 200-area development sample used throughout the AZ round),
   flag `hard_genuine` cases, quantify prevalence the way Phase 3 did but at full scale.
5. Contrasting deep-dives: select a handful of areas spanning confident / genuinely-multimodal
   / heavily-noise-flagged / boundary cases (parallel to `REFERENCE_AREAS`, chosen from the
   full run rather than reusing the 200-area set unmodified).
6. Deliverables:
   - Full characterization artifact/report (HTML, reusing `html_report.py` patterns).
   - **CSV deliverables** (not just an HTML report): every summary table and per-area/per-cell
     result computed for the report (borough-level summary stats, per-area spatial/Moran's I
     inputs, per-cell multimodality classification, the deep-dive example area data) saved as
     a set of CSVs alongside the report.
   - **A reproduction script** that reads those CSVs and regenerates the full artifact/report
     without re-loading or re-scanning the 24 GB trace — the trace is read exactly once, by
     the analysis script; every subsequent regeneration (e.g. after a plotting tweak) runs off
     the CSVs.
   - Storage location: `results/artifacts/az3_full_characterization/` — the CSVs, the report
     HTML, and the reproduction script live together in one directory, separate from the
     scratch/one-off plots already in `results/scratch/` and `results/artifacts/`.
   - `docs/az3-full-dataset-report-method.md` — the reusable methodology for producing this
     report (and its CSVs + reproduction script) on any future full-scale trace.

---

## Task 2 — Proper CV: AZ3 vs AZ0a vs M5

**Status: DONE.**

**Result — a clean, defensible ranking that INVERTS the earlier PSIS-LOO ordering for
AZ0a/M5**:

| rank | model | K-fold elpd | se | elpd_diff vs AZ3 | diff / combined-se |
|---|---|---|---|---|---|
| 0 | **AZ3** | **-15077.0** | 135.8 | 0 | — |
| 1 | AZ0a | -16485.5 | 152.4 | -1408.5 | -6.9 |
| 2 | M5 | -17156.9 | 151.5 | -2079.8 | -10.2 |

Both differences are far past the `|diff/se| > 2` significance bar — AZ3 beats AZ0a beats
M5, unambiguously, on genuine out-of-sample refits (not PSIS approximation). This directly
contradicts the first-pass PSIS-LOO ranking in `az-family-work-plan.md` Phase 6 (which had
M5 ahead of AZ0a among the M-family, and both behind AZ4/AZ4b/AZ5) — exactly the outcome
`model-evaluation-methods.md` flagged as a live risk given every family member's 17-40%
bad-Pareto-k rate. **K-fold, not PSIS-LOO, should be treated as the trustworthy ranking for
this project going forward.**

Held-out calibration (mean LOO-PIT across all held-out P/E cells, computed alongside each
fold's log-likelihood — 0.5 is perfectly calibrated):

| model | mean PIT (P) | mean PIT (E) | read |
|---|---|---|---|
| AZ3 | 0.450 | 0.460 | best-centered of the three |
| AZ0a | 0.428 | 0.440 | mild low skew |
| M5 | 0.370 | 0.423 | most skewed, especially on P |

Calibration and predictive accuracy agree: AZ3 is both the best-predicting and the
best-calibrated model out-of-sample; M5 is worst on both axes. A coherent, mutually
reinforcing result, not one metric overriding a contradicting one.

**Engineering summary** (`scripts/kfold_comparison.py`): built a hand-rolled
`SamplingWrapper` rather than relying on the trace's own `log_likelihood` group, because none
of the three models needs the held-out area physically "in" a refit to score it correctly —
AZ0a/AZ3's z-prior (`_build_zero_sum_z_prior`) is a fixed-form, per-area-independent
ZeroSumNormal fully determined by that area's own census `D` (no fitted dependence at all),
and M5's Normal-prior-plus-soft-census-constraint construction is exactly Gaussian-conjugate
(verified numerically against brute-force Gaussian conditioning before use) — so a held-out
area's predictive distribution is analytically recoverable from the model's own fixed prior
formula plus the refit's posterior draws of shared hyperparameters (`sigma_plan`,
`sigma_ben`, `rho_P/E`, `sigma_noise_P/E`, `sigma_slab`, `lambda_weights`, `alpha_spatial`).
M5's spatial misallocation term (Queen-contiguity smearing across ALL 200 areas, not just the
training subset) needed the most care: held-out areas' geometric neighbours that were
themselves held out in the same fold get their own freshly-drawn `z_new` too, so the smearing
step always has a value for every neighbour regardless of fold membership.

**A real bug caught and fixed before trusting any result**: `arviz_stats.loo_kfold`'s own
`group_by` fold-splitting path (`_kfold_split_grouped`) calls `np.random.default_rng()` with
no seed — running each model as a separate process (for parallelism) would have given each
model a *different* random partition of held-out areas, silently confounding the comparison
(elpd differences would partly reflect "which areas each model happened to lose," not model
quality) and making the whole run irreproducible. Fixed by building one shared, seeded
area→fold assignment (`make_shared_area_folds`) and passing it via `folds=` instead of
`group_by=` — deterministic given `(n_areas, k, fold_seed)`, so the three separate model
invocations still scored on an identical partition. Caught by inspecting the library's fold-
assignment source before trusting the first (already-running) attempt, which was stopped and
restarted rather than left to produce a subtly-invalid comparison.

**Compute**: K=10, 600 draws/500 tune/4 chains per fold, 3 models run as separate concurrent
background processes. AZ0a: 198s total (~20s/fold). M5: 419s total (~42s/fold). AZ3: 1638s
total (~164s/fold) — its noise-mixture geometry samples markedly slower per fold, consistent
with its own documented ESS/funnel-geometry history (`az-family-work-plan.md` Phase 3), not a
new problem.

**Result storage**: `results/artifacts/kfold_comparison/{model}/kfold_summary.json`
(elpd/se/compute time), `elpd_i.csv` (per-held-out-cell elpd, area/year-labelled),
`pit_records.csv` (per-cell PIT for both P and E); `results/artifacts/kfold_comparison/
comparison.csv` combines all three models' summaries. Moved here from the script's original
default (`results/traces/kfold/`) — that location is gitignored/regenerable-cache territory
(same treatment as `results/traces/comparison.csv`, the existing PSIS-LOO cache, also never
committed), and buries a meaningful, lightweight analysis result inside a directory whose
purpose is holding multi-GB trace files. `results/artifacts/` is this project's existing
convention for discoverable, self-contained analysis outputs (mirrors Task 1's
`az3_full_characterization/`) — `kfold_comparison/` is a direct sibling.
Individual fold refits (posteriors) are NOT persisted to disk — at 30 refits this would be a
large amount of low-reuse storage, and the run is fully reproducible from
`--k`/`--fold-seed`/`--draws`/`--tune`/`--chains` if a specific fold's posterior is ever
needed again (a deliberate scope simplification from the original plan below, not an
oversight).

**Goal**: replace the directional PSIS-LOO comparison with a real, defensible comparison
between the current finalist (AZ3), the AZ-family baseline (AZ0a), and the best M-class model
(M5), per the plan already laid out in `model-evaluation-methods.md`.

Steps:
1. Build the `loo_kfold` `SamplingWrapper` (signature: `sel_observations`, `sample`,
   `get_inference_data`, `log_likelihood__i`) — reuse the moment-matching wrapper's
   value-var/transform handling as a reference, per that doc's own note that it's comparable
   engineering effort.
2. Decide fold structure from Task 1's findings — no borough hierarchy or temporal-lag
   mechanism in these three models, so the doc's default heuristic doesn't directly apply;
   propose a concrete design (e.g. grouped by area, or by whatever axis Task 1 shows carries
   the real correlation structure) before running anything.
3. Run grouped K-fold CV (`arviz_stats.loo_kfold`, `group_by=...`) for AZ3, AZ0a, M5.
4. Compute held-out predictive calibration (PIT / coverage) from the K-fold refits — the
   genuinely out-of-sample complement to `diagnostics.py`'s in-sample `_check_calibration`.
5. Report both axes (K-fold ELPD, held-out calibration) side by side, not collapsed into one
   number.
6. **Result storage for reuse**: original plan was to save each fold's raw refit under
   `results/traces/kfold/{model}/fold_{k}.nc` plus per-model/combined summary CSVs, mirroring
   the `comparison.csv`/`comparison_meta.json` cache convention. **Actually built**: summary
   JSON/CSVs only (no raw fold traces — see the "Result storage" paragraph above for why),
   and moved to `results/artifacts/kfold_comparison/` rather than `results/traces/kfold/` for
   discoverability.

**Decisions locked in**:
- Scope: **200-area development sample** (tractable refit cost), per user instruction.
- Fold count: **K=10**. Standard default across LOO tooling (ArviZ's own `loo_kfold` example,
  R's `loo` package) is 10-fold; going lower (e.g. 5) would roughly halve compute at a real
  cost to the ELPD estimate's precision, and 10 × 3 models = 30 refits is tractable on the
  32-core/128GB machine run in parallel batches (per [[project_machine_specs]]: 2-3 concurrent
  8-chain jobs at a time, not serialized) rather than one at a time.
- Fold structure: **leave-area-out** (group_by/folds = area index, so all 10 years of a held-
  out area move together). Decided from the models' own structure, not Task 1's spatial
  findings directly: none of AZ0a/AZ3/M5 pool information ACROSS years (no AR/temporal-lag
  sharing in AZ0a/AZ3; M5's lag kernel is fully-pooled/shared, not year-linked), so a
  leave-year-out fold would test nothing about the models' actual cross-area sharing
  mechanisms (global sigma_plan/sigma_ben/rho/etc., plus M5's explicit spatial term) — leave-
  area-out is the only fold axis that stresses what these models actually share.

**Deferred, not part of this task** (per `model-evaluation-methods.md`'s own cost/value
ranking): the census-anchored held-out-decade check (#3 in that doc) — biggest surgery,
highest cost, revisit only if K-fold leaves the AZ3-vs-M5 question unsettled.

---

## Task 3 — Stopping criterion + stakeholder communication method doc

**Status: DONE.** Written up in `docs/model-stopping-criteria-and-communication.md`: a
five-axis stopping checklist (convergence, calibration, predictive accuracy/parsimony,
domain fitness, diminishing returns) with concrete thresholds tied to this codebase's own
diagnostics, applied directly to AZ3 using Task 1/2's numbers — **AZ3 clears all five
criteria and is the recommended model to stop iterating on**. Also specifies a three-tier
(confident / characterized-ambiguous / genuinely-diffuse) scheme for communicating
year-by-year estimates to stakeholders, built entirely from existing tooling
(`z_identifiability_summary`, `detect_z_multimodality`/`plot_z_area_modes`,
`hierarchical_mode_summary`) rather than new machinery — directly encoding the AZ3 Phase 3
E01002702 lesson (a misleading mean between two real modes) as a "what not to do."

**Goal**: a defensible answer to "when is this model good enough to ship," and a concrete
plan for producing and explaining the (uncertain) year-by-year area-level estimates to
stakeholders.

Steps:
1. Survey standard Bayesian "good enough" criteria and apply them to this project
   specifically:
   - Convergence (r-hat/ESS/divergences) — necessary, not sufficient.
   - Posterior-predictive calibration (in-sample from `diagnostics.py`, out-of-sample from
     Task 2's K-fold refits).
   - Predictive-accuracy plateau — does added model complexity (AZ0a → AZ2 → AZ3 → AZ4/AZ5)
     keep buying real K-fold ELPD gains, or has it already flattened?
   - Diminishing-returns evidence already visible across the AZ-family phases themselves
     (each phase's marginal win/cost, as logged in `az-family-work-plan.md`).
2. Propose concrete numeric thresholds tied to diagnostics already built in this codebase —
   r-hat, ESS, divergence count, coverage, `frac_flat_despite_active`, year-allocation
   confidence — as an explicit checklist, not a vague standard.
3. Communication plan for year-by-year estimates:
   - Point estimate + credible interval per area/year.
   - Explicit flagging of genuinely multimodal/ambiguous years rather than reporting a
     misleading mean (the concrete lesson from AZ3 Phase 3's E01002702 deep-dive).
   - A tiered confidence scheme reusing existing tooling (`z_identifiability_summary`,
     `hierarchical_mode_summary`, `plot_z_area_modes`) rather than inventing new machinery.
4. Deliverable: `docs/model-stopping-criteria-and-communication.md`.

**Dependency**: needs Task 2's comparison numbers to state the predictive-accuracy-plateau
argument with evidence rather than assertion.

---

## Ground rules for this round

- Each task's "open questions" above must be resolved (by the user) before that task's work
  starts — no silent scope decisions on compute cost or fold design.
- Update this doc's per-task status immediately when a task starts/completes, mirroring
  `az-family-work-plan.md`'s convention.
- Fold durable findings into `model-progression-notes.md` once all three tasks conclude.
