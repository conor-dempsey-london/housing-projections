# Model finalization work plan — full-dataset characterization, CV, stopping criterion

**Status doc for this round of work.** Like `az-family-work-plan.md`, this is a live
checklist, not a finished record — check back here at the start of each task, update it
immediately when a task's status changes, and fold durable findings into
`model-progression-notes.md` once this round concludes. Three related tasks, described
below, run in the order given for the dependency reasons stated in each task's rationale.

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

**Status: in progress.**

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

**Status: not started.**

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
6. **Result storage for reuse**: save each fold's raw refit (trace + fold assignment) under
   `results/traces/kfold/{model}/fold_{k}.nc`, and the assembled per-model K-fold summary
   (per-fold elpd, held-out calibration/PIT stats, fold membership) as
   `results/traces/kfold/{model}_kfold_summary.csv`, plus a top-level
   `results/traces/kfold/comparison.csv` combining all three models — mirroring the existing
   `comparison.csv`/`comparison_meta.json` cache convention (mtime-checked, not recomputed
   unless a fold trace changes) so a future model addition doesn't require rerunning already-
   completed folds.

**Decisions locked in**:
- Scope: **200-area development sample** (tractable refit cost), per user instruction.
- Fold count: **K=10**. Standard default across LOO tooling (ArviZ's own `loo_kfold` example,
  R's `loo` package) is 10-fold; going lower (e.g. 5) would roughly halve compute at a real
  cost to the ELPD estimate's precision, and 10 × 3 models = 30 refits is tractable on the
  32-core/128GB machine run in parallel batches (per [[project_machine_specs]]: 2-3 concurrent
  8-chain jobs at a time, not serialized) rather than one at a time.
- Fold structure: to be decided from Task 1's findings before Task 2 starts (see Task 1).

**Deferred, not part of this task** (per `model-evaluation-methods.md`'s own cost/value
ranking): the census-anchored held-out-decade check (#3 in that doc) — biggest surgery,
highest cost, revisit only if K-fold leaves the AZ3-vs-M5 question unsettled.

---

## Task 3 — Stopping criterion + stakeholder communication method doc

**Status: not started.**

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
