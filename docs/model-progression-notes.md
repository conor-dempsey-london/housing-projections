# Model progression: open threads & design notes

Reference doc for where the `housing-projections` model iteration stands, what's
been tried, what's still open, and a detailed look at the marginalized
profile-library alternative to M14. Read this before starting a new modelling
iteration — it's meant to save re-deriving context that's already been worked
out (and re-discovering dead ends that have already been ruled out).

> **Update**: M0-M16 (sections 1-3 below) were superseded by a fresh-slate
> rewrite, the **AZ (Anchored Zero-sum) family**, after M9-M16 all turned out
> to share one underlying failure mode (`z_flatness_summary`: several produced
> z that is ~95-100% flat despite ~95% of areas having real active P/E signal
> — good r-hat/divergence diagnostics do not imply the model is doing anything
> with the data). Sections 1-3 remain as a record of what was tried and ruled
> out in that family; **section 0 covers the AZ family, which is where active
> work is now.**

## Table of contents

- [0. The AZ (Anchored Zero-sum) family — current state](#0-the-az-anchored-zero-sum-family-current-state)
  - [AZ0b in detail: why it's not (yet) a win over AZ0a](#az0b-in-detail-why-its-not-yet-a-win-over-az0a)
  - [Reusable infrastructure fix (affects every Potential-based model, AZ or M-family)](#reusable-infrastructure-fix-affects-every-potential-based-model-az-or-m-family)
  - [Moment matching for `pm.Potential`-based models — a reusable pattern, not (yet) committed to the repo](#moment-matching-for-pmpotential-based-models-a-reusable-pattern-not-yet-committed-to-the-repo)
  - [Open next steps for the AZ family](#open-next-steps-for-the-az-family)
- [1. Model-by-model: what each one is and how it did](#1-model-by-model-what-each-one-is-and-how-it-did)
- [2. Unexplored / open threads](#2-unexplored-open-threads)
  - [Z-prior architecture](#z-prior-architecture)
  - [Spatial](#spatial)
  - [Temporal / likelihood](#temporal-likelihood)
  - [Model comparison / LOO](#model-comparison-loo)
  - [Diagnostics infrastructure](#diagnostics-infrastructure)
  - [Closing the original question](#closing-the-original-question)
- [3. The marginalized profile-library alternative — M16, done and working](#3-the-marginalized-profile-library-alternative-m16-done-and-working)
  - [What M14/M15 actually did](#what-m14m15-actually-did)
  - [Obstacle 1 — `pm.Potential` isn't traceable, `pm.CustomDist` is](#obstacle-1-pmpotential-isnt-traceable-pmcustomdist-is)
  - [Obstacle 2 — `z` can't be a `pm.Deterministic` either](#obstacle-2-z-cant-be-a-pmdeterministic-either)
  - [Obstacle 3 — `area` must be a batch dimension, not folded into one opaque call](#obstacle-3-area-must-be-a-batch-dimension-not-folded-into-one-opaque-call)
  - [Recovering `z` after marginalized sampling](#recovering-z-after-marginalized-sampling)
  - [A fourth gotcha: two `CustomDist`s sharing one marginalized RV](#a-fourth-gotcha-two-customdists-sharing-one-marginalized-rv)
  - [What this buys, confirmed](#what-this-buys-confirmed)
  - [What this does *not* fix](#what-this-does-not-fix)

## 0. The AZ (Anchored Zero-sum) family — current state

**Core idea** (validated by direct simulation before implementation): `z`'s
prior is `mu_area = D/n_years` (fixed per-area mean, exact census anchor) plus
`pm.ZeroSumNormal` deviations (`sigma_delta = floor + k*|D|`), so z sums to the
census total *by construction* — no separate census-constraint likelihood
term needed. Confirmed via direct simulation that `ZeroSumNormal`'s pairwise
correlation is exactly `-1/(n_years-1)`: a spike in one year is absorbed
collectively across all other years (small shared decrement), not by a single
offsetting trough. This is the key structural difference from every M0-M16
model, all of which fit z freely then reconciled against D via a separate
likelihood term.

| Model | What it adds | Verdict |
|---|---|---|
| AZ0 | Zero-sum z prior + 3-way per-record mixture likelihood (same-year / one-year-prior / free-scale noise) per source | **Catastrophic non-convergence** (max r-hat 17.8, ~83% divergent draws, 3/4 chains in distinct modes). Root cause, confirmed by direct trace inspection, not assumed: `sigma_noise_P/E` collapsed to 1e-81..1e-108 (different magnitude per chain) — 54.3%/28.2% of P_obs/E_obs cells are exactly 0, and `StudentT(mu=0, sigma=sigma_noise→0)` has unbounded density there, a classic degenerate-variance mixture pathology. Left in the codebase as-is (not patched directly) — superseded by AZ0a→AZ0b below. |
| AZ0a | Zero-sum z prior alone + plain `add_observation_likelihoods` (no mixture) | **Confirmed converges cleanly** on real 200-area data (max r-hat 1.006, min ESS 3180, 0 divergences). Establishes the prior design works; this is the current best-validated model in the family. |
| AZ0b | AZ0a + 2-way backward-only reallocation mixture (same-year vs one-year-prior, no noise branch) per source | Motivated by a fresh cross-source lag-matching check on this exact data (48.1% of matched P/E pairs show E leading P by 1-2 years vs. 27.2% lagging, 24.7% same-year — a real backward skew in the raw data). **Result: not a validated improvement — see below.** |

### AZ0b in detail: why it's not (yet) a win over AZ0a

Dropping the noise branch (vs. AZ0's 3-way mixture) removes AZ0's specific
degenerate-mode failure — both remaining components (same-year, prior-year)
are centred on a genuinely moving target (`z`), not a fixed point, so there's
no equivalent free-scale-collapsing-onto-zero pathology. AZ0b samples without
divergences. But four independent checks all point at the same deeper
problem — the 2-way mixture creates near-discrete component-identity
ambiguity for a meaningful fraction of cells (removing a cell likely flips
whether it's "explained" as same-year or reallocated-from-prior-year, rather
than producing a smooth posterior shift):

1. **Convergence is worse than AZ0a's**: max r-hat 1.09 vs 1.006, min ESS 31
   vs 3180 (still 0 divergences, no chains trapped in distinct modes).
2. **Year-allocation confidence collapses**: areas with low confidence in
   *which* year absorbed a change jumps from 4% (AZ0a) to 39% (AZ0b) — total
   change per area (the census-anchored sum) stays reliable regardless.
3. **The naive LOO win is not trustworthy**: `az.compare` shows AZ0b beating
   AZ0a by a large margin (elpd -6900±130 vs -9100±120, stacking weight
   0.91/0.09) — but AZ0b's `P_like` PSIS-LOO has Pareto k>0.7 for 561/2000
   points (28%) vs AZ0a's 100/2000 (5%). A **full moment-matching correction**
   (built specifically for this — see below) could not improve a single one
   of AZ0b's 561 bad-k `P_like` points (elpd unchanged to the decimal after
   ~46 minutes of correction attempts across the complete set; only 2/80
   `E_like` points improved). That's a much stronger signal than ordinary
   PSIS unreliability — moment matching only fails outright when the true
   leave-one-out posterior isn't reachable by an affine (shift/scale)
   transform of the full posterior, consistent with the discrete
   component-identity story above.
4. **Ruled out tuning as the cause, not just assumed**: re-sampled at
   `target_accept=0.98` (up from 0.9) specifically to test whether this is a
   step-size problem. It made r-hat/ESS *worse* (max r-hat 1.09→1.12, min ESS
   31→22), not better — confirming the geometry is structural to the mixture,
   not a NUTS-tuning shortfall. `target_accept` is left at the default in the
   saved model.

**Net**: AZ0b is not currently a demonstrated improvement over AZ0a. Its
apparent LOO advantage is an artefact of unreliable PSIS-LOO on a model whose
leave-one-out geometry moment matching cannot correct. Until resolved,
**AZ0a remains the model to build from**, and AZ0b's 2-way mixture mechanism
should not be assumed reliable if reused elsewhere without redesign.

### Reusable infrastructure fix (affects every Potential-based model, AZ or M-family)

`_attach_pointwise_log_likelihood` (`models/base.py`) used attribute-style
assignment (`trace.log_likelihood = ...`) to attach a Potential-based mixture's
pointwise log-likelihood into the trace's `log_likelihood` group. On the
`xarray.DataTree` objects `pm.sample`/`nutpie.sample` now return, that
assignment reads back correctly *in the same session* but silently fails to
update the node `to_netcdf()` serialises from — confirmed via a minimal
DataTree repro. Every saved Potential-based trace (AZ0, AZ0b at minimum) had
an **empty `log_likelihood` group after save/reload**, which is why
`housing-projections compare` raised `"No log likelihood data named P_like
found"` for AZ0b until this was fixed. Fixed to item-style assignment
(`trace['log_likelihood'] = ...`); regression test added
(`test_pointwise_log_likelihood_survives_save_reload` in `tests/test_models.py`)
since the existing tests only checked in-memory state, which is exactly why
this slipped through originally.

### Moment matching for `pm.Potential`-based models — a reusable pattern, not (yet) committed to the repo

`pm.stats.loo_moment_match` / `arviz_stats.loo_moment_match` need
`log_prob_upars_fn`/`log_lik_i_upars_fn` callables operating in *unconstrained*
parameter space — neither is derivable automatically for a `pm.Potential`
likelihood (no observed RV for PyMC to introspect). Built and verified this
session (script lived in the session scratchpad, **not committed** — rebuild
following this recipe if needed again):

1. `model.replace_rvs_by_values([model['P_like_pointwise'], ...])` — the raw
   Deterministic graph is written in terms of the RVs themselves (e.g. the
   `ZeroSumNormal` FreeRV output), not their value_vars; PyMC only does that
   substitution internally at compile time, so it must be done explicitly
   before vectorizing.
2. `pytensor.graph.replace.vectorize_graph(outputs, replace={value_var:
   batched_placeholder})` to add a batch dimension over posterior draws to
   `model.logp()` and the (now value_var-based) pointwise graphs — reuses
   PyMC's actual logp graph (correct priors, Potential terms, and transform
   Jacobians by construction) rather than hand-deriving the math.
3. Forward-transform posterior draws (constrained) into each value_var's
   unconstrained representation by hand for the specific transforms in play
   (`LogTransform`, `LogOddsTransform`, `ZeroSumTransform` — the latter's
   `extend_axis_rev` ported to numpy, confirmed negative-axis-safe under an
   added batch dimension).
4. Verified correctness against `model.compile_logp()` (PyMC's own compiled,
   non-batched function) on several draws before trusting it at scale — max
   abs error ~1e-11.

Performance note: a full-batch (6000 draws) logp eval took ~1.7s, a
pointwise-likelihood eval ~1-5s: fast enough to make the ~1350-point full
moment-matching run (P_like + E_like) complete in about an hour rather than
the many-hours a naive per-draw Python loop would have taken.

### Open next steps for the AZ family

- **AZ0b's core question is unresolved**: does backward reallocation help at
  all, once evaluated with a trustworthy method? Two paths: exact k-fold CV
  (`az.loo_kfold` / `pm.stats.loo_kfold`, expensive — refits the model K
  times) for a trustworthy answer on the current formulation, or redesigning
  the reallocation mechanism to avoid discrete component-identity ambiguity
  (e.g. a continuous lag-weighting instead of a hard same-year/prior-year
  mixture, or an analytically marginalized version — `pymc_extras.marginalize()`
  and `_build_temporal_reallocation_likelihood_marginalizable` in the M-family
  work show this is possible in principle for this codebase, at real
  engineering cost — see section 3 below for the obstacles that entails).
- **The ~half of AZ0a's worst-fit cells that exceed the area's entire decade
  census total D** were explicitly out of scope for AZ0b (these look like
  genuine outliers/source noise, not timing misalignment) and remain
  unaddressed by any AZ model so far. A properly-floored noise/outlier
  mechanism for these is still a later, separate addition.
- Spatial misallocation was re-checked (independently, via Moran's I and a
  neighbour-cancellation permutation test) as an alternative explanation for
  AZ0a's D-exceeding residuals and again found weak/inconsistent (P:
  p=0.037 one-sided; E: p=0.147, not significant) — consistent with the
  M-family's earlier negative finding (section 2). Not worth pursuing further
  without new evidence.
- AZ0 itself remains in the codebase in its known-broken state, registered in
  `_ALL_MODELS` — worth archiving once the AZ0b question above is settled, to
  avoid confusing future `diagnose`/`compare` runs.

## 1. Model-by-model: what each one is and how it did

| Model | What it adds | Verdict |
|---|---|---|
| M0 / M0h | Global / per-area flat z prior, simple StudentT likelihoods | Baseline. Works, uninteresting — no lag, no disagreement handling. |
| M1 / M1h | + `_build_lag` planning convolution | Lag helps aggregate fit, still one shared kernel for every year/area. |
| M5 | + spatial misallocation (`_build_spatial_misallocation`) | Never revisited after the empirical spatial-reallocation test (section 2) found no real signal — the mechanism this model encodes may not be worth keeping. |
| M6 | + AR(1) z instead of iid per-year | Never combined with any of the gating/reallocation architecture from M9 onward. |
| M7 | + zero-inflated planning likelihood, borough pooling | First `pm.Potential`-based mixture — was silently missing pointwise log-likelihood until this session's fix (section 2). |
| M8 | + time-varying planning noise | Not touched this session; no known issues, also not re-examined. |
| M9 | Per-area hierarchical sigma_slab, independent P/E lag kernels | Established the per-area sigma hierarchy reused unchanged through M14. `mu_log_sigma` r-hat=1.17 originally motivated M10's borough version. |
| M10 | + per-borough sigma_slab, capture-rate kappa | Addressed M9's mu_log_sigma issue via borough pooling — never merged with M11+'s gating architecture. |
| M11 | Joint agreement-gated mixture (single `rho_agree` gates both P and E together) | First real attempt at "track P/E when they agree, ignore when they disagree." Centred-z version had 2-of-4 chains trapped in distinct modes (rho_agree, sigma_agree_plan); non-centering fixed the *sampling* pathology but not areas with a large census/(P+E) coverage gap (r-hat there scales with gap size, not parameterization). E01002694 investigation showed a "flat" solution can be the genuinely higher-posterior-density outcome given a badly-fit shared lag kernel, not a sampler failure — the real problem was the *architecture* (shared lambda_weights), not the parameterization. |
| M12 | Independent per-source `rho_P`/`rho_E` gating | Motivated by the area/year taxonomy (P-only 10.6%, E-only 27.4% — one source silent, not conflicting, is the dominant non-trivial pattern). Made convergence *worse* than M11: all 4 chains landed on different `rho_P`/`rho_E` values (max r-hat 4.24, min ESS ~4) — splitting the gate multiplied the number of label-switching-like modes. |
| M13 | + per-record temporal-offset marginalization, replacing `_build_lag`'s shared kernel entirely | **Best model so far.** Fixed the M11/M12 scalar multimodality outright (all chains agree on rho_P/rho_E/sigma_agree/disagree) — confirms the shared lag kernel, not per-source gating, was the deeper cause. Traded that for a narrower problem: 462 divergences, 460 in a single chain (a geometry issue, not multimodality). Individual z-cell r-hat still poor (100% of areas low year-confidence) but this looks like the same benign "weakly identified which year" issue characterized for M11, now compounded by offset marginalization spreading credit across candidate years. |
| M14 | + flat D/n_years baseline z prior with a marginalized-*by-Gibbs* single-active-year profile library | **Worst convergence of all five in this family** — max r-hat 5.67, 1553 divergences (concentrated in 2 of 4 chains), ~20x slower to sample (compound NUTS+CategoricalGibbsMetropolis, no nutpie). `sigma_agree_plan` collapsed to ~1e-10 in *every* chain (agreement without multimodality, but on a likely-overfit solution). The null/flat library row turned out not to be functionally privileged — see section 3. |
| M15 | M14 + regularised (Finnish) horseshoe prior on `amplitude`, null library row dropped | **Real improvement over M14** on real 200-area data: max r-hat 2.11 (vs 5.67), mean r-hat 1.11 (vs 2.64), 777 divergences (vs 1553, though still worse than M13's 462), no chains trapped. Confirms the redundant-null geometry was a genuine contributor to M14's instability, not sampled `profile_k`/Gibbs alone — though Gibbs is still in the mix here (M16 isolates that variable). Note: `plan_cov_90`/`ben_cov_90` in the CLI diagnostics came back near-zero (0.008/0.006) — this is a **pre-existing tooling gap**, not a real finding: `_check_calibration` looks for `sigma_plan`/`sigma_ben` trace variables, which don't exist under the M11+ agree/disagree-gated architecture (it uses `sigma_agree_*`/`sigma_disagree_*` instead), so it silently falls back to a degenerate z-only coverage check. Affects M11-M16 equally; not investigated further this session. |
| M16 | M15 + `profile_k` fully **marginalised** via `pymc_extras.marginalize()` instead of sampled via `CategoricalGibbsMetropolis` | Built and validated on synthetic data (see section 3 — this took three real technical obstacles to resolve, not a drop-in swap). Confirmed nutpie-compatible again (unlike M14/M15, which always forced `use_nutpie=False`) via `sample_stats` showing nutpie-specific fields (`fisher_distance`, `divergence_message`). Real-200-area result pending. |

## 2. Unexplored / open threads

### Z-prior architecture
- **Process/horseshoe prior alternative** — partially pursued: M15 applies a regularised (Finnish) horseshoe to `amplitude` directly (see the model table above and `_build_z_prior_profile_library_horseshoe`'s docstring), which is the "spike-and-slab-via-continuous-shrinkage" half of this idea. The other half — a horseshoe (or similar) directly on the *raw per-year deviations* with no discrete profile library / `profile_k` at all, fully differentiable and nutpie-compatible from the start — is still untried.
- **Marginalized profile library** — investigated and **closed as not a shortcut**: `pymc_extras.marginalize()` was tested directly against this architecture and raises `"No RVs depend on marginalized RV profile_k"` — it only traces dependencies through actual observed `Distribution`s, not through a `Deterministic` (`z`) feeding a `pm.Potential` (the pattern used by every marginalised mixture in this codebase, including `_build_temporal_reallocation_likelihood`). Getting proper marginalisation still requires the full hand-written rewrite described in section 3 — `pmx.marginalize()` does not shortcut it.
- **The null-row redundancy problem found in M14** — addressed in M15 via the horseshoe prior above (see that row's verdict once sampled) rather than via an explicit hurdle gate, specifically to avoid adding a *second* discrete per-area latent on top of `profile_k` (which would risk compounding M14's compound-step instability rather than fixing it). A literal hurdle (Beta-Bernoulli "active or not", no amplitude in the "not" branch) remains untried if the horseshoe version doesn't resolve it.
- **More than one active atom per area** (`K_max > 1`, summing multiple library rows): explicitly deferred in M14 in favour of capping at one atom; not attempted.
- **Pair-swap/reallocation shapes inside the z-prior library** itself (as opposed to leaving reallocation entirely to M13's likelihood-side offset marginalization): deliberately excluded from M14 to avoid two mechanisms competing to explain the same "which year" ambiguity. Worth reconsidering only if the likelihood-side offset mechanism were removed instead of kept.

### Spatial
- **Empirically tested and closed (negative result)**: 1-hop Queen-neighbour spatial reallocation, tested two ways — Moran's I on the census gap and on P/E disagreement (I≈0.004–0.03, not significant), and a rigorous optimal-assignment-vs-null-shuffle test mirroring the temporal one (real reduction *below* null at every window, z≈-1 to -1.7). Conclusion: no evidence 1-hop spatial reallocation helps, unlike temporal (+8pp over null at window ±2). This closes the thread for 1-hop Queen contiguity specifically — 2-hop or population-weighted variants were not tested.
- **M5's existing spatial-misallocation mechanism** was never revisited in light of this negative finding — it may be worth removing or justifying separately.
- **Partial/fractional spatial reallocation**: the user's original question mentioned "either complete or partial" reallocation; both empirical tests used exact optimal assignment (effectively "complete" moves), not a continuous partial-split mechanism. A genuinely fractional reallocation test was never run.

### Temporal / likelihood
- **AR(1) (M6) or spatial misallocation (M5) combined with the M9-M14 gating/reallocation architecture**: never attempted — these two axes (temporal structure of z itself vs. the P/E disagreement-gating likelihood) have been developed independently.
- **`pi_offset_P`/`pi_offset_E` validation**: M13's fitted values (peaked at 0, skewed toward P lagging E) were compared qualitatively to the raw-data offset histogram but not with a formal goodness-of-fit check.
- **Window size sensitivity**: M13/M14 fix `max_offset=2` based on the empirical test's clear signal at that window; ±1 and ±3 were checked empirically on raw data but never re-tried as the model's actual `max_offset`.

### Model comparison / LOO
- **`housing-projections compare` has not actually been run across M9-M14** since fixing the pointwise log-likelihood bug — the plumbing is fixed but the comparison itself is still outstanding.
- **M11's likelihood is named `PE_like` (joint)**, while M12/M13/M14 use split `P_like`/`E_like` — `compute_model_comparison`'s hardcoded `var_name='P_like'` will still fail specifically for M11. Not yet resolved; the deeper question (what's the fair apples-to-apples LOO quantity when models decompose the likelihood differently — joint vs split vs plain observed RVs) hasn't been settled either.
- **Existing saved traces for M7, M11, M12, M13 predate the pointwise log-likelihood fix** and would need resampling before LOO/`compare` will work on them.
- **Pareto-k diagnostics and LOO-PIT calibration** (both covered by the model-evaluation skill) have never been run on any model in this family — LOO numbers, once computable, haven't been checked for reliability.

### Diagnostics infrastructure
- **`full_diagnostics`'s other checks** (`_check_census_constraint` residuals, `_check_morans_i` on spatial residuals, posterior-predictive calibration coverage, `_check_sigma_slab_vs_disagreement`, `_check_kappa_vs_recording_rate`) have not been run on M11-M14 this session — only the CLI summary table, `z_identifiability_summary`, and manual `check_chain_agreement` calls were used.
- **`sigma_slab`'s own hierarchy** has shown poor r-hat in the worst-converging models (M12: 4.15, M14: elevated too) but has never been redesigned or specifically diagnosed — every model from M9 through M14 reuses the same non-centred log-normal hierarchy unmodified.
- **Soft-outlier correlation with convergence**: 85 LSOAs are flagged as soft outliers and retained (not excluded); whether they disproportionately show up among the poorly-converged areas in M11-M14 has not been checked.
- **`pixi run report` (the self-contained HTML report)** has not been generated for any of M9-M14.

### Closing the original question
- The original ask (session start) was: make z track P/E in years they agree, ignore them when they disagree, without collapsing to a constant per-area z. M13's docstring explicitly names E01002694 as the validation case ("z[2014] should now be free to track the P=7/E=8 spike directly") — the M13 diagnostics artifact includes this area's z-timeseries, but there's been no explicit side-by-side narration confirming the fix actually worked as intended for E01002694 / E01002802 / E01002719 specifically (as opposed to reporting aggregate r-hat/divergence numbers). Worth a direct callback before treating the original problem as resolved.

## 3. The marginalized profile-library alternative — M16, done and working

> **Update**: this was originally scoped out below as "expensive, not
> attempted." It turned out to be tractable — `pymc_extras.marginalize()`
> *can* do the heavy lifting, once the likelihood is restructured the right
> way. M16 implements this. The three real obstacles hit along the way (not
> guesses — each confirmed by a failing test before being fixed) are
> documented below so nobody has to rediscover them.

### What M14/M15 actually did

```
z[a, :] = D[a]/n_years + amplitude[a] * profile_library[profile_k[a], :]
```

`profile_k[a]` is a discrete `pm.Categorical` per area, sampled via PyMC's
auto-assigned `CategoricalGibbsMetropolis`, alternating with NUTS for
everything continuous in a `CompoundStep`. Literal discrete sampling, not
marginalization — each posterior draw commits to one specific `profile_k`
per area, and M14/M15 always forced `use_nutpie=False` because of it.

### Obstacle 1 — `pm.Potential` isn't traceable, `pm.CustomDist` is

`pmx.marginalize(model, rvs_to_marginalize=['profile_k'])` on the M14/M15
construction (z as a `Deterministic`, likelihood as a `pm.Potential`) raises:

```
ValueError: No RVs depend on marginalized RV profile_k
```

`pmx.marginalize()` traces dependencies by walking the graph for actual
observed `Distribution` nodes — its own worked example is
`y = pm.Normal("y", mu=mu[comp], ..., observed=y_obs)`. A `pm.Potential` isn't
a `Distribution` it can enumerate branches over, so it finds nothing
downstream of `profile_k`. Fix: rebuild the likelihood as a `pm.CustomDist`
(a genuine Distribution/RV node) with a hand-written `logp=` callable that
does the *exact same math* as the `pm.Potential` version — see
`_build_temporal_reallocation_likelihood_marginalizable`, used only by M16.

### Obstacle 2 — `z` can't be a `pm.Deterministic` either

Even after switching to `CustomDist`, marginalizing raised:

```
NotImplementedError: Cannot marginalize profile_k due to dependent Deterministic z
```

A marginalized RV can't have a downstream `Deterministic` depending on it (a
`Deterministic` needs one concrete value per draw; `profile_k` no longer has
one once marginalized out). Fix: `_build_z_prior_profile_library_horseshoe`
gained a `wrap_z_as_deterministic` flag — M16 builds `z` as a bare pytensor
expression, never registered in the model, so it never appears directly in a
sampled trace. Consequence: `z`, `agreement_prob_P/E`, and pointwise
log-likelihood all vanish from the native trace and have to be reconstructed
after the fact (see "Recovering z" below) — this is real added complexity,
not just a technicality.

### Obstacle 3 — `area` must be a batch dimension, not folded into one opaque call

The first working `CustomDist` attempt kept `_build_temporal_reallocation_likelihood`'s
original strategy: flatten all active `(area, year)` cells across the *whole*
200-area grid into one global list, gather/scatter by index, and return one
`pm.Potential`-style scalar per area via `pt.set_subtensor`. Marginalizing that
raised:

```
NotImplementedError: The graph between the marginalized and dependent RVs
cannot be marginalized efficiently. You can try splitting the marginalized RV
into separate components and marginalizing them separately.
```

`pmx.marginalize()` needs `area` to remain a genuine *batch* dimension — since
`profile_k` has one value per area, marginalizing it is conceptually "for each
area, sum over that area's own candidate values," and treating the whole
200-area grid as one indivisible core-dimension computation hides that
structure. Fix: rewrite the likelihood to operate on **one area's row at a
time** (core dims = `year` only), using `pt.switch` on a per-year active/inactive
mask instead of global index-gathering across areas — computed for *every*
year, not just the active ones, since a per-row vectorizable function can't do
ragged/variable-length gathering. Needed two further fixes to make this
actually work under `CustomDist`'s automatic per-area vectorization:

- An explicit `signature=` kwarg (gufunc-style, e.g.
  `'(year),(),(),(),(),(offset)->()'`) — `CustomDist`'s default shape
  inference assumes simple elementwise broadcasting across all `dist_params`,
  which fails once `pi_offset`'s `(offset,)` shape can't broadcast against
  `z`'s `(year,)` shape.
- Every axis-insertion inside the `logp` function written as `[..., None]`
  (Ellipsis-relative), never `[:, None]`/`[None, :]` — the latter assumes a
  fixed, small number of leading dimensions, which breaks the moment
  `pymc_extras` vectorizes the function over extra leading batch axes (one
  for `area`, one for enumerating `profile_k`'s candidate values). Same
  reasoning killed a first attempt at the offset-window gather
  (`z[shifted_year_clipped]`, plain fancy indexing) — replaced with a
  broadcast-safe `pt.einsum('...y,tky->...tk', z, select_pt)` against a fixed
  one-hot selection tensor.

### Recovering `z` after marginalized sampling

Losing `z` as a live `Deterministic` (obstacle 2) means every diagnostic this
session has been built around has nothing to read post-sampling.
`pymc_extras.recover_marginals(idata, model=marginal_model, var_names=['profile_k'])`
reconstructs `profile_k`'s posterior *after* sampling, conditioned on the
sampled continuous parameters — verified this actually restores usable
per-draw samples (not just a summary). M16's `sample()` then reconstructs `z`,
`agreement_prob_P/E`, and `P_like`/`E_like`'s pointwise log-likelihood in numpy
from the recovered `profile_k` + `amplitude`, reimplementing the `CustomDist`'s
logp formula exactly (StudentT logpdf via `scipy.stats.t`, `logsumexp`/`logaddexp`
via `scipy.special`/`numpy`, vectorized over `(chain, draw)`).

### A fourth gotcha: two `CustomDist`s sharing one marginalized RV

`P_like` and `E_like` are separate `CustomDist`s that both depend on the same
`z` (hence the same `profile_k`). Marginalizing correctly accounts for
*both* in the total joint logp (verified: perturbing `E_obs` measurably
changes the marginal model's logp) — but `pm.compute_log_likelihood()`
afterward can't cleanly attribute the pointwise breakdown to each of the two
nodes separately: it emits a `NonSeparableLogpWarning` and one of the two
(observed non-deterministically — whichever the internal graph traversal
visits second) collapses to a degenerate `(chain, draw)` scalar instead of a
proper `(chain, draw, area)` breakdown. This is why M16 reconstructs
`log_likelihood['P_like']`/`['E_like']` manually in numpy too, rather than
relying on `pm.compute_log_likelihood()` on the marginalized model.

### What this buys, confirmed

- **No discrete latent anywhere in the sampled model** → nutpie-compatible
  again, confirmed via `sample_stats` containing nutpie-specific fields
  (`fisher_distance`, `divergence_message`) after sampling — unlike M14/M15,
  which always forced `use_nutpie=False`.
- Whether it actually fixes M14/M15's divergence problem (as opposed to just
  restoring nutpie) is an open empirical question — pending the real
  200-area run (see the model table in section 1).

### What this does *not* fix

Marginalizing `profile_k` does **not** resolve the null-row redundancy problem
(M14's finding, fixed differently in M15 via the horseshoe prior — see section
1). That issue is about `amplitude` being free to shrink to 0 regardless of
which row is chosen, with nothing structurally favouring "no activity" over
"activity with a tiny amplitude" — true whether `profile_k` is sampled
discretely or marginalized exactly. M16 keeps M15's horseshoe fix for this;
marginalization is an orthogonal, separate improvement (targeting the sampling
mechanism, not the prior's redundancy).
