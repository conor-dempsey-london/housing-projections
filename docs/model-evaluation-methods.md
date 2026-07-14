# Model evaluation methods — open ideas for a future round

**Status: ideas captured, not yet implemented.** `az-family-work-plan.md` explicitly
deferred LOO/moment-matching/k-fold comparison "until there's a real finalist worth
evaluating carefully." This doc exists so that when that point arrives, the options
below don't need to be re-derived. Nothing here has been built or run yet — treat it
as a plan to revisit, not a record of completed work.

## The problem, as already diagnosed in this codebase

`model-progression-notes.md`'s AZ0b entry contains the concrete finding that motivates
this doc: `az.compare` showed AZ0b beating AZ0a by a large margin (elpd −6900±130 vs
−9100±120, stacking weight 0.91/0.09), but AZ0b's `P_like` PSIS-LOO had Pareto k>0.7 for
561/2000 points (28%) vs AZ0a's 100/2000 (5%). A full moment-matching correction, built
specifically for this (recipe documented in `model-progression-notes.md` under "Moment
matching for `pm.Potential`-based models"), could not improve a single one of those 561
bad-k points — elpd unchanged to the decimal after ~46 minutes of correction attempts.

That's a specific signature, not generic PSIS strain: moment matching only fails outright
when the true leave-one-out posterior isn't reachable by an affine (shift/scale) transform
of the full posterior. The working hypothesis is that this comes from near-discrete
component-identity ambiguity in the reallocation mixtures (AZ0b's same-year vs
prior-year mixture) — removing a cell likely flips which component "explains" it, rather
than producing a smooth posterior shift. AZ0a itself (no discrete mixture) had far fewer
bad-k points (5% vs 28%), consistent with this: **the fix likely depends on which model is
flagged**, not a blanket policy for the whole AZ family.

- Non-mixture / continuous-lag models (AZ0a, AZ1a–f's continuous lag convolution,
  AZ2/AZ2b's smooth sigmoid ramp) are plausibly fine with standard PSIS-LOO +
  moment-matching — check Pareto k per-model before reaching for anything heavier.
- Discrete-mixture / gated models (AZ0b, and the M9-M14 agreement-gated family) are where
  PSIS-LOO and moment-matching are already shown not to be trustworthy.

## Candidate quantitative measures for a future round

### 1. Grouped K-fold CV (primary recommendation for flagged models)

`arviz_stats.loo_kfold` — confirmed available in the installed version, signature:

```
loo_kfold(data, wrapper, pointwise=None, var_name=None, k=10, folds=None,
          stratify_by=None, group_by=None, save_fits=False)
```

The `group_by` argument does exact, grouped K-fold CV (real refits, not PSIS-approximated),
which sidesteps the Pareto-k problem entirely. Fold assignment should match the
correlation structure of whichever model is being evaluated:

- Leave-one-**year**-out for models with temporal lag convolution (AZ1a-f) or AR(1) z (M6).
- Leave-one-**area**/**borough**-out for hierarchical partial-pooling models (M0h/M1h,
  M9/M10's per-borough sigma).

Cost: K refits per model. Real but tractable given the 32-core/128GB machine — reserve
it for models the Pareto-k check actually flags, not a blanket re-run of every model.

**Engineering cost, not conceptual cost**: `loo_kfold` needs a `SamplingWrapper`
implementing `sel_observations`, `sample`, `get_inference_data`, and `log_likelihood__i`
against this codebase's PyMC/`pm.Potential`-based models — comparable effort to the
moment-matching wrapper already built (and deliberately not committed) for the AZ0b
investigation. That wrapper's recipe (`model-progression-notes.md`, "Moment matching for
`pm.Potential`-based models") is a useful reference for the value-var/transform handling
a K-fold wrapper will also need.

### 2. Held-out predictive calibration from the K-fold refits

`diagnostics.py`'s `_check_calibration` computes posterior-predictive coverage, but it's
**in-sample** — it will look artificially good regardless of true model quality. Once
K-fold refits exist (from #1), the genuinely useful complementary metric is calibration
computed from each fold's *held-out* posterior predictive: a real PIT histogram / coverage
number that doesn't share PSIS-LOO's approximation risk. This is a second, independent
quantitative axis (calibration) alongside K-fold ELPD (predictive accuracy) — the two
answer different questions and neither substitutes for the other.

### 3. Domain-grounded held-out census validation (most defensible, most expensive)

Every AZ-family model anchors `z`'s prior mean directly to `D` (the trusted, only-every-
10-years census total): `mu_area = D/n_years`. That construction makes a specific,
non-approximate test available: hold out one decade's census figure, refit with the
z-prior anchored only on the *other* decades, and compare the model's predicted decadal
total for the held-out period against the recorded census number for that period (treated as
exact for this comparison, per the project's current census-exactness assumption).

This is the single most defensible quantitative check available, because it tests the
actual thing under active iteration — the temporal reallocation / lag mechanism — against
real data the project already treats as ground truth, not an approximation of held-out
performance. It's the natural quantitative complement to `plot_spike_tracking_examples`
and the project's core goal ([[project_goal_year_by_year_inference]] in memory: the
intercensal total is already trusted, the annual pattern is the real deliverable).

Caveat: this is bigger surgery than #1/#2 — it requires refitting the z-prior's
census-anchoring logic itself for the held-out decade, not just refitting the model as-is
on a data subset. Treat as the highest-value but highest-cost option of the three.

## Smaller gaps worth fixing regardless of which option above gets picked

- **`compute_model_comparison` (`analysis.py`) hardcodes `var_name='P_like'`** — it
  silently scores planning-source fit only, ignoring `E_like` entirely, and breaks
  outright on M11's joint `PE_like` variable name. Already flagged in
  `model-progression-notes.md` under "Model comparison / LOO". Any renewed comparison
  effort should score a combined P+E quantity (sum of pointwise log-likelihoods, or a
  joint variable), not one source in isolation.
- **Pareto-k diagnostics and LOO-PIT calibration have never actually been run** across
  the AZ family as a matter of routine — `plot_spike_tracking_examples`
  (`plots/core.py`) opportunistically surfaces the worst Pareto-k area when a usable
  `P_like` log-likelihood exists, but that's a bonus category inside a different plot,
  not a standalone check. Worth a dedicated `check-pareto-k`-style pass per model before
  investing in K-fold infrastructure, simply to see how many models are actually
  affected vs already fine.

## Next step, when this round starts

Before building the K-fold `SamplingWrapper`, run a cheap first pass: compute standard
`az.compare` + Pareto-k for every current AZ-family model with a usable log-likelihood
group, and sort them into "PSIS-LOO looks trustworthy" vs "needs K-fold" before deciding
how much wrapper-engineering effort is actually justified.
