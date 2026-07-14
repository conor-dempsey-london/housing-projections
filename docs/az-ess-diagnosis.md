# AZ-family ESS diagnosis — AZ1a, AZ1b, AZ2

Separate from `az-family-work-plan.md` (the live per-phase log for the current modelling
round). This document is a focused pass over every AZ-family model that had a documented
convergence/ESS problem *before* AZ3's floor fix: diagnose the mechanism, try one concrete
fix, resample on the real 200-area dataset, and report the honest result — including the
negative ones. AZ3's own ESS story (the `sigma_plan`/`sigma_ben` floor) is already fully
covered in `az-family-work-plan.md` Phase 3 and the `az3-floor-followup` artifact; not
repeated here.

**Summary table** (max r-hat / min ESS bulk, real 200-area data):

| Model | Problem | Fix tried | Result | max r-hat | min ESS |
|---|---|---|---|---|---|
| AZ1a | max r-hat 1.74, min ESS 6 (4 chains) | 8 chains (diagnostic only, no model change) | Confirms architectural, not a sampling failure | 1.65 (8 chains) | 12 |
| AZ1b | max r-hat 1.12–1.19, min ESS 23–29 | Hard cap on tau (`AZ1c`) | **Negative — made it worse** | 1.32 | 19 |
| AZ1b | max r-hat 1.12–1.19, min ESS 23–29 | Remove E's lag entirely (`AZ1d`) | **Positive — clean win, best result in this doc** | 1.036 | 169 |
| AZ2 | min ESS 47 (bulk) | Smooth sigmoid ramp instead of hard cutoff (`AZ2b`) | **Positive on `sigma_delta_top_boost` — but see follow-up: the same pathology resurfaced on `sigma_plan`/`sigma_ben`** | 1.027 (was 1.054) | 126 (was 47) |
| AZ2b | `sigma_plan`/`sigma_ben` ESS 126/180 (4 chains), no leakage/ridge mechanism found | `target_accept` 0.9→0.97, 8 chains (sampler-only, no model change) | **Mixed — fixes `sigma_ben` (570), `sigma_plan` accepted as a disclosed limitation** | 1.015 (sigma_ben) | 570 (sigma_ben); sigma_plan stays 234–534 across configs |

New models added to the codebase: `AZ1c`, `AZ2b` (both registered in `models/__init__.py`,
`cli.py`, tested in `tests/test_models.py`). AZ1a unchanged except `sample_kwargs` now
requests 8 chains.

---

## AZ1a — fully-pooled lag convolution

### Symptom (already diagnosed pre-this-document)
Real-data run (4 chains): max r-hat 1.744 (8 params over threshold), min ESS 6,
year-allocation confidence collapsed to 89% low-confidence (vs AZ0a's 4%).

### Diagnosis (already established, re-verified here)
`lambda_weights_E` (the single Dirichlet lag-weight vector shared across all 200 areas) is
genuinely bimodal — different areas' true lag patterns disagree, and a single shared kernel
has no one right answer. This is an **architecture** problem: a fully-pooled parameter
mechanically cannot represent two areas' genuinely different lag behaviour simultaneously,
regardless of how long or how many chains you sample. AZ1b was already built specifically to
fix this (per-area hierarchical pooling) — this section exists to give AZ1a the same
diagnostic treatment AZ1b got (more chains, for a trustworthy split estimate) rather than
leave it at an under-powered 4-chain readout, not to find a tuning fix that doesn't exist for
a structural problem.

### Fix tried
`AZ1a.sample_kwargs` bumped to `chains=8, cores=8` (previously the `DEFAULT_SAMPLE_KWARGS`
default of 4/4) — no change to `build()`. This mirrors the exact treatment that made AZ1b's
residual multimodality trustworthy to report, applied here to see the *same* diagnostic
value, not to fix r-hat.

### Result — as expected, r-hat/ESS did NOT improve; the split is now cleanly characterized
| | 4 chains | 8 chains |
|---|---|---|
| max r-hat | 1.744 | 1.649 |
| min ESS | 6 | 12 |
| n_bad_rhat | 8 | 8 |
| frac_flat_despite_active | — | 16.0% |
| low year-confidence | 89.0% | 87.5% |

r-hat stayed bad (expected and correct — r-hat measures chain agreement, and a genuinely
multimodal posterior *should* disagree; forcing it to 1 would mean suppressing real
structure). min ESS roughly doubled simply because there are twice as many total draws, not
because mixing improved.

The 8-chain split is clean and directly interpretable, unlike the noisy 4-chain 2-vs-2
readout:

```
lambda_weights_E chain means (8 chains):
  chains 0, 3, 7   -> [0.083, 0.877, 0.040]   "E lags true completion by 1 year"   (3/8 = 37.5%)
  chains 1,2,4,5,6 -> [0.893, 0.061, 0.045]   "E is same-year"                     (5/8 = 62.5%)

lambda_weights_P chain means (8 chains):
  chains 0, 3, 7   -> [0.012-0.021, 0.051-0.056, 0.93-0.94]
  chains 1,2,4,5,6 -> [0.020-0.021, 0.028-0.029, 0.95]
```

Two things worth noting that weren't visible at 4 chains:
1. The true split is **~37.5%/62.5%**, not 50/50 — the 4-chain run's apparent 2-vs-2 tie was
   an artefact of too few chains, not the actual relative mass of the two explanations.
2. `lambda_weights_P`'s split, while much milder than E's, is **correlated with which E-mode
   the same chain landed in** — chains in the "E lags 1yr" group also carry slightly more P
   mass on the middle lag category than chains in the "E same-year" group. The two sources'
   ambiguity isn't independent per chain; a chain's initial trajectory settles both at once,
   consistent with both being driven by the same underlying z realization that chain
   happened to find first.

### Verdict
**No fix exists at the tuning/sampling level — confirmed, not just assumed.** AZ1a's
multimodality is a direct, structural consequence of forcing one shared lag kernel onto 200
areas with genuinely different lag behaviour. The only real fix is the architectural one
already built: AZ1b (and its own fix attempt, AZ1c, below). AZ1a should continue to be
treated as a **rejected intermediate step** documenting why AZ1b's design was necessary, not
as a model to bring into Phase 4.

---

## AZ1b → AZ1c — hard cap on tau

### Symptom (already diagnosed pre-this-document)
Real-data run: max r-hat 1.12–1.19 (4 vs 8 chains), min ESS 23–29, isolated to the per-area
`lag_P_raw_offset`/`lag_E_raw_offset` parameters (~10-15% of areas). Confirmed via direct
inspection: individual chains spend all 1500 draws entirely inside one of two disconnected
modes and never cross over — genuine hard multimodality for a minority of areas whose sparse
data (~10 obs/area/source) can't distinguish between two candidate lag years that each
explain a spike about equally well.

### What was already tried and ruled out
Tightening tau's *prior* (`tau_sigma` 1.5 → 0.5) — made it **worse** (max r-hat 1.12 → 1.24,
min ESS 23 → 12), because the posterior tau barely moved (e.g. `lag_E_tau` ~2.7-3.2 under the
tighter prior vs ~5.1-5.9 before) — the likelihood's pull toward per-area divergence is
strong enough to mostly override a 3x tighter prior. This document's fix attempt targets that
exact finding: since a softer *prior* nudge doesn't reach the mechanism, try a *hard,
inescapable* ceiling instead.

### Fix tried
Built `_build_hierarchical_lag_capped` (`models.py`) and a new model, `AZ1c`. Only change
from `_build_hierarchical_lag`: `tau` is reparameterised from a free
`HalfNormal(tau_sigma)` to `tau = tau_cap * Beta(2, 2)`, `tau_cap = 1.5`. Bounded by
construction (Beta support is `[0,1]`), not merely discouraged by a tighter prior scale — so
whatever made the earlier prior-tightening attempt fail (posterior tau resisting the prior)
cannot happen here; the actual per-area divergence effect is mechanically capped at 1.5,
regardless of what the likelihood wants. `tau_cap=1.5` is well below AZ1b's converged
2.6-5.9 (P) / 5.1-5.9 (E), and below even the failed tighter-prior attempt's ~2.7-3.2 — a
real, substantial constraint, not a token gesture.

### Result — negative, made it worse, and the tau posterior shows exactly why
| | AZ1b (chosen, 8 chains) | AZ1c (tau capped at 1.5, 8 chains) |
|---|---|---|
| max r-hat | 1.188 | **1.323** |
| min ESS | 29 | **19** |
| n_bad_rhat | 10 | 7 |
| frac_flat_despite_active | — | 13.5% (AZ1b also 13.5%) |
| low year-confidence | 16.0% | 12.0% |

```
AZ1c posterior tau (both P and E), all 8 chains:
  lag_P_tau[0]: mean 1.37, chain means 1.31-1.40
  lag_P_tau[1]: mean 1.41, chain means 1.40-1.42
  lag_E_tau[0]: mean 1.46, chain means 1.46-1.46
  lag_E_tau[1]: mean 1.47, chain means 1.46-1.47
  (tau_cap = 1.5 for reference; observed max draw: 1.4993/1.4999)
```

tau is **pinned right against the 1.5 ceiling** for both P and E, on every chain. The areas
that wanted to diverge sharply under AZ1b still want to — capping the mechanism that lets
them do so doesn't remove the underlying near-tied likelihood, it just relocates where the
disagreement shows up. And it did relocate, not disappear: the **population-level**
`lag_E_mu_logit` parameters, which were "fine to borderline" under AZ1b, are now themselves
badly converged (r-hat 1.24–1.32, ESS bulk 19-46) — worse than AZ1b's population-level
parameters ever were. Capping how far an individual area can move pushes some of that same
tension into the shared population mean instead, since a capped area with genuinely
conflicting data can no longer resolve the conflict locally and instead pulls on `mu_logit`.

**One prediction that did NOT hold up, checked directly rather than assumed:** the docstring
for `AZ1c` predicted a flexibility cost to AZ1b's flagship win, LSOA E01033711's multi-spike
tracking. Checked by comparing posterior-mean z directly:

```
year   AZ1b     AZ1c
2012     0.89    -0.96
2013     1.94     5.23
2014   201.65   207.51
2015    70.59    72.11
2016     8.80     6.68
2017     6.83     7.26
2018   231.05   233.95
2019    40.60    34.32
2020     3.95    -1.90
2021    67.70    69.80
```

The multi-year spike pattern is essentially unchanged. Read: z's shape is anchored more by
the zero-sum census constraint and the likelihood directly than by the lag-weight mechanism's
own flexibility — capping tau constrained lag_weights, but z found a similar solution anyway.
So the predicted cost didn't materialize *here*, even though the sampling got measurably
worse through a different channel (population-parameter disagreement).

### Verdict
**Rejected.** A hard ceiling on tau is not the fix — it doesn't remove the genuine per-area
ambiguity, it just moves where the unresolved tension shows up (from per-area `raw_offset` to
population-level `mu_logit`), and does so while making both max r-hat and min ESS worse than
the already-accepted AZ1b. This directly extends the existing finding from the failed
`tau_sigma=0.5` prior-tightening attempt: shrinking an area's freedom to diverge, whether via
a softer prior or a hard cap, fights the data instead of resolving it, because the ambiguity
is a genuine property of the data (two lag years that fit about equally well), not an
artefact of too much prior freedom. **AZ1b's existing "accept and report via 8 chains"
approach (`hierarchical_mode_summary`) remains the best available treatment** — confirmed by
elimination, not just by not having tried harder.

### Follow-up: is the ambiguity actually all genuine? Checked directly, and no — it's mixed

Requested specifically: dig deeper into the empirical cause rather than stop at "genuine hard
multimodality." Every prior check characterized WHICH mode each chain lands in, but never
asked how much WORSE (in actual log-likelihood) the losing mode is — a near-zero gap is real
epistemic ambiguity (both explanations fit about equally well, no amount of sampling resolves
it); a large gap means some chains are stuck in a genuinely worse solution, which IS a
sampling problem more chains/better initialization could fix.

AZ1b's `P_like`/`E_like` are plain observed StudentT RVs (not `pm.Potential`-based mixtures),
so `pm.compute_log_likelihood` already populates a real per-(area,year) log-likelihood array.
For every area with chains split across genuinely different dominant lag categories (not just
elevated r-hat — actual disagreement on which category each chain's draws concentrate on),
computed each chain-group's mean total log-likelihood (summed over years) and compared:

```
lag_P: 15 areas with genuine chain-group disagreement
  median gap: 6.35 nats   |  13.3% essentially tied (<2 nats)  |  26.7% decisively split (>10 nats)
lag_E: 19 areas with genuine chain-group disagreement
  median gap: 2.98 nats   |  31.6% essentially tied (<2 nats)  |  10.5% decisively split (>10 nats)

worst offenders (both sources):
  E01033711:  P gap=16.47 nats,  E gap=29.70 nats   <- this round's OWN flagship spike-tracking example
  E01002703:  P gap=16.82 nats
  E01035656:  E gap=17.27 nats
  E01002800:  P gap=11.07 nats
  E01002702:  P gap=11.98 nats,  E gap=4.44 nats
```

**Two genuinely different phenomena were being lumped together as "hard multimodality."**
Roughly a third of lag_E's flagged areas (and a smaller but real fraction of lag_P's) ARE
near-tied (gap < 2 nats) — real epistemic ambiguity, exactly as previously characterized, not
resolvable by more sampling. But a meaningful minority (10-27%, worse for P than E) show gaps
of 10-30 nats between chain groups — in probability terms, `e^30` is not a close call, that's
a decisively better solution some chains simply never found. That is a **convergence
failure for those specific areas**, not genuine ambiguity, and more/better sampling could
plausibly fix it.

**The single worst offender by this measure is E01033711 — this round's own headline
spike-tracking win**, with the largest gap for BOTH P (16.5 nats) and E (29.7 nats) of any
area checked. Its widely-quoted "6/8 chains same-year, 2/8 chains 1yr-lag" split (~75/25 mass)
is not obviously two comparably-good stories at all — 2/8 chains may simply be stuck in a
substantially worse mode. This doesn't overturn the area's headline result (the majority-mode
z estimate, backed by 6/8 chains, is very likely still the right one), but it does mean the
~25% "1yr-lag" mass quoted via `hierarchical_mode_summary` for this specific area should NOT
be read as "a quarter of the real posterior mass genuinely favours this alternative" — it may
just be un-converged chains, an important correction to how that number gets used downstream.

**Practical implication, not yet acted on**: this suggests a cheaper, more targeted fix than
either tried so far (prior-tightening or hard-capping tau, both of which fight the genuine
ambiguity and made things worse) — for the specific areas with LARGE log-likelihood gaps,
better initialization or more chains might let the stuck minority find the already-known
better mode, improving r-hat without sacrificing any real flexibility. Untested this round;
flagged as the most promising next step for AZ1b specifically if it's revisited.

### AZ1b → AZ1d — remove E's lag mechanism entirely: a real, clean win

Motivated directly by the asymmetry found at every step of this round (AZ1a's dramatically
bimodal `lambda_weights_E` vs. much milder `lambda_weights_P`; AZ4's `sigma_noise_E`
collapsing while `sigma_noise_P` stayed healthy; the log-likelihood-gap check immediately
above finding E's flagged areas are more often near-tied than P's): built `AZ1d`, identical to
AZ1b except E is compared directly against same-year `z` (`_build_planning_likelihood_simple`,
AZ0a's plain likelihood) instead of going through the hierarchical lag mechanism. P keeps
AZ1b's hierarchical lag unchanged.

| | AZ1d | AZ1b | AZ0a |
|---|---|---|---|
| max r-hat | **1.036** | 1.188 | 1.006 |
| min ESS (bulk) | **169** | 29 | 3180 |
| n_bad_rhat | 6 | 10 | 0 |
| frac_flat_despite_active | **7.5%** | 13.5% | 11.5% |
| low year-confidence | **7.0%** | 16.0% | 4.0% |
| plan_cov_90 / ben_cov_90 | 0.869 / **0.909** | 0.854 / 0.858 | 0.947 / 0.911 |

**A clear win on every axis, not a tradeoff.** min ESS improved ~5.8x (29 → 169), max r-hat
dropped from 1.19 to 1.04. `frac_flat_despite_active` (7.5%) is better than AZ0a's own
baseline (11.5%) — removing E's lag didn't just fix convergence, the resulting model attributes
real signal to actual years BETTER than the un-lagged baseline, presumably because P's
lag still does useful reallocation work while E's no-longer-ambiguous same-year comparison stops
fighting it. `ben_cov_90` improved to 0.909 — near-perfect nominal coverage, the best of any
AZ1-family model checked. The remaining 6 bad-r-hat parameters are now entirely within P's own
(milder, previously-known) lag hierarchy — exactly the asymmetry predicted, not a new problem.

**Spike-tracking spot-check — the flagship win survives, even strengthens**:
```
E01033711 (D=634), AZ1d vs AZ1b:
  2014:  265.31 vs 201.65   (real spike -- both track it, AZ1d MORE confident)
  2018:  268.36 vs 231.05   (real spike -- both track it, AZ1d MORE confident)
  2016:    0.72 vs   8.80   (quiet year -- AZ1d settles closer to zero)
  2019:    6.44 vs  40.60   (quiet year -- AZ1d settles closer to zero)
```
E01033711's two genuine BEN spikes (2014, 2018) — the example used throughout this whole round
to justify AZ1b's per-area lag flexibility — are tracked just as well, if not better, without
E ever having its own lag mechanism. This makes sense in hindsight: the area's total change is
still exactly pinned by the census constraint, and P's own lag flexibility plus z's zero-sum
freedom apparently already carry enough of the reallocation work; E's lag mechanism was mostly
contributing ambiguity, not irreplaceable signal. E01002703 shows one genuinely reshuffled year
(2020: 4.4 vs 71.8) — worth a closer look before treating this as unconditionally cost-free,
but not alarming on its own.

**Verdict: accepted.** `AZ1d` should replace `AZ1b` as this family's lag-pooling design —
strictly better convergence, better practical flatness, and no evident loss on the one
flagship example this round has repeatedly used to justify the lag mechanism's value. Revisit
Phase 4 (AZ4) with `AZ1d`'s asymmetric lag design in place of AZ1b's symmetric one, since AZ4's
own diagnosed leakage mechanism traces specifically to E's lag ambiguity (see the
az4-diagnostics artifact) — this fix targets that exact channel more directly than AZ4b's
tau-cap attempt does.

### AZ1d's own residual ESS/r-hat, dug into properly: what's left after removing E's lag

The central question this was asked to answer: **can a lag term be included at all without
compromising reliability?** AZ1d still has 6 named-scalar bad-r-hat params and cell-level
problems concentrated entirely in P's own lag hierarchy (`z`: 3.6% of cells bad, down from
AZ1b's ~26%; `lag_P_lambda_weights`: 6.2% of cells, `lag_P_raw_offset`: 6.5%). Dug into this
with the same tools used on AZ1b and AZ4 — mode summary, log-likelihood-gap check, and
autocorrelation — rather than stop at "P has some residual ambiguity too."

**1. The area count roughly halved, and it's mostly the SAME areas, not new ones.**
`hierarchical_mode_summary` flags 16 areas for `lag_P_lambda_weights` in AZ1d, down from 34 in
AZ1b. Cross-referenced directly: **13 of those 16 (81%) were ALSO flagged in AZ1b** — a
persistent hard core, not an artifact of the architecture change. The other 21 of AZ1b's
original 34 P-flagged areas were resolved simply by removing E's lag — i.e. most of P's
apparent ambiguity in AZ1b was actually **E's genuine ambiguity leaking into P's convergence**
via their shared `z`, not real P-lag ambiguity in its own right. Only 3 areas are newly
flagged in AZ1d (likely just run-to-run noise, not investigated further).

```
persistent core (13):  E01000958, E01001716, E01002702, E01002735, E01002786, E01002800,
                        E01002802, E01033700, E01033711, E01035646, E01035649, E01035656,
                        E01035708
resolved by removing E-lag: 21 areas
newly appeared: 3 areas (E01002794, E01033488, E01035655)

median |D|, persistent core:            410
median |D|, resolved-by-removing-E:     170
median |D|, all 200 areas:               54
```

The persistent core's median |D| is 7.6x the dataset median — this is exactly the areas with
large real spikes, where "which year gets the credit" is a meaningful question in the first
place. The 21 resolved areas had much smaller, more dataset-typical |D| — consistent with
their apparent ambiguity having been cross-source noise, not a real P-lag decision.

**2. Even within the persistent core, most of it looks like a SAMPLING problem, not
irreducible ambiguity** — checked via the same log-likelihood-gap method used on AZ1b (compare
actual total log-likelihood between the chain-groups that commit to different lag
categories):

```
E01033711:  gap=35.23 nats   <- by far the worst, same area that topped this metric in AZ1b too
E01035708:  gap=15.77 nats
E01035656:  gap=14.57 nats
E01033700:  gap= 8.62 nats
E01002735:  gap= 8.07 nats
E01002800:  gap= 6.50 nats
E01035649:  gap= 0.60 nats   <- genuinely tied
E01002702:  gap= 0.43 nats   <- genuinely tied
E01035646:  gap= 0.11 nats   <- genuinely tied
E01002802, E01002786, E01000958, E01001716:  all chains agree on dominant category
                                              despite flagged r-hat (soft/mixed disagreement,
                                              not hard multimodality)
```

Only 3 of the 13 (E01035649, E01002702, E01035646) are genuinely, decisively tied — the
irreducible kind of ambiguity no amount of sampling fixes. 6 of the 13 show large, decisive
gaps — chains stuck in a clearly worse mode, a convergence failure. The other 4 show mild,
soft disagreement (no chain purely commits to one category) rather than hard multimodality.
**E01033711 is, again, the single worst-converged area in this entire investigation** — it
topped this same metric for both P and E in AZ1b, and now tops it again for P alone in AZ1d.
This is a specific, identifiable, recurring problem area, not a generic property of "areas
with lag ambiguity."

**3. Autocorrelation for the worst-ESS scalars (`lag_P_mu_logit[1]`, `lag_P_tau[0]`) decays
properly** — 0.65/0.59 at lag 1, down to ≈0 by lag 20-50 on every chain. This rules out slow
within-chain mixing. What's left is between-chain disagreement on the exact population-level
value: `mu_logit[1]` chain means range 1.64-2.01 (≈19% of its mean), `tau[0]` chain means range
3.48-4.05 (≈15%) — the same "shallow basin" signature already seen in AZ2's/AZ3's/AZ4's other
collapsed scalars, not AZ1b's harder disconnected-mode signature. Consistent with the milder
magnitude here (ESS in the 100s-1000s, not the 10s-40s AZ1b's E-driven problems produced).

**Answer to "can we include a lag term at all without compromising reliability": yes, with a
quantified, honest caveat, not an unqualified yes.** A lag term on P alone:
- Leaves 93.5% of areas (187/200) with no detectable lag-driven convergence problem at all.
- Of the remaining 13, only 3 (1.5% of the dataset) show genuine, irreducible ambiguity —
  areas where the data itself cannot distinguish which year deserves credit, no matter how
  well the model samples.
- The other 10 (5% of the dataset) look like a **sampling problem, not a modelling limit** —
  large log-likelihood gaps between chain-groups mean a better mode exists and some chains
  simply haven't found it. This is the same conclusion reached for AZ1b, now confirmed to
  persist in AZ1d's cleaner setting rather than being an artifact of E's larger-scale
  contamination.
- Total per-area change is unaffected regardless (always exactly pinned to the census
  constraint by construction) — the only thing ever at stake is WHICH YEAR within an already-
  identified minority of high-|D| areas gets the credit.

### Two follow-up fix attempts, tried and both fully resolved (one accepted, one rejected)

**Attempt 1 — more chains (8→16) + informed initialization for the flagged areas — accepted,
a clean success that also validates the log-likelihood-gap methodology itself.** Seeded ALL 16
chains' `lag_P_raw_offset` toward the empirically better-supported lag category (from the gap
check above) for the 8 areas with a large decisive gap, then sampled normally (no other model
change). Checked per-area chain agreement afterward, split by the gap check's own
classification:

```
area          gap-check class     gap     majority fraction after seeding (16 chains)
E01033711     2b-decisive         35.23   100% (16/16 agree)   <- this round's flagship example
E01002800     2b-decisive          6.50    94%
E01002735     2b-decisive          8.07    88%
E01035656     2b-decisive         14.57    81%
E01033700     2b-decisive          8.62    69%
E01035708     2b-decisive         15.77    50%  <- exception, see below
E01035649     2a-tied              0.60    69%
E01002702     2a-tied              0.43    50%
E01035646     2a-tied              0.11    62%
```

The pattern holds almost perfectly: areas the gap check called "decisive" mostly resolved to
near-unanimous agreement once seeded there (up to 100% for the worst offender, E01033711);
areas the gap check called "genuinely tied" stayed split even when seeded — exactly correct,
since there's no "right" mode to seed toward for those. **This cross-validates the
log-likelihood-gap method itself**: it isn't just a plausible heuristic, seeding based on its
classification produces exactly the outcome the classification predicts. Named-scalar ESS also
jumped 5-6x across the board (`sigma_plan` bulk ESS 269→1725, `lag_P_tau[0]` 284→1604), zero
divergences. One clean exception: `E01035708` (a genuine 3-way split, not 2-way) stayed at 50%
majority despite a large gap — a 2-category seed may be the wrong shape of intervention for a
3-mode area; not investigated further this round.
**Not yet folded into a permanently-registered model** — this used 16 chains and a manually
constructed `init_mean` dict via a one-off script
(`results/traces/AZ1d_16chain_informedinit.nc`), not a new `AZ1*` class. Promoting it to a
standing default would mean hard-coding the flagged-area list and their seed categories into
`AZ1d.sample()`, which feels premature before deciding whether this becomes the production
lag design.

**Attempt 2 — horseshoe-style local/global tau (`AZ1e`) — rejected, and instructively so.**
Built `_build_hierarchical_lag_horseshoe`: `tau[a,k] = global_tau[k] * local_scale[a,k]`,
`local_scale ~ HalfCauchy(1)`, replacing AZ1d's one shared `tau` with a per-area value —
the "outside the box" proposal aimed at letting the ~13-area core diverge further without
constraining the other 187 areas at all (unlike AZ1c's/AZ4b's uniform cap, which constrained
everyone identically and failed both times).

| | AZ1e (horseshoe) | AZ1d |
|---|---|---|
| max r-hat | 1.101 | 1.036 |
| min ESS | 58 | 169 |
| divergences | **17** | **0** |
| low year-confidence | 48.5% | 7.0% |

Worse on every axis, and **the first divergences seen anywhere in this entire AZ1/AZ4
investigation** (scattered 1-5 per chain across all 8 chains — the "scattered, not clustered"
signature of a step-size struggle in an extreme-tail region, not a single funnel-at-low-sigma
location). Confirmed the exact predicted risk, not a surprise: `local_scale` ranged from
2e-6 to **3.09 million** — an unregularized `HalfCauchy`'s classically unbounded tail, exactly
the horseshoe geometry warning already on record in this codebase's own reference material
(pymc-extras skill, `troubleshooting.md`'s "Horseshoe Prior Challenge"). Year-confidence
(48.5% low-confidence) is the worst of any AZ-family model checked this entire round — worse
than AZ4's 38.0%.
**Verdict: rejected**, but a useful negative result for the diagnostic guide (see
`docs/ess-rhat-diagnostic-guide.md`): letting per-area flexibility vary via an unregularized
heavy-tailed local scale can introduce a NEW, worse pathology (Pattern 5-adjacent: a scale
parameter's own tail exploding) rather than fixing the one it targeted. A regularized
horseshoe (with a slab scale bounding the local multiplier) is the standard literature fix for
this exact failure mode and was deliberately not tried on this first pass (see the builder's
own docstring) — worth trying before concluding the whole local/global approach is dead, but
the plain version tested here is not viable.

**Net read on the two proposals**: the "obvious," cheap, model-unchanged fix (more chains +
informed init) worked cleanly; the more architecturally sophisticated fix (horseshoe) failed
by introducing a new, worse pathology. Consistent with this round's recurring lesson (first
seen in AZ2's top-boost simplification): prefer the smallest change that targets the diagnosed
mechanism directly over a more complex reparameterisation, and test complexity increases
skeptically rather than assuming they're strict upgrades.

### Follow-up: which areas are genuinely multimodal, and does that explain everything else?

Requested directly, on the baseline 8-chain `AZ1d.nc` trace (not the 16-chain informed-init
variant): characterize the genuinely-tied areas visually, then check whether every OTHER
residual r-hat/ESS problem — cell-level `z` issues in non-flagged areas, and the 6 bad
named-scalar params — is downstream of the same handful of areas or comes from elsewhere.
Full artifact: `results/artifacts/az1d_multimodal_deepdive.html`.

**Only 4 of the 16 `lag_P_lambda_weights`-flagged areas are genuinely, irreducibly
multimodal**: `E01035649` (gap 0.60 nats, 7v1 chains, mild — z bands overlap heavily),
`E01002702` (gap 0.43 nats, 7v1 chains — a ~520-unit planning spike with no matching BEN
signal, so 2yr-lag vs 1yr-lag are both plausible), `E01035646` (gap 0.11 nats, 6v2 chains,
"soft" — no chain is even purely committed, more a smooth spread than a hard mode split), and
`E01035708` (the informed-init experiment's one clean exception — turns out to be a genuine
**3-way** split, 1/5/2 chains across same-yr/1yr-lag/2yr-lag, each implying a materially
different year for this area's D=637 change; a 2-category informed seed was structurally the
wrong shape of intervention for a 3-mode area, which is why it alone didn't respond to
Attempt 1). The other 12 lambda-flagged areas are Pattern 2b (resolvable, as Attempt 1 already
demonstrated), not genuine ambiguity.

**Cross-referencing `hierarchical_mode_summary` (16 areas, flags on `lag_P_lambda_weights`
r-hat) against `z_identifiability_summary` (14 areas, flags independently on per-cell `z`
r-hat) finds exactly one area outside the lambda-flagged set**: `E01002703` (max r-hat 1.017).
Checked directly — its own `lag_P_lambda_weights` r-hat is 1.0013-1.0028, just under the 0.01
flagging threshold, i.e. the same phenomenon sitting right at the edge of the cutoff, not a
second mechanism. **183 of 200 areas (91.5%) show no detectable r-hat/ESS issue of any kind**;
every one of the 17 that do traces to P's lag-category ambiguity. AZ1d has no independent,
unrelated source of non-convergence.

**Scalar check, tested rather than assumed**: for each of the 8 chains, counted how many of
the 16 lag-flagged areas that chain lands in the minority mode for, then correlated that count
against the chain's own mean `lag_P_mu_logit[1]` and `lag_P_tau[0]` (the two worst-converged
scalars: r-hat 1.04/ESS 169 and r-hat 1.03/ESS 284 respectively). **r = -0.01 and r = +0.26** —
negligible to weak, not the signature of 1-2 "stuck" chains single-handedly dragging the
population parameters. Consistent with the earlier autocorrelation finding (mixing decays
fine within-chain): the scalar instability is a genuinely diffuse consequence of the population
level having to reconcile 184 well-behaved areas against a *collective*, not individually
localized, pull from the ~16-area lag-ambiguous minority — a structural effect of hierarchical
pooling in their presence, not a single-chain or single-area failure.

### AZ1d → AZ1f — marginalize P's lag likelihood instead of mean-mixing it — rejected, badly

Follow-up direct check: even after Attempt 1's informed init (16 chains, seeded), `lag_P_mu_logit`
stayed at r-hat 1.02-1.03 (chat transcript, not reproduced in this doc) — more chains alone
cannot get every named scalar under 1.01 while genuine per-area ambiguity persists under the
CURRENT likelihood. The hypothesis tested: `_build_hierarchical_lag`'s likelihood mean-mixes
lag categories (`P_mean = sum_k lambda_k * z_shifted_k`, one StudentT around that single
blended value) — a known anti-pattern for representing genuine discrete-category uncertainty,
since it scores every observation against a value no individual lag category actually predicts.
Built `AZ1f`: identical hierarchical `lambda_weights` construction to AZ1d, but P's likelihood
marginalizes over lag categories instead (`_build_hierarchical_lag_marginalized` +
`_build_planning_likelihood_marginalized_lag`, `log p(P_obs) = logsumexp_k(log lambda_k +
logT(P_obs; z_shifted_k, sigma))`, via `pm.Potential`). Explicitly flagged before running this
as a hypothesis to test, not an assumed fix — a toy numerical check (chat transcript) had
already shown the simple "mean-mixing creates a likelihood valley in lambda" framing doesn't
hold up as a general claim, so the actual basis for trying it was the generative-story argument
(mixture of densities vs a blended mean), not a proven geometric mechanism.

**Result: substantially worse on every axis, real-data run:**

| | AZ1f (marginalized) | AZ1d (mean-mixed) |
|---|---|---|
| max r-hat | 1.174 | 1.036 |
| mean r-hat | 1.058 | 1.025 |
| min ESS (bulk) | **33** | 169 |
| low year-confidence | **33.0%** (66/200 areas) | 7.0% (14/200) |
| `hierarchical_mode_summary` flagged areas | **200/200** | 16/200 |
| `sigma_plan` mean (r-hat) | **0.72** (1.17) | 5.01 (1.02) |
| `lag_P_tau` mean (r-hat) | **~0.09-0.11** (1.00) | 3.74-4.12 (1.01-1.03) |

Every named scalar got worse (`lag_P_mu_logit` r-hat 1.06-1.07 vs AZ1d's 1.02-1.04, `sigma_ben`
1.04 vs 1.02), and `hierarchical_mode_summary` — previously a targeted diagnostic that flagged a
specific 16-area minority — now flags literally every area in the sample, i.e. the model no
longer has a coherent, stable per-area lag-category assignment ANYWHERE, not just in the
previously-ambiguous core.

**Mechanism, confirmed directly rather than just inferred from the aggregate numbers**:
`sigma_plan` collapsing to 0.72 (from 5.01) alongside `lag_P_tau` collapsing to ~0.1 (from
~3.7-4.1, with *excellent* ESS ~8500-9100 and r-hat 1.00 — a confident, converged collapse, not
noise) is the signature of a specific failure: marginalizing at the observation level doesn't
just remove the *unwanted* commitment pressure mean-mixing was creating, it also removes the
*wanted* consistency-across-years pressure that gave the shared per-area `lambda_weights` any
meaning in the first place. Under mean-mixing, ALL of an area's observations must be explained
by the SAME blended lag commitment; under the marginalized/logsumexp form, each individual
`(area, year)` term in the sum can effectively pick whichever category fits IT best, independent
of what other years in that area picked, as long as that category's `lambda_weights` entry is
non-zero. With ~10 observations/area and only 3 categories, this is enough freedom to fit almost
every point opportunistically, which is exactly what let `sigma_plan` shrink toward zero (near-
perfect fit via per-point category cherry-picking) rather than reflecting genuine observation
noise. The spike-tracking plot (`results/artifacts/az1f_spike_tracking.png`) confirms the
practical cost directly: several areas now badly UNDER-track genuine large spikes (e.g.
`E01033491`: real P_obs=762 in 2016, posterior z reaches only 26; `E01035656`: P_obs=498,
z reaches only 5) — the model is worse at its most basic job, not just less well-converged.

**Checked the genuinely-multimodal areas specifically** (the whole reason this was tried):
all 4 got worse, not better — `E01035708`'s max z r-hat went from 1.86 to **2.36**, `E01002702`
from 1.28 to 1.35, `E01035646` from 1.04 to 1.01 (only one of the four nominally improved, and
by a hair). Marginalizing did not remove their genuine ambiguity cleanly; it just added a new,
worse, dataset-wide pathology on top of it.

**Verdict: rejected.** The generative-story argument for marginalizing (mixture of densities vs
a blended mean) is not wrong in the abstract, but it ignored a second role mean-mixing was
quietly serving: forcing a single, shared per-area lag commitment across all of that area's
years, which is exactly the consistency that keeps the hierarchy identified. Removing that
consistency traded a contained, ~8%-of-areas ambiguity problem for a dataset-wide one. This is
the third distinct fix attempt in the AZ1 family to fail (after the tighter tau prior and the
horseshoe), and the third to fail via a genuinely different mechanism — see
`docs/ess-rhat-diagnostic-guide.md` for the updated pattern.

### E01002702, characterized properly via `docs/multimodality-characterization-guide.md`

The "Follow-up" section above reported E01002702 as a clean 7-vs-1-chain, two-mode split
(gap 0.43 nats) from the baseline 8-chain `AZ1d.nc` trace. Ran the new characterization guide's
pipeline on it directly (plus a fast light-pass on `E01035649`/`E01035646`), reusing the
already-existing `AZ1d_16chain_informedinit.nc` trace rather than resampling — and the fuller
picture is meaningfully different from the two-mode summary above, though it doesn't overturn
the "genuinely tied" conclusion.

**Step 1 (artifact checks) — not label-switching, not a collapsible ridge.** Confirmed two
ways: (a) structurally, `_build_hierarchical_lag`'s categories pull from *different* slices of
`z_padded` (a 0/1/2-year shift each), so there is no transformation that swaps two categories
while leaving `P_mean` unchanged — no exchangeable-component symmetry exists in this
construction at all; (b) the raw, *unconstrained* `lag_P_raw_offset[area,:]` (not just the
softmax-squashed weights) shows the competing chain-groups' means ~2-3 SDs apart, i.e. real
separation in the parameter actually being sampled, not a ridge that softmax saturation makes
look separated.

**Step 2 (mode mapping) — a third, minor mode exists that the 8-chain trace never found.**
Checking the 16-chain trace (which wasn't seeded toward a specific category for this area,
since it wasn't one of the 8 large-decisive-gap areas the seeding experiment targeted) reveals
one chain sitting 100% in **category 0 (same-year, no lag at all)** — never seen in the 8-chain
run. This mirrors `E01035708`'s already-documented genuine 3-way split above almost exactly
(same-year / 1yr-lag / 2yr-lag), which is worth noting as a cross-check: **two independent
areas now show this same same-year/1yr/2yr triple structure**, suggesting it's a recurring
feature of how this dataset's lag ambiguity actually looks, not a one-off curiosity specific to
one area.

`E01002702`'s three modes, decoded against the raw data (P_obs 2012=161, 2017=57; E_obs
2016=520, 2019=236, 2021=227 — note the existing "Follow-up" section above describes this as "a
~520-unit **planning** spike with no matching BEN signal"; checked directly against
`data['P_obs']`/`data['E_obs']` and that's backwards — **the 520 is BEN's spike** (E_obs[2016]),
and P's own spike is the much smaller 57 in 2017; worth a correction if that phrasing is reused
elsewhere):

| Mode | Chains (16-chain run) | Story | Total logp (P+E) vs best |
|---|---|---|---|
| cat2 (2yr lag) | 8 | 2017's P=57 ← a 2015 completion | -7.47 nats |
| cat1 (1yr lag) | 6 | 2017's P=57 ← a 2016 completion | 0 (best) |
| cat0 (same-year) | 1 | both P spikes (2012, 2017) same-year, no lag | -15.86 nats |

cat0 is real (well-separated, structurally coherent — not an artifact) but decisively
low-probability: it fits `P_obs` almost exactly (z[2012]≈165, z[2017]≈58) but fits BEN's much
larger 2016/2019/2021 spikes substantially worse than cat1/cat2 do, and the net (P+E) fit loses
by ~16 nats. Only 1 of 16 chains has ever found it, so no within-mode r-hat/ESS is computable —
reported qualitatively, not chased with a dedicated resampling run given how decisively minor it
is.

**A genuine methodological finding, not just a numeric footnote**: recomputing the
log-likelihood-gap check (§4 of the ESS guide) using `P_like` alone (the literal recipe, since
`lag_P` only directly appears in `P_like`) reproduces the original **0.32-0.43 nat, genuinely
tied** result on both the 8-chain and 16-chain traces — no real disagreement between the two
runs. But `raw_offset`'s category choice also pulls the *shared* `z` in different directions
(z[2016]≈59 vs z[2015]≈53), and that same `z` feeds `E_like` too (BEN is same-year, no lag, in
AZ1d) — so the FULL joint (`P_like`+`E_like`) gap is **~7.5 nats, a real but non-decisive lean
toward cat1**, not tied at all. **The gap-check recipe as literally written can understate how
resolved two modes are when the flagged variable has indirect effects on other likelihood terms
via a shared latent** — worth checking the fuller joint measure, not just the one likelihood
term the flagged variable is most directly attached to, whenever such a shared-latent path
exists (folded back into `docs/ess-rhat-diagnostic-guide.md`).

**Step 4 (mass estimation) — a local Laplace/bridge approximation on `raw_offset[area,:]`
alone (2D, no PyTensor/transform engineering needed) gives `P_like`-only mass ≈ 38%/62%
(cat1/cat2)**, using each mode's own pooled-draws empirical mean/covariance as the Gaussian fit.
Adding `E_like` to the *same* 2D-local calculation broke it badly (99.9%/0.1%) — a real,
instructive failure: `E_like`'s ~7.5-nat pull operates through `z`'s own (9-dimensional,
zero-sum-constrained) spread, not through `raw_offset`, so crediting that full height
difference to only 2 dimensions' worth of volume massively overstates confidence — the same
"exponentiating a bare density gap" mistake the characterization guide warns against, smuggled
back in through an under-scoped local approximation. **Net: 38%/62% is a documented, honest
lower/upper bound for cat1/cat2 respectively (P-alone, trustworthy); the true combined split is
further toward cat1 by an unquantified amount.** Getting a precise combined number needs full
bridge sampling over `z`'s actual (`ZeroSumTransform`-handled) unconstrained space — not run,
given this is one cell's year-attribution in one of 200 areas and the total `z` sum for this
area is unaffected regardless of which mode is true.

**`E01035649`/`E01035646` — a different, milder phenomenon than "hard multimodality" at all.**
Checked switch counts (how many times each chain's dominant lag category changes across its
1500 draws) before assuming these behave like `E01002702`: **`E01002702`'s 8 chains show
*zero* switches each** (genuine, permanently-trapped hard multimodality) — but `E01035649`'s 8
chains switch 26-55 times each, and `E01035646`'s switch 82-108 times each. Both are genuinely
bimodal at the individual-draw level (confirmed via `_detect_modes`' KDE peak check on the
pooled marginal — real separated peaks, not one broad hump), but **every chain successfully
round-trips between both peaks within its own run**, rather than getting stuck in one. This
means the *ordinary pooled posterior mean* is already a legitimate direct mass estimate for
these two (ESS 274/169 on the contested category's weight — acceptable, not razor-precise):
`E01035649` ≈ 45%/55%, `E01035646` ≈ 43%/57%. No bridge sampling needed for either — a much
cheaper case than `E01002702`'s hard-trapped one, and worth telling apart from it before
assuming the same treatment applies (folded back into `docs/multimodality-characterization-guide.md`
as a distinct sub-pattern; also flagged for `docs/ess-rhat-diagnostic-guide.md`'s Pattern 2).

Both areas' underlying ambiguity checks out as substantively real, not artifactual: `E01035649`
has genuine BEN corroboration in *both* contested years (E=192 in 2014, E=102 in 2015 — the
data honestly can't tell which one the P=212 registration belongs to); `E01035646`'s contested
years (2019, 2020) have no corroborating signal from either source at all — an irreducible
ambiguity between two silent years, a different flavor of the same underlying phenomenon.

**Practical takeaway, unchanged from the "Follow-up" section above**: total `z` per area
remains exactly pinned by the census constraint regardless of which mode is true in all three
areas — what's newly established here is a properly-caveated (not over-claimed) picture of
*how* ambiguous the year-attribution actually is for `E01002702` specifically, and confirmation
that `E01035649`/`E01035646` are a materially easier, already-resolved-enough case that doesn't
need the same investment.

---

## AZ2 → AZ2b — smooth top-boost ramp

### Symptom (already diagnosed pre-this-document)
Real-data run: max r-hat 1.05 (3 bad params), min ESS 47 (bulk; tail was much better, 185).
Deeper investigation (already done, see `az-family-work-plan.md` Phase 2 follow-up) ruled out
ordinary slow mixing (autocorrelation near zero past lag ~20 for every chain) and ruled out a
simple pairwise ridge among `sigma_plan`/`sigma_ben`/`sigma_delta_top_boost` (all |r| < 0.12,
even in a 4000-draw follow-up run). Diagnosed instead as a small, persistent
**between-chain** disagreement: `sigma_delta_top_boost`'s 4 chain means sat at 28.5-30.3 for
the whole run, never converging toward each other. "More draws" was tried and gave a mixed
result (`top_boost` improved, `sigma_plan` got worse) — not a clean fix, and left the
mechanism itself unexplained.

### Fix tried
Built `_build_zero_sum_z_prior_top_boost_smooth` and a new model, `AZ2b`. Only change from
AZ2: `is_top` (a hard 0/1 step exactly at the 75th percentile of `|D|`) is replaced by a
smooth logistic ramp over `|D|`'s **rank percentile** (not `|D|` itself, to stay scale-free
across the dataset's wide 10-600+ range):

```
rank_pct[a]      = (rank of |D[a]| among all areas + 0.5) / n_areas
smooth_weight[a] = sigmoid((rank_pct[a] - 0.75) / 0.08)
sigma_delta[a]   = floor + k*|D[a]| + smooth_weight[a] * top_boost
```

Untested hypothesis being checked: a hard step means areas straddling the cutoff see
`sigma_delta` respond *discontinuously* to `top_boost` — exactly the kind of sharp threshold
that can produce a shallow, hard-to-mix ridge (a few boundary areas effectively voting on a
knife-edge). The smooth version adds no new sampled parameters — still exactly one extra
scalar (`top_boost`) — so any change in ESS isolates the discontinuity itself as the cause,
not some other side effect of a bigger reparameterisation.

### Result — positive, a real and fairly large improvement
| | AZ2 | AZ2b |
|---|---|---|
| max r-hat | 1.054 | **1.027** |
| min ESS (bulk) | 47 | **126** (2.7x) |
| frac_flat_despite_active | 9.0% | 8.5% |
| low year-confidence | 12.5% | 13.0% |
| plan_cov_90 / ben_cov_90 | 0.954 / 0.922 | 0.957 / 0.916 |

Per-scalar detail, AZ2b:

```
                        mean    sd  r_hat  ess_bulk  ess_tail
sigma_plan              5.81  0.20   1.02       180      1172
sigma_ben               7.77  0.25   1.03       126      1523
sigma_delta_top_boost   24.80 1.80   1.01       605      1958   <- was bulk=47 in AZ2

chain means (top_boost), AZ2b: [24.72, 25.03, 24.89, 24.70]   (range 0.33, ~1.3% of mean)
chain means (top_boost), AZ2  (for comparison): 28.5-30.3      (range 1.8,  ~6.1% of mean)
```

`sigma_delta_top_boost` itself — the parameter the original investigation flagged as worst —
improved roughly 13x in bulk ESS (47 → 605) and its between-chain spread tightened from ~6%
of its mean to ~1.3%. The overall model min ESS (126) is now set by `sigma_plan`/`sigma_ben`
instead, which is a **different, milder problem** than before (they were never the flagged
parameter in AZ2's original investigation, and 126/180 bulk ESS, while still below the 400
rule-of-thumb, is a different order of problem than 47).

`frac_flat_despite_active` and year-confidence are essentially unchanged (within noise of
AZ2's own numbers), confirming the smoothing didn't cost anything on the metric that matters
most for this model family — it targeted the ESS mechanism specifically without touching the
model's practical behaviour.

### Verdict
**Accepted as a real improvement, though not a complete fix.** The discontinuity-at-the-
boundary hypothesis held up under direct test: removing the hard step measurably loosens the
between-chain disagreement on exactly the parameter that had it. Residual min ESS (126, now
on `sigma_plan`/`sigma_ben` rather than `top_boost`) is a smaller, different, and lower-
priority problem than AZ2's original one — worth a note if AZ2 or AZ2b is chosen for Phase 4,
not worth blocking on now. **AZ2b should replace AZ2 as this family's better version of the
top-boost idea** unless a reason emerges to prefer AZ2's originally-validated spike-tracking
behaviour specifically (not expected, given the flatness/confidence numbers above are a wash,
but not independently re-verified via the spike-tracking plot in this pass — see Not Done
below).

---

## What this changes about Phase 4 (combine validated pieces)

`az-family-work-plan.md`'s Phase 4 plan was to combine "whichever lag-pooling design won"
with AZ2 and the floored AZ3 branch. This document's findings sharpen that:
- **Lag pooling: still AZ1b.** AZ1c was a genuine, reasonable attempt at AZ1b's open
  question and it lost — AZ1b's existing "accept and report" treatment stands, confirmed
  rather than merely un-improved-upon.
- **D-magnitude boost: AZ2b, not AZ2.** A real, verified improvement with no evident cost —
  swap it in.
- **AZ1a is fully superseded**, not a Phase 4 candidate; kept in the codebase as the
  documented negative result motivating AZ1b's design, per this codebase's convention of
  keeping instructive rejected models rather than deleting them (see M-family history).

---

## Phase 4 — AZ4, the combined model (built after this document's three fixes)

Built `AZ4`: AZ0a + AZ2b's smooth top-boost z-prior + AZ1b's area-hierarchical lag (both
sources) + AZ3's floored noise-mixture likelihood, applied to the LAG-CONVOLVED `P_mean`/
`E_mean` rather than raw `z` (a genuine new composition, not a copy-paste — AZ3 alone had no
lag structure). Full design rationale is in `AZ4`'s docstring in `models.py`. Sampled on the
real 200-area dataset, 8 chains (inherited from AZ1b/AZ1c's precedent).

**The composition-risk hypothesis flagged in AZ4's own docstring (AZ1b's lag ambiguity and
AZ3's signal/noise ambiguity compounding rather than cancelling) was checked directly, and it
partially holds — but not where the top-line diagnostics table would suggest.**

| | AZ4 | AZ1b | AZ2b | AZ3 | AZ0a |
|---|---|---|---|---|---|
| max r-hat | 1.168 | 1.188 | 1.027 | 1.004 | 1.006 |
| min ESS (bulk) | 31 | 29 | 126 | 1627 | 3180 |
| frac_flat_despite_active | **2.0%** | 13.5% | 8.5% | 3.5% | 11.5% |
| low year-confidence | **38.0%** | 16.0% | 13.0% | 24.5% | 4.0% |
| plan_cov_90 / ben_cov_90 | **0.843 / 0.796** | 0.854/0.858 | 0.957/0.916 | 0.912/0.849 | 0.947/0.911 |
| divergences | 0 | 0 | 0 | 0 | 0 |

At the level of this summary table, AZ4 looks merely "about as bad as AZ1b" (its worst
component) on max r-hat/min ESS — which would read as a wash, not a compounding effect. Two
things the summary table hides:

1. **`frac_flat_despite_active` is the best of every AZ-family model tested this round
   (2.0%)** — practically, AZ4 attributes real P/E signal to actual years better than any
   single piece alone. The three mechanisms (lag reallocation, top-boost headroom, and a
   noise branch to discount genuine outliers) are, in this one specific sense, complementary
   rather than redundant.
2. **That practical win comes with a real, and previously invisible, cost**: year-allocation
   confidence (38.0% low-confidence) is worse than the WORST single component (AZ3's 24.5%),
   not just "as bad as" one of them — more than a simple max-of-components would predict.
   Coverage calibration also degrades past any individual piece: `ben_cov_90=0.796` is the
   worst of any AZ-family model checked this round (should be ~0.90; AZ4 under-covers BEN
   observations meaningfully more than AZ1b, AZ2b, AZ3, or AZ0a do individually).

**Per-scalar breakdown shows exactly where the compounding lands, and it's asymmetric**:

```
                         mean    sd  r_hat  ess_bulk  ess_tail
lag_P_tau[0]              2.1   0.4   1.17        31        38
sigma_noise_E             34      4   1.16        33        69   <- was ESS>3600 in AZ3 alone
sigma_delta_top_boost   40.1   2.9   1.13        40        94   <- was ESS=605 in AZ2b alone
lag_E_mu_logit[0]        0.7  0.44   1.07        72       176
lag_E_tau[0]             3.6  0.57   1.07        79       178
rho_E                   0.81 0.015   1.06        80       256
...
sigma_noise_P           25.6   0.6   1.01      3321      5844   <- stayed healthy
sigma_plan             2.01  0.01   1.00      8510      4796   <- stayed healthy (AZ3's floor fix held)
sigma_ben               2.02  0.02   1.00      8574      7500   <- stayed healthy
```

`sigma_plan`/`sigma_ben` (AZ3's floor fix) are untouched by the combination — still pinned
tightly near the floor with excellent ESS, exactly as in AZ3 alone. But **`sigma_noise_E`
collapsed from AZ3's >3600 ESS to 33, and `sigma_delta_top_boost` collapsed from AZ2b's 605
to 40** — both of AZ2b's and (half of) AZ3's own fixes got measurably undone by the
combination, while AZ1b's own lag parameters stayed roughly at AZ1b's own (already-imperfect)
level. Notably it's specifically `sigma_noise_E`, not `sigma_noise_P`, that degraded — E was
already the more genuinely lag-ambiguous source in every prior AZ1-family diagnosis
(`lambda_weights_E`'s bimodality was the dramatic one; P's was always milder), so this
asymmetry is consistent with the mechanism, not a random artefact: E's own lag-category
uncertainty now propagates into which specific E observations look like signal vs noise,
since the noise-mixture's signal branch is E's lag-convolved mean, not raw z.

Checked directly, not assumed: chain means for the three degraded scalars are close together
(`rho_E` 0.803-0.816, `top_boost` 38.3-41.4, `sigma_noise_E` 31.5-36.6) with **zero
divergences** — this is the SAME "small persistent between-chain disagreement" (shallow
basin) signature as AZ2's and AZ3's original problems, not AZ1b's hard multimodality
re-appearing. The mechanism compounded existing shallow-basin geometry rather than
introducing new hard multimodality.

**Spike-tracking spot-check (E01033711, E01001774, E01033491) — practical wins preserved, but
visibly noisier**:

```
E01033711 (D=634):
  year   AZ4      AZ1b
  2012    28.08     0.89
  2013   -23.51     1.94   <- AZ4 introduces a spurious-looking dip AZ1b didn't have
  2014   248.04   201.65   <- both track the real spike; AZ4's is even larger
  2015   -54.84    70.59   <- AZ4 flips sign here; AZ1b stayed positive
  2018   283.02   231.05   <- both track the second real spike
```
The two genuine spikes (2014, 2018) are tracked by both, and AZ4's estimates are if anything
larger/more confident there. But AZ4 introduces negative dips in 2013/2015 that AZ1b kept
near zero — a visible, concrete instance of the extra degrees of freedom (lag + top-boost +
noise-branch, all simultaneously available) reallocating credit in ways the chains don't
fully agree on, consistent with the 38% year-confidence number rather than contradicting it.
E01001774 and E01033491 (AZ3's and AZ0a's respective motivating cases) are both reproduced
closely, with only minor year-by-year reshuffling.

**Verdict: real, mixed result — not a clean win, not a clean loss.** AZ4 is the best AZ-family
model tested this round on the metric this whole family exists to serve
(`frac_flat_despite_active`), and its practical spike-tracking wins survive. But it is also
the worst on year-allocation confidence and BEN coverage calibration, and it measurably undid
part of two of this document's own fixes (AZ2b's top-boost ESS win, and half of AZ3's noise-
mixture health) rather than merely inheriting AZ1b's pre-existing imperfection. The
composition-risk warning in `AZ4`'s docstring was justified, though not in the form the
summary diagnostics table alone would reveal — a reminder that comparing top-line r-hat/ESS
across combined vs. component models isn't sufficient; per-scalar breakdown was necessary to
see the actual effect. **Not yet a recommended finalist as-is** — see the follow-up options
below.

### Follow-up options, not yet chosen
1. ~~Investigate `sigma_noise_E` specifically~~ — done, see the `az4-diagnostics` artifact:
   traced to lag ambiguity (96% overlap between z-unstable and lag-unstable areas; 92% of
   top-quartile-D areas are lag-ambiguous), not a new independent mechanism.
2. **Tried: hard cap on tau within AZ4 (`AZ4b`), same `tau_cap=1.5` as AZ1c — negative,
   confirms AZ1c's verdict transfers.** Tested on its own terms (does capping the specific
   leakage channel into the shared scalars help, even though AZ1c lost on its own turf) rather
   than assumed to fail — it failed anyway, and worse than AZ4 itself:

   | | AZ4b (tau capped) | AZ4 |
   |---|---|---|
   | max r-hat | **1.459** | 1.168 |
   | min ESS | **15** | 31 |
   | frac_flat_despite_active | 1.0% | 2.0% |
   | low year-confidence | 36.0% | 38.0% |

   max r-hat and min ESS both got meaningfully worse (mirroring AZ1c vs AZ1b exactly);
   flatness and year-confidence moved marginally in AZ4b's favour, within noise. Capping tau
   is now confirmed to fail in BOTH the isolated (AZ1c) and combined (AZ4b) settings — not the
   right lever anywhere in this family. **Superseded by AZ1d (below): removing E's lag
   entirely, rather than constraining how far ANY area's lag can diverge, is the targeted fix
   that actually works, and does so more directly (it removes the specific asymmetric E-lag
   ambiguity that AZ4's own causal-chain analysis identified as the leakage source, rather
   than blunting all per-area lag divergence indiscriminately).**
3. **Recommended next step, not yet built**: rebuild the combined model using AZ1d's
   asymmetric lag design (P hierarchical, E same-year) in place of AZ1b's symmetric one —
   see AZ1b's own section above for AZ1d's full result (max r-hat 1.19→1.04, min ESS 29→169,
   best-in-family flatness). Deferred pending explicit direction to return to the AZ4 family
   (current priority is finishing the AZ1-family work).
4. **Still on the table if (3) doesn't fully resolve AZ4's remaining problems**: widen
   `sigma_noise_floor` specifically for AZ4 (untested whether 25 is still right once the
   signal branch is lag-aware); let the still-deferred LOO comparison be the tie-breaker; or
   report AZ4 as-is with an explicit "reliable for total change and which spikes are real, not
   which year" caveat.

### Ablation test: is it really "the lag-ambiguous minority leaking into the shared scalars"?

Everything above about AZ1d's/AZ4's worst-converged POPULATION-level scalars (`lag_*_mu_logit`,
`lag_*_tau`, and for AZ4 `sigma_noise_E`/`sigma_delta_top_boost` downstream of them) being a
diffuse consequence of the hierarchy reconciling the well-behaved majority against a "collective
pull" from a lag-ambiguous minority was, until this section, supported only by correlational
checks — a weak per-chain count correlation for AZ1d (r=-0.01/+0.26) and an area-overlap
statistic for AZ4 (96%/92%). Neither is a direct intervention. Tested directly instead, via two
new builders (`_build_hierarchical_lag_pinned`, `_build_fixed_lag`, `models.py`) and a driver
script (`results/scratch/az_ablation_investigation.py`, not a registered model — a one-off
ablation, matching this codebase's convention for investigation-only code):

- **DROP** — remove the flagged areas from the dataset entirely, resample. Confounded (also
  shrinks `n_areas`, changing what `mu_logit`/`tau` are estimated FROM), but cheap.
- **PIN** — keep all 200 areas' P/E data in the likelihood and zero-sum prior, but fix the
  flagged areas' own `area_logit` at their already-converged posterior mean instead of sampling
  it. Removes the confound: only the flagged areas' OWN unresolved posterior stops feeding
  `mu_logit`/`tau`, nothing else about the model changes.
- **EFIXED** (AZ4 only) — replace E's entire hierarchical lag mechanism (every area, not just
  the flagged ones) with one constant vector fixed at AZ4's own posterior mean. Tests whether
  it's E's lag mechanism's own sampling freedom, not particular areas' data, that is the channel.

Flagged-area lists were derived programmatically from the saved baseline traces via
`multimodality.classify_multimodality` (union, across every `*_lambda_weights` var, of areas
classified as anything other than `round_tripping`/`not_multimodal`) — 12 areas for AZ1d, 43 for
AZ4 (AZ4's combined P+E flagged surface turned out far larger than AZ1d's alone: 64+74 raw
flags, mostly benign `round_tripping`, before filtering to the 43 non-benign union).

**AZ1d: the hypothesis is confirmed, cleanly, by both interventions.**

| | max r-hat | min ESS (bulk) |
|---|---|---|
| AZ1d baseline (200 areas) | 1.04 | 169 |
| AZ1d dropped (188 areas) | 1.01 | 1886 |
| AZ1d pinned (200 areas, 12 pinned) | **1.00** | **2632** |

Both interventions resolve the residual `lag_P_mu_logit`/`lag_P_tau` problem outright — pinning
(which keeps all 200 areas' data) does slightly better than dropping, plausibly because
`sigma_plan`/`sigma_ben` get informed by 12 more areas' worth of P/E observations, marginally
easing the sampler's overall geometry even though those 12 areas' own `raw_offset` no longer
feeds `mu_logit`/`tau` either way. This is real, direct support — not just correlational — for
the "these 12 areas' own unresolved posterior was the channel" story specifically for AZ1d.

**AZ4: the SAME intervention makes things WORSE, not better — the naive extension of AZ1d's
story to AZ4 is wrong.**

| | max r-hat | min ESS (bulk) |
|---|---|---|
| AZ4 baseline (200 areas) | 1.17 | 31 |
| AZ4 dropped (157 areas) | **1.65** | **13** |
| AZ4 pinned (200 areas, 43 pinned) | **1.65** | **12** |
| AZ4 efixed (200 areas, E lag constant) | 1.46 | 15 |

Every one of AZ4's population-level lag scalars gets meaningfully worse under DROP/PIN
(`lag_P_mu_logit[0]` r-hat 1.06→1.48/1.27; `lag_E_tau[0]` r-hat 1.07→1.65/1.65), and
`sigma_delta_top_boost` — already AZ4's single worst-converged scalar — is not helped by any of
the three interventions (ESS 40 baseline → 17/19/32).

**Mechanism, checked directly rather than left as a puzzle**: `lag_P_tau`/`lag_E_tau` collapse
toward zero under both DROP and PIN (e.g. `lag_E_tau[0]` mean 3.60 baseline → 1.19 dropped → 0.91
pinned) — consistent with a Neal's-funnel explanation, not the "leakage" story. `tau` (the
between-area variance of the lag hierarchy) is identified BY the spread of `raw_offset` across
areas; the flagged areas are flagged precisely because they diverge most from the population
kernel, i.e. they are exactly the areas carrying the most identifying signal for `tau`. Removing
or neutralizing them doesn't remove a source of contamination — it starves `tau` of the evidence
needed to estimate it, pushing the remaining hierarchy into a smaller-`tau`, harder-to-sample
funnel regime. AZ1d's much smaller, more homogeneous 12-area flagged set didn't trigger this
(its own `tau` barely moved: 3.74/4.12 baseline → 3.16/3.5 pinned) — AZ4's larger 43-area
(21.5% of the dataset) flagged set apparently carries a much larger share of the total
between-area spread, so removing it is a bigger hit to an already more weakly-identified joint
model (AZ4's own baseline ESS on these scalars, 31-201, was already far below AZ1d's 169-1463
before any ablation).

**EFIXED partially confirms the MORE SPECIFIC `sigma_noise_E` claim, but doesn't fix AZ4
overall.** `sigma_noise_E` recovers dramatically once E's lag mechanism has no sampling freedom
at all (ESS 33→4830, r-hat 1.16→1.00 — back near AZ3-alone's own ESS>3600) — real, direct
support for "E's lag-category sampling noise specifically leaks into sigma_noise_E" (the
mechanism `AZ1d` was independently built to avoid). But the SAME intervention makes several of
P's own hierarchy parameters worse (`lag_P_mu_logit[0]` r-hat 1.06→1.30; `lag_P_tau[1]` r-hat
1.03→1.46) and doesn't move `sigma_delta_top_boost` (ESS 40→32). So the two claims this section
set out to test are NOT both true: `sigma_noise_E`'s specific problem really is E's lag
mechanism's own sampling freedom (confirmed); but AZ4's overall worst-converged scalar
(`sigma_delta_top_boost`) and its lag hierarchies' general fragility are NOT explained by either
"a leaking minority of areas" or "E's lag mechanism specifically" — something else (plausibly
just AZ4's accumulated joint complexity — three near-discrete-choice mechanisms sharing one `z`
— rather than any one isolable channel) is responsible, and remains uncharacterized.

**Practical upshot**: this strengthens, rather than changes, the existing recommendation (option
3 above) — AZ1d's asymmetric fix (P hierarchical, E same-year) is validated further by this
test (its underlying mechanism holds up under direct intervention, not just correlational
evidence), while AZ4's own convergence problems are now confirmed NOT fixable by identifying and
excluding/pinning "bad" areas — that lever actively backfires. Rebuilding the combined model on
AZ1d's design (removing E's hierarchical lag from the combination entirely, per option 3) sidesteps
this failure mode altogether rather than requiring a fix for it.

### AZ1d → AZ1g — regularized horseshoe: the leakage mechanism confirmed far more strongly than the ablation test showed, and a real (if partial, and sample-dependent) fix

A follow-up round asked to dig further into AZ1d's residual `lag_P_mu_logit`/`lag_P_tau`
problem using only the traces already on disk (`AZ1d.nc`, `AZ1d_16chain_informedinit.nc`,
`AZ1d_ablation_dropped.nc`, `AZ1d_ablation_pinned.nc`, `AZ1d_resolved.nc`), then to check
whether the whole picture (flagged-area count, mechanism) generalizes to a different 200-area
sample, then to look for an actual (non-oracle) parameterisation fix — the ablation test above
already showed PINNING fully recovers `lag_P_mu_logit`/`lag_P_tau` (r-hat 1.04→1.00, ESS
169→2632), but pinning requires already knowing each flagged area's posterior mean in
advance, so it's a diagnostic, not a deployable model.

**The mechanism is far more direct than the ablation section's own correlational check found.**
That check counted, per chain, how many flagged areas landed in the chain's minority mode, and
found only a weak correlation with the chain's own `lag_P_mu_logit`/`lag_P_tau` (r=-0.01/+0.26)
— too weak to be the whole story on its own. Redone with the actual MAGNITUDE of each chain's
flagged-area logits (how far a chain's own realized `area_logit` for the 12 flagged areas sits
from that chain's own mean, not just a binary "is this chain in the minority" flag) instead of a
discrete mode-membership count: **r=0.85-0.98** between that per-chain spread and the SAME
chain's own `lag_P_tau` draw (`results/scratch/az1d_leakage_mechanism.py`). This is close to
mechanically deterministic, not a diffuse hierarchical side-effect: `tau` is estimated from the
empirical spread of `area_logit` across all 200 areas, and the flagged minority contributes
disproportionately to that spread (that's *why* they're flagged), so a chain that commits to a
more extreme value for them almost by construction estimates a different `tau` than a chain that
committed to a milder one. The flagged areas are also not an arbitrary minority: median |D| is
7.6-8.6x the dataset median and median max|P_obs| is ~12x (`az1d_flagged_characteristics.py`) —
exactly the large-spike, stakeholder-critical areas this model family exists to serve, so
dropping/damping them (AZ1c's/AZ4b's already-rejected uniform tau cap) is not an acceptable fix
even if it worked.

**Generalization check, on a fresh Croydon-centred 200-area sample (`select_spatial_sample`,
different `center_latlon`, `az1d_altsample_generalization.py`) — the mechanism holds, but the
SIZE of the problem is sample-dependent.** A similar-sized minority gets flagged (13/200, vs
Islington's 12/200) and the same chain-level tau correlation reproduces (r=0.87 at k=0, weaker
but still positive at k=1, r=0.42) — this is not an artefact of one particular spatial sample.
But several of Croydon's flagged areas carry sequential LSOA codes (`E01034141`...`E01034154`),
suggestive of one contiguous development spanning several adjacent LSOAs with genuinely
correlated lag ambiguity, rather than scattered independent areas the way Islington's 12 were —
not investigated further this pass, but worth knowing before assuming every sample's flagged
set looks like Islington's.

**Fix built: `AZ1g`, a REGULARIZED horseshoe** (`_build_hierarchical_lag_regularized_horseshoe`)
— direct follow-through on AZ1e's own docstring, which named this as the standard literature
fix for AZ1e's specific failure (`local_scale`, an unregularized `HalfCauchy`, exploring to 3.09
million, producing this whole investigation's only divergences) but deliberately didn't build
it. Adds a slab (`slab_scale=10.0`, calibrated against AZ1d's own observed area_logit range,
up to ~30) that bounds the local multiplier's effective reach: as `local_lambda[a,k] →
∞`, the reparameterised `local_lambda_tilde[a,k] → slab_scale/global_tau[k]`, so the offset
saturates at `raw_offset[a,k] * slab_scale` — capped by construction regardless of how far into
`local_lambda`'s own heavy tail the sampler wanders, unlike AZ1e where the offset scaled
linearly with an unbounded `local_scale` all the way to the blowup. Verified directly (not just
by argument): a unit test evaluates the `tau` expression via symbolic substitution at
`local_lambda=1e6` and confirms it stays under `1.5 * slab_scale` regardless.

**Result, Islington sample — a real, clean win on the exact target (the shared scalars),
confirmed mechanistically, not just numerically:**

| | AZ1d | AZ1g |
|---|---|---|
| max r-hat | 1.036 | **1.021** |
| min ESS (bulk) | 169 | **437** (2.6x) |
| n_bad_rhat | 6 | 4 |
| divergences | 0 | **0** (AZ1e's blowup does not recur) |
| frac_flat_despite_active | 7.5% | 7.5% (unchanged) |
| plan_cov_90 / ben_cov_90 | 0.869 / 0.909 | 0.870 / 0.909 (unchanged) |
| low year-confidence | 7.0% | 8.0% (noise-level) |

`check-multimodality` confirms this is the mechanism working as designed, not just a favourable
number: `classify_scalar_multimodality` now returns `not_multimodal` for `lag_P_mu_logit` AND
`lag_P_global_tau` (their still-elevated r-hat, 1.01-1.02, has no detectable chain-level cluster
structure any more) — under AZ1d these same scalars' elevated r-hat WAS attributable to the
flagged minority's mode-switching (this section's own magnitude-based check, above). And
**`stuck_fixable` dropped from 2 (AZ1d: `E01033711`, `E01035656`) to 0** — giving these areas
their own escape valve let their own chains find the better mode unaided, changing
`E01033711`'s classification from "stuck in a worse mode" to `hard_genuine` (a real, honestly-
reported tie) rather than needing the manual reseed-and-resample `--resolve` step AZ1d required.
Spike-tracking (`results/scratch/az1g_spike_tracking.png`) confirms no practical cost:
`E01033711`'s two flagship spikes (2014, 2018) track at ~230/~268, matching AZ1d's own 265/268
closely.

**Result, Croydon sample — the shared-scalar improvement generalizes; full resolution of the
flagged minority does not, because Croydon's minority is a harder case:**

| | AZ1d (Croydon) | AZ1g (Croydon) |
|---|---|---|
| `lag_P_mu_logit` r-hat | 1.02-1.04 | **1.01** |
| `lag_P_tau`/`global_tau` r-hat | 1.08 | **1.05-1.06** |
| divergences | 0 | 0 |
| whole-model max r-hat (incl. per-area cells) | 1.883 | 1.709 |
| n_bad_rhat (incl. per-area cells) | 315 | 362 |
| flagged areas (lambda_weights) | 13/200 | 15/200 (2 `stuck_fixable`, 1 `hard_genuine`) |

The named population scalars improve in the same direction as Islington (real, if more modest),
and zero divergences on both models confirms the slab does its job even on this harder sample.
But the whole-model max r-hat/bad-cell-count barely moves and the flagged-area count doesn't
shrink — consistent with Croydon's flagged set containing a larger, more genuinely difficult
cluster (the adjacent-LSOA-code block noted above) than a "give it room" mechanism alone can
fully resolve. Read honestly: AZ1g fixes the specific, confirmed mechanism (the flagged
minority's own posterior noise contaminating the shared hierarchy scalars) on both samples
tested, without reintroducing AZ1e's divergence risk on either — but it is not a general cure
for per-area hard multimodality, and how MUCH of a model's residual r-hat problem this fix
closes depends on how large/entangled that sample's own flagged minority happens to be.

**What's left in AZ1g that is NOT multimodality, checked directly rather than left unlabelled.**
`check-multimodality`'s own `not_multimodal` category exists precisely to separate "bad r-hat
confirmed NOT attributable to multimodality" from `needs_review` ("doesn't cleanly classify" —
still an open question, not a clean negative). Running `classify_scalar_multimodality` on AZ1g's
named scalars gives a clean, definitive `not_multimodal` verdict for every one of them, on BOTH
samples:

| sample | cells classified `not_multimodal` | max r-hat among them |
|---|---|---|
| Islington | `sigma_ben`, `lag_P_mu_logit[0]`, `lag_P_mu_logit[1]`, `lag_P_global_tau[0]` | 1.021 |
| Croydon | `sigma_plan`, `sigma_ben`, `lag_P_mu_logit[0]`, `lag_P_global_tau[0]`, `lag_P_global_tau[1]` | 1.060 |

This matches Islington's own `n_bad_rhat=4` from the top-line `diagnose` table exactly — every
named-scalar r-hat problem AZ1g has left is now in this bucket, none is multimodality-driven any
more. Followed `docs/ess-rhat-diagnostic-guide.md`'s own procedure rather than stopping at the
label: autocorrelation for all of these decays to ≈0 by lag 20 on every chain (e.g. Croydon's
`lag_P_global_tau` acf(lag1)=0.69 → acf(lag20)=-0.02), ruling out slow within-chain mixing: this
is a between-chain, not within-chain, problem. Chain means show small but real, non-clustering
spread (Croydon `lag_P_global_tau[1]`: chain means 1.81-2.40, ~26% of its own mean; Islington
`lag_P_mu_logit[1]`: 1.09-1.28) — continuous jitter across all 8 chains, not two groups splitting
into distinct camps. This is the exact **"shallow basin / small persistent between-chain
disagreement"** signature already on record for AZ2's `sigma_delta_top_boost` (pre-AZ2b) and
AZ3's pre-floor `sigma_plan` — a hierarchy reconciling ~200 areas' worth of collective pull
against a moderately-sized draw budget, not a funnel, not a sampler failure, and (per those two
precedents) typically low-priority: bounded, non-catastrophic, and not previously found to cost
anything on the metrics that matter (flatness, coverage, spike-tracking). Not chased further
this pass, consistent with how AZ2's/AZ3's own instances of this same pattern were left open
rather than over-fitted.

**Verdict: adopt as an improvement over AZ1d for the shared-hierarchy-scalar problem
specifically, not yet a full replacement recommendation.** The design goal stated at the start
of this round — parameterise the lag mechanism so a minority of areas can be genuinely
multimodal without dragging the majority's shared parameters into disagreement — is confirmed
achieved, mechanistically, on real data, twice. What's NOT yet established: whether AZ1g's
Croydon-sample residual (the larger, harder cluster) would resolve with a bigger `slab_scale`,
a per-lag-category slab, or is a genuinely different problem (the suspected single-development
spatial cluster) that no per-area-independent mechanism can address — flagged as the natural
next step, not run this pass. Registered in `ALL_MODELS`/tested in `tests/test_models.py`
(`TestAZ1gStructure`, `TestAZ1gSampling`) so it's available for that follow-up without
rebuilding.

### AZ1g's residual: does "just sample more" fix it? Tested directly, not assumed — no

Before reaching for AZ1h's prior-recalibration attempt (below), the cheaper standard levers
this document's own §2 step 5 and the `pymc-modeling` skill both list first — more draws/tune,
more chains — should have been tested on AZ1g directly. They hadn't been; closing that gap:

| | baseline (1500 draws/500 tune, 8 chains) | +draws/tune (4000/1500) | +chains (16) |
|---|---|---|---|
| `sigma_plan` r-hat / ESS | 1.01 / 2768 | 1.01 / **1118** (worse, despite 2.67x more draws) | 1.01 / 5266 |
| `sigma_ben` r-hat / ESS | 1.02 / 437 | 1.01 / 606 | 1.02 / 708 |
| `lag_P_mu_logit[0]` | 1.01 / 1770 | 1.00 / 4413 | 1.01 / 3252 |
| `lag_P_mu_logit[1]` | 1.01 / 826 | 1.01 / 2122 | 1.01 / 1602 |
| `lag_P_global_tau[0]` (worst cell) | **1.02 / 686** | **1.02 / 752** (unmoved) | **1.03 / 454** (worse) |
| `lag_P_global_tau[1]` | 1.00 / 2635 | 1.01 / 7115 | 1.01 / 5494 |

**Neither lever meaningfully helps.** The worst cell, `lag_P_global_tau[0]`, is completely
unmoved by 2.67x more draws/tune and gets slightly WORSE with twice the chains — the same
signature already documented for AZ1a/AZ1b's genuine hard multimodality (more chains
characterize a real disagreement more precisely, they don't average it away), now observed for
this milder shallow-basin case too: r-hat doesn't improve because there's nothing wrong with the
sampling budget, the disagreement itself is a small but real feature of the posterior.

**A second, unprompted finding worth flagging**: `sigma_plan`'s ESS got WORSE with more
draws/tune (2768→1118) despite 2.67x more total draws — not a fluke specific to AZ1g. This
exact pattern (`sigma_plan`'s bulk ESS dropping when draws/tune were increased, 308→94) was
already documented for AZ2's own "just needs more draws" test earlier in this document. Now
confirmed on a second, unrelated model — `sigma_plan` specifically seems to have a recurring,
non-monotonic sensitivity to longer runs across this AZ family, worth remembering before
reflexively trying "more draws" as a fix for THAT particular scalar again.

**Verdict: confirms, rather than merely assumes, that AZ1g's residual is compute-irreducible.**
Both standard, cheap levers were tested directly on the real model rather than skipped over —
neither works, which is itself useful evidence for the `not_multimodal`/shallow-basin
characterization above: a genuine sampling-budget shortfall would respond to more draws or
chains; this doesn't. Strengthens (rather than merely leaves untested) the "stop and report"
conclusion for this specific residual.

### AZ1g → AZ1h — moving to the CANONICAL regularized horseshoe recipe: rejected, on both changes tested

AZ1g's own `not_multimodal` residual (a mild "shallow basin" on `lag_P_global_tau`/
`lag_P_mu_logit`, confirmed via rank-histogram inspection: clean autocorrelation, a smooth
per-chain tilt, no discrete cluster split) prompted re-reading this codebase's `pymc-modeling`/
`pymc-extras` skill references for the standard treatment of exactly this signature. Two
concrete deviations from the skills' own canonical regularized-horseshoe recipe
(`references/r2d2_horseshoe.md`) stood out in AZ1g's construction: (1) the slab scale
(`slab_scale=10.0`) was a hand-fixed constant rather than sampled (`c2 ~ InverseGamma(2,1)` in
the reference), and (2) `global_tau`'s prior reused AZ1d's unrelated `tau_sigma=1.5` rather
than being derived from this model's own expected sparsity via the reference's
`tau0 = p0/(p-p0)/sqrt(n)` formula. Built `AZ1h` with both corrections
(`_build_hierarchical_lag_regularized_horseshoe_v2`): `c2 ~ InverseGamma(2, slab_c2_beta=100)`
(prior mean matching AZ1g's old fixed `slab_scale^2`, so the change is "sample it" not "pick a
different number"), and `global_tau ~ HalfCauchy(tau0)` with `tau0` computed from `p0=15`
(a documented, slightly-conservative estimate spanning both samples' own flagged-area counts),
`n_areas=200`, `n_years=10`.

**Result on Islington — a clear regression, not an improvement, on named scalars AND
dramatically worse at the per-area level:**

| | AZ1g | AZ1h |
|---|---|---|
| max r-hat (whole model) | 1.021 | **1.065** |
| min ESS (whole model) | 437 | **108** |
| `lag_P_global_tau` r-hat | 1.021 | **1.02-1.07** |
| `lag_P_global_tau` posterior | mean 2.35, sd 0.5-0.6 | mean 3.3-31, **sd 10-100** |
| divergences | 0 | 0 |

`lag_P_global_tau`'s posterior sd exceeding its own mean by 3-30x is a textbook **prior-data
conflict** (`troubleshooting.md`'s own named pattern: "Prior too narrow: Data suggests values
outside prior range") — the `tau0≈0.026` implied by the sparsity formula is 2-3 orders of
magnitude tighter than what the data actually wants (~3-4, matching AZ1d's/AZ1g's own converged
value), so the HalfCauchy prior fights the likelihood rather than informing it. This should have
been caught by a prior predictive check before spending a full 8-chain run on it (the exact
workflow step `pymc-modeling`'s own skill emphasizes) — noted as a process lesson, not just a
model-design one.

**Isolated which of the two changes actually caused this, rather than reject both on one
combined result** (`results/scratch/az1h_isolate_c2_vs_tau0.py`): built `AZ1h_sampled_slab_only`
— AZ1h's sampled `c2`, but `global_tau` REVERTED to AZ1g's plain `HalfNormal(1.5)`. Named
scalars recovered completely (`lag_P_global_tau` mean 2.31/3.97, sd 0.7/0.82, r-hat 1.00-1.02,
matching AZ1g's own healthy range) — confirming the prior-data conflict really was `tau0`
specifically, not the sampled slab. **But the WHOLE-MODEL picture got dramatically worse, not
better**: max r-hat 1.021 (AZ1g) → **1.961**, n_bad_rhat 4 → **201**, min ESS 437 → **10** — all
concentrated in exactly the per-area machinery AZ1g had fixed cleanly
(`lag_P_raw_offset` max r-hat 1.64, `lag_P_local_lambda` 1.24, `lag_P_tau` 1.27,
`lag_P_lambda_weights` 1.67, `z`/`delta` 1.96).

**Mechanism**: sampling `c2` (shared across all 200 areas) alongside each area's own
`local_lambda` reintroduces a genuine non-identifiability that AZ1g's fixed slab didn't have.
With `slab_scale` fixed, each area's `local_lambda` alone determines its own `tau[a,k]`. With
`c2` free, a GIVEN `tau[a,k]` can be reached by many different `(local_lambda[a,k], c2[k])`
combinations, and `c2` is shared, so its value is a compromise across all 200 areas — the ~185
well-behaved areas want `c2` small (so their own near-zero `local_lambda` reliably shrinks
`tau` to ~0), while the ~15 flagged areas want `c2` large enough to let them commit strongly to
a lag category. This is the exact class of problem `troubleshooting.md`'s "Redundant
Intercepts" pattern warns about in a different guise: adding one shared free parameter that
trades off against many already-free per-group parameters, with no new information to
separately pin down both — a new, genuine ridge, not the "sampled hyperparameter" textbook
description would suggest on its own.

**Croydon (both the full `AZ1h` design and, by the mechanism above, presumably the isolated
variant too) shows the same directional cost, muted only because Croydon's whole-model number
is already dominated by its own larger, harder multimodal cluster**: `lag_P_global_tau` r-hat
1.05-1.06 (AZ1g) → 1.06-1.08 (AZ1h), `sigma_ben` r-hat worsened to 1.04. Whole-model max r-hat
is a near-wash (1.709 → 1.702) only because that number was already set by the unrelated
adjacent-LSOA-code cluster, not because AZ1h's own regression is absent there.

Spike-tracking (`results/scratch/az1h_spike_tracking.png`) shows no wrong-direction tracking —
`E01033711`'s flagship spikes still track correctly — but visibly wider credible-interval bands
in several panels (e.g. `E01033700`'s 2021 band reaching ~250 vs AZ1g's tighter equivalent),
consistent with degraded internal convergence costing precision even where point estimates
survive.

**Verdict: rejected, both changes.** The canonical regularized-horseshoe recipe, applied
faithfully from this codebase's own `pymc-extras` skill reference, is measurably worse than
AZ1g's simpler, hand-tuned construction on the metric that matters most (per-area convergence),
via two independently-confirmed mechanisms (a mis-transferred sparsity formula, and a new
shared-slab/per-area ridge). This is the same lesson this document has reached repeatedly from
different directions (AZ1c's tau cap, AZ4b's tau cap, AZ1e's unregularized horseshoe, and now
AZ1h's "more textbook-correct" horseshoe) — textbook/literature-standard fixes are hypotheses
to test against this specific model's own geometry, not defaults to prefer on authority, and
"more principled-looking" is not the same as "better here." **AZ1g remains the recommended
design** for this family's lag-pooling problem; `slab_scale` staying a fixed, hand-calibrated
constant is now confirmed (not just untested) to be the right choice, not a shortcut that a
more sophisticated version would strictly improve on. `AZ1h` is kept in the codebase (registered
in `ALL_MODELS`, tested in `tests/test_models.py`) as the documented negative result, per this
codebase's own convention for instructive rejected models.

## AZ2b's `sigma_plan`/`sigma_ben` follow-up: closing the branch

Direct question that prompted this: AZ1g found a real fix for AZ1d's leakage (a genuinely-
ambiguous minority of areas' unresolved posterior dragging a *shared* scalar's between-chain
agreement down) by decoupling local per-area shrinkage from the shared global scale via a
regularized horseshoe. Does an analogous fix exist for AZ2b's `sigma_plan`/`sigma_ben`
(126/180 bulk ESS in the original diagnose table)?

**First, a correction to the record.** Both AZ1g's and AZ2b's earlier "achieved large ESS"
claims were based only on each model's own named `var_names`-scoped `diagnose` table, not a
full whole-model scan. Running that scan on AZ1g for the first time found the same hidden
per-area `z`/`delta` ESS problem AZ2b has (min ESS ~10-13, max r-hat ~1.6-2.2) — AZ1g's real
achievement was narrower than claimed: it stopped the ambiguous minority's own unresolved
posterior from leaking into the *shared* scalars, not universal high ESS. AZ2b's `z`/`delta`
ESS (min bulk 6-19 depending on chain count) is the SAME accepted, out-of-scope limitation
(no reallocation mechanism, weak year confidence — both explicitly declared acceptable for
this branch) — not a new problem, and not what this follow-up chases.

**Leakage check — not confirmed.** The 4-chain leakage-correlation check (chain-level
deviation of the 12 known-ambiguous areas' `z` from the pooled mean, vs that chain's own
`sigma_plan`/`sigma_ben` draw) gave r=0.17/-0.16 — too few chains (n=4) to trust, unlike
AZ1d's check which needed 8 chains to be reliable. Resampled AZ2b at 8 chains
(`AZ2b_8chain.nc`) specifically to redo it properly:

```
corr(flagged-area z deviation, chain sigma_plan) = 0.186
corr(flagged-area z deviation, chain sigma_ben)  = 0.097
```

Nowhere near AZ1d's 0.85-0.98 signature. AZ2b's shallow basin is **not** the same leakage
mechanism AZ1g fixed — and structurally couldn't be fixed the same way even if it were:
AZ1g's horseshoe worked by giving each area its own local shrinkage parameter decoupled from
a shared global scale, a local/global split that exists because AZ1's lag mechanism gives each
area its own `lag_P_lambda_weights`. `sigma_plan`/`sigma_ben` have no such per-area
decomposition — they're single global noise scales feeding every area's likelihood at once,
so there's no "local" piece to give a horseshoe.

**Chain count alone is not a reliable fix — bounces non-monotonically.**

| chains | target_accept | sigma_plan ESS (r-hat) | sigma_ben ESS (r-hat) |
|---|---|---|---|
| 4 | 0.9 | 126 | 180 |
| 8 | 0.9 | 534 (1.017) | 350 (1.020) |
| 12 | 0.9 | 250 (1.032) | 411 (1.023) |
| 8 | 0.97 | 234 (1.025) | 570 (1.015) |
| 12 | 0.97 | 348 (1.026) | 476 (1.021) |

Going 4→8 chains looked like a fix (126→534) but going 8→12 *reversed* it (534→250) — the
opposite of AZ1g's own longrun/16-chain test, where more sampling reliably did nothing at all.
This non-monotonic bounce, combined with tight between-chain mean agreement (~3% spread) at
every chain count tested, is the signature of slow within-chain mixing on flat local
curvature — the same "small, persistent between-chain disagreement, not slow mixing, not a
simple ridge" pattern already on record for AZ2's original `sigma_delta_top_boost` problem
(pre-AZ2b) — not multimodality, and not something chain count reliably resolves.

**Pairwise ridge check — clean, no structural culprit.** Pooled and per-chain correlation
between `sigma_plan`, `sigma_ben`, and `sigma_delta_top_boost` on both the 8- and 12-chain
traces: `corr(sigma_plan, sigma_ben)` ≈ 0.03-0.04 (no mutual ridge); `corr(sigma_plan/ben,
sigma_delta_top_boost)` ≈ -0.17 to -0.25 (mild, not a strong identifiability ridge).

**`target_accept=0.97` — fixes `sigma_ben`, not `sigma_plan`.** A flat-curvature region
typically needs a smaller NUTS step size than `target_accept=0.9`'s adaptation gives it (the
same fix AZ1g used, going 0.9→0.98). At 8 chains this gave `sigma_ben` its best result across
every config tried (570 ESS, r-hat 1.015) but left `sigma_plan` unmoved (234, actually below
the plain 8-chain/ta=0.9 result). The combined 12-chain+`ta=0.97` config (the one untested
permutation) didn't rescue `sigma_plan` either (348, r-hat 1.026) — four distinct
sampler-setting configurations, no reliable win for `sigma_plan` in any of them.

**P_obs-sparsity hypothesis — proposed, then refuted by a clean ablation.** `P_obs` is far
more zero-inflated than `E_obs` on this 200-area sample (54.3% vs 28.1% exact-zero cells; 25%
of areas have <3 active planning-years vs 6% for BEN) — a plausible reason `sigma_plan` would
be harder to pin down than `sigma_ben`. Refuted directly: `AZ0a` — identical `P_obs`/`E_obs`
data, identical `nu=4` StudentT likelihoods, same 200-area sample, same 4-chain default
config, but WITHOUT AZ2b's top-boost mechanism — converges cleanly on both scalars (`sigma_plan`
ESS 3511, r-hat 1.0046; `sigma_ben` ESS 3180, r-hat 1.0060). Since the data is identical
between the two models, sparsity cannot be the cause; the top-boost mechanism itself is.

**Top-boost residual/ridge check, restricted to the boosted subset — also inconclusive.**
`_build_zero_sum_z_prior_top_boost_smooth` gives the top-|D|-quartile ~25% of areas a much
larger `sigma_delta` (floor=3 + up to ~`top_boost`'s posterior mean of ~25), so `z` can absorb
almost any residual there without `sigma_plan`/`sigma_ben` having to explain it — the
hypothesis being that this trades noise-explanation between `z` and the shared scalars in a
way the all-200-area pooled ridge check (above) would dilute. Checked directly: correlation
between `top_boost`'s per-draw value and the mean `|P_obs - z|`/`|E_obs - z|` residual, split
top-quartile vs non-top-quartile — `corr(top_boost, residual)` is weak in both subsets
(-0.05 to -0.13), and `corr(sigma_plan/ben, own residual)` (0.26-0.41) is just the ordinary
mechanical relationship any StudentT scale has to its current residual, not a pathological
ridge. No clean single mechanism found.

### Verdict

Confirmed, via a clean causal ablation (`AZ0a` vs `AZ2b`, identical data): the top-boost
mechanism is *what* causes `sigma_plan`/`sigma_ben`'s shallow basin. Not confirmed: *how* —
leakage, a pairwise ridge, and a top-quartile residual split were all checked directly and
came back weak or inconclusive. This matches `AZ2`'s own pre-`AZ2b` diagnosis exactly (a
small, persistent between-chain disagreement, explicitly not reducible to slow mixing or a
simple 2-3 variable ridge even after direct checking) — `AZ2b`'s smooth ramp fixed
`sigma_delta_top_boost`'s own ESS as intended (47→800+) but the same underlying pathology
resurfaced on `sigma_plan`/`sigma_ben` instead of being resolved. A displaced symptom, not a
fix.

**Adopted `AZ2b.sample_kwargs`: `chains=8, target_accept=0.97`** (`models.py`) — the best
config found across four tested. `sigma_ben` is fixed by this (570 ESS, r-hat 1.015).
`sigma_plan` is accepted as a disclosed, out-of-reach-by-sampler-tuning limitation of this
branch (234-534 ESS depending on config, never reliably ≥400, r-hat consistently 1.02-1.03) —
not chased further, since sampler tuning is now exhausted (four configurations tried) and no
structural fix was identified within AZ2's own scope (no reallocation/lag mechanism, per this
round's ground rules). Branch closed as "good enough with a disclosed limitation," not
abandoned — the top-boost mechanism's original purpose (fixing `frac_flat_despite_active`,
9.0% vs AZ0a's 11.5%) stands, and `z`/`delta`'s low ESS was already accepted as genuine,
out-of-scope year-allocation ambiguity before this follow-up began.

## Not done in this pass (flagged, not forgotten)
- No spike-tracking plot (`plot_spike_tracking_examples`) generated for AZ1a (8-chain),
  AZ1c, or AZ2b — the diagnose-table numbers and (for AZ1c) a single direct spot-check on
  E01033711 were used instead, given this was a breadth-first pass across three models
  rather than a single deep dive. Worth generating before any of these three feed into
  Phase 4, per this round's ground rule that aggregate diagnostics alone aren't sufficient
  evidence of practical model quality.
- No attempt at AZ1b's other two untried options (coarser D-band grouping; analytically
  marginalizing the near-discrete choice) — AZ1c was chosen as the most direct test of the
  doc's own "Status: open" option 1. The other two remain open if AZ1b/AZ1c's ceiling is
  revisited later.

## Follow-up: the manual multimodality investigation is now a reusable pipeline

Everything in this document's Pattern-2 sections (`hierarchical_mode_summary`, the
log-likelihood-gap check, the within-chain switch-rate check that caught `E01035649`'s
round-tripping, and the validated informed-init reseeding fix) was, at the time, a sequence of
one-off scripts re-run by hand for each model. `housing_projections.multimodality` (exposed via
`housing-projections check-multimodality`) now assembles all four into one tested, documented
pipeline — see `docs/multimodality-diagnostic-pipeline.md` for the full walkthrough. Running it
against `AZ1d.nc` directly reproduced this document's own hard-won characterization in one
command: `E01002702` as `hard_genuine`, `E01033711`/`E01035656` as `stuck_fixable`,
`E01035649`/`E01035646` correctly separated into `round_tripping` rather than lumped in with the
stuck areas, and `E01035708` correctly flagged as `mixed` (a genuine tie plus one stuck
straggler) rather than mischaracterized as a simple 2-way split. Use this pipeline first for any
new model's Pattern-2-shaped r-hat/ESS problem; fall back to this document's manual approach only
for what it can't cleanly classify, or when the pipeline's own thresholds (calibrated on this
family) need re-deriving for a materially different model.
