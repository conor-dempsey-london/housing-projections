# Data manifest

`data/` (where `DATA_PATH` points, per `.env`) is gitignored — raw input data isn't tracked
in this repo. This file exists so that, even without the data itself under version control,
each tagged release still has an auditable record of exactly which data version it was built
against, and how to reproduce or verify it.

Update this file whenever the data referenced by `DATA_PATH` changes in a way that could
affect model results, and tag the commit that updates it.

## PLD completions (`<DATA_PATH>/pld/lsoa_completions_time_series_pivot.csv`)

### Production file, as of tag `pld-reextraction-production-2026-08-06`

- SHA256: `4184e1bf112d3fe4e04e5fe27de19d93487425cdf157453ec8115d0d05ad871`
- 4,994 LSOA rows (+ header)
- This is the corrected unit-level re-extraction described below, promoted to the
  production filename on 2026-08-06 after the evidence in "Promotion decision"
  cleared the bar. Also still available under its original filename,
  `lsoa_completions_time_series_pivot_unit_level.csv` (identical content/checksum).
- Generated: 2026-08-02, via `scripts/pld_completions_reextraction.py`
  (`level='unit'`, all London LPAs, FY2009/10-2026/27, `extract_type='residential'`)
- Regenerate with: `pixi run python scripts/pld_completions_reextraction.py`. Verify the
  output's checksum before treating a fresh run as equivalent to this one — PLD is a live
  data source (applications' `status`/date fields can be revised after the fact), so a
  re-run at a later date is not guaranteed to reproduce this exact file.

### Superseded file — `lsoa_completions_time_series_pivot_scheme_level_legacy.csv`

- SHA256: `2b5c5b373e0db0562f789593d884aa35ec60c57acee153eaf3ca2c18ed85b62a`
- 4,823 LSOA rows (+ header)
- This was the production file up to and including tag `pre-pld-reextraction-2026-08-02`.
  Kept under this archival filename (identical content) so it can still be selected via
  `--pld-file` for reproduction or comparison — not deleted.
- Provenance: unknown. Produced by an undocumented extraction run against
  `pld_database_live` at some unrecorded past date; the original extraction parameters
  were never recorded by whoever produced it. An investigation found strong circumstantial
  evidence — not proof — that it used `level='scheme'`, which has a known bug: it
  attributes a multi-phase scheme's entire lifetime unit count to whichever single year
  the scheme's own `actual_completion_date` field happens to report, rather than the years
  its individual units actually completed in.

### Validation history leading to promotion

- Cross-checked totals against the (then-)production file (whole-window total matches to
  within ~1.5%, but redistributed substantially across years — e.g. FY2009/10 alone: 459
  in the old file vs 23,414 here); cross-checked against an equivalent unit-level `starts`
  extraction (behaves like a real construction pipeline — starts lead completions through
  the mid-2010s buildout, completions catch up from 2021/22); per-LSOA totals over
  2011/12-2020/21 correlate with the 2011-2021 census change at 0.75 vs the old file's
  0.51, with RMSE roughly halved (99 vs 207); AZ3 sampled on the standard 200-LSOA test
  sample against both versions — `frac_flat_despite_active` improves (3.5% -> 2.5%),
  planning-residual MAE improves (~11.2 -> 10.8), UPRN-side metrics roughly unchanged.
  Known caveat: unit-level extraction also surfaces some genuinely messy individual PLD
  records (large single-line loss entries on applications that never reach
  `status='Completed'`) that need the project's existing `apply_outlier_exclusion` step to
  not distort results locally — already handled by that step, not a new gap.
- **2026-08-03: AZ3 re-sampled at full scale, matched to the old production trace's frozen
  LSOA list** (4,987 LSOAs; saved to `results/traces_full_corrected/AZ3.nc`, ~2.4h, 0
  divergences, max r-hat 1.01). Confirmed the 200-LSOA direction but had a known gap: the
  frozen-list matching meant `apply_outlier_exclusion` wasn't re-run independently on the
  corrected values, so 5 of 4,987 matched LSOAs (E01000153, E01001356, E01001358,
  E01033876, E01033926) carried an unscreened hard-outlier cell. Superseded by the
  2026-08-06 re-run below, which fixes this.
- **2026-08-06: AZ3 re-sampled at full scale with outlier exclusion run INDEPENDENTLY on
  each input** (`results/traces_full_corrected_v2/AZ3.nc`, 0 divergences, max r-hat 1.01).
  Confirms independent screening finds a genuinely different, though almost entirely
  overlapping, hard-outlier set per input — 4,987 areas either way, but 10 LSOAs differ
  (exactly the 5 flagged above, plus 5 more on the old-file side). Metrics below are
  reported on the 4,982-area intersection of both screened universes (`scripts/
  compare_pld_fullscale.py`), so neither side's outlier set choice affects the comparison:

  | metric | old (scheme-level) | corrected (unit-level) |
  |---|---|---|
  | `frac_flat_despite_active` | 4.66% | 3.69% |
  | planning coverage (90% nominal) | 0.9205 | 0.9248 |
  | UPRN coverage (90% nominal) | 0.8492 | 0.9007 |
  | planning residual std / MAE | 35.14 / 8.25 | 28.64 / 7.52 |
  | UPRN residual std / MAE | 37.48 / 8.86 | 37.41 / 8.97 |
  | Moran's I, planning residual (p) | -0.013 (p=0.045) | -0.038 (p=0.001) |
  | Moran's I, UPRN residual (p) | 0.110 (p=0.001) | 0.110 (p=0.001) |

  Dwelling totals are unaffected at every LSOA/borough by construction (AZ3 pins `sum(z)`
  to the census figure exactly regardless of input) — the entire effect above is on year
  attribution and residual fit, not headline counts. Full write-up, input-side breakdown,
  and per-area detail (including the Islington scheme's LSOA followed through to its
  posterior `z`) from the earlier full-scale pass in the published report: local copy at
  `results/pld_reextraction_report.html` (gitignored, not tracked — regenerate or re-fetch
  from the live artifact link in the conversation record if missing); the table above
  supersedes that report's own full-scale numbers (computed before independent outlier
  screening was fixed).
- **K-fold CV check found NOT usable as a discriminator for this decision, and why**: a
  leave-area-out K-fold CV of AZ3 on the two inputs (200-LSOA dev sample, K=10, same fold
  partition/seed as `docs/model-finalization-work-plan.md` Task 2) gave ELPD -15077.0
  (se 135.8) on the old file vs -15175.7 (se 138.8) on the corrected file — a small,
  non-significant difference (`|diff/se| ≈ 0.5`) but in the "worse" direction, apparently
  contradicting the domain-fitness numbers above. Traced to a real construction artifact,
  not evidence against the corrected input: 85% of the total ELPD difference came from the
  19% of held-out cells where the correction shifted the P value the most (checked directly
  against `elpd_i.csv` from both runs). AZ3's K-fold wrapper draws a held-out area's `z`
  from the model's fixed, per-area-independent prior alone (`docs/model-finalization-
  work-plan.md` Task 2's own design rationale — no per-area likelihood conditioning for the
  cells being scored), which is exactly the mechanism that lets AZ3 attribute a spike to the
  right year in real (non-CV) use. Held-out K-fold switches that mechanism off, so it
  penalises any input whose true pattern is more concentrated/redistributed relative to a
  flat prior guess — regardless of whether the redistribution is more accurate. This K-fold
  construction was validated for comparing model STRUCTURES on a fixed input (the confound
  applies equally to every model compared that way); it is not valid for comparing INPUTS
  on a fixed model, where one input can genuinely carry more of the area/year-specific
  signal the held-out mechanism can't see. Worth carrying into
  `docs/model-evaluation-methods.md` as a general caveat before this K-fold pipeline is
  reused for a similar input-comparison question.
- Not yet done, deferred as a separate follow-up (not required for this promotion
  decision): full-London K-fold/finalization validation of AZ3/AZ6c against the newly
  -promoted input (the checklist AZ3 was originally selected against, on the OLD input —
  see `docs/work-log.md`); a cross-check against PLD's own pre-aggregated
  `residential_details.total_no_proposed_residential_units` field; EPC (`epc_number`)
  fill-rate at full scale.

### Promotion decision (2026-08-06)

Domain fitness (the axis that matters most per the stakeholder priority — see
`docs/model-stopping-criteria-and-communication.md`) improves clearly and consistently
across every scale tested: `frac_flat_despite_active` (the core spike-attribution failure
mode this model family exists to avoid) drops by roughly a fifth at full scale (4.66% ->
3.69%), planning-residual fit improves substantially (MAE 8.25 -> 7.52), and UPRN
calibration coverage moves from borderline (0.849, just below the healthy 0.85-0.95 band)
to comfortably inside it (0.901). Two honest caveats, neither judged to outweigh the above:
the K-fold predictive-accuracy check is not usable here at all (see above — a construction
artifact, not a real signal either way); planning-residual spatial autocorrelation
(Moran's I) becomes more statistically significant under the corrected input (p=0.045 ->
p=0.001), though the effect size stays small in both cases (|I|<0.04) — worth a closer look
if it doesn't shrink once the model suite is re-run against this input, but not treated as
a blocker on its own given the small effect size. **Decision: promote — the corrected
unit-level re-extraction is now the production PLD file.**

This promotion is about the INPUT FILE only. It does not itself re-run or re-validate the
AZ3/AZ6c model finalization checklist, dashboard, or stakeholder report against the new
input — those still reflect the old file and are flagged as a separate, larger follow-up in
`docs/work-log.md`.

## OS AddressBase / UPRN net change (`<DATA_PATH>/uprn/final_residential_uprn_net_changes_by_oa_fy (1).csv`)

Unchanged by this investigation — not re-extracted, no new checksum recorded here.
