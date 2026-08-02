# Data manifest

`data/` (where `DATA_PATH` points, per `.env`) is gitignored — raw input data isn't tracked
in this repo. This file exists so that, even without the data itself under version control,
each tagged release still has an auditable record of exactly which data version it was built
against, and how to reproduce or verify it.

Update this file whenever the data referenced by `DATA_PATH` changes in a way that could
affect model results, and tag the commit that updates it.

## PLD completions (`<DATA_PATH>/pld/lsoa_completions_time_series_pivot.csv`)

### Production file, as of tag `pre-pld-reextraction-2026-08-02`

- SHA256: `2b5c5b373e0db0562f789593d884aa35ec60c57acee153eaf3ca2c18ed85b62a`
- 4,823 LSOA rows (+ header)
- Provenance: unknown. Produced by an undocumented extraction run against
  `pld_database_live` at some unrecorded past date; the original extraction parameters
  were never recorded by whoever produced it. An investigation (see below) found strong
  circumstantial evidence — not proof — that it used `level='scheme'`, which has a known
  bug: it attributes a multi-phase scheme's entire lifetime unit count to whichever single
  year the scheme's own `actual_completion_date` field happens to report, rather than the
  years its individual units actually completed in.

### Candidate corrected cut — NOT the production default

- SHA256: `4184e1bf112d3fe4e04e5fe27de19d93487425cdf157453ec8115d0d05ad871`
- 4,994 LSOA rows (+ header)
- Generated: 2026-08-02, via `scripts/pld_completions_reextraction.py`
  (`level='unit'`, all London LPAs, FY2009/10-2026/27, `extract_type='residential'`)
- Regenerate with: `pixi run python scripts/pld_completions_reextraction.py`. Verify the
  output's checksum before treating a fresh run as equivalent to this one — PLD is a live
  data source (applications' `status`/date fields can be revised after the fact), so a
  re-run at a later date is not guaranteed to reproduce this exact file.
- Validation done so far: cross-checked totals against the production file (whole-window
  total matches to within ~1.5%, but redistributed substantially across years — e.g.
  FY2009/10 alone: 459 in the production file vs 23,414 here); cross-checked against an
  equivalent unit-level `starts` extraction (behaves like a real construction pipeline —
  starts lead completions through the mid-2010s buildout, completions catch up from
  2021/22); per-LSOA totals over 2011/12-2020/21 correlate with the 2011-2021 census
  change at 0.75 vs the production file's 0.51, with RMSE roughly halved (99 vs 207);
  AZ3 sampled on the standard 200-LSOA test sample against both versions — `frac_flat_
  despite_active` improves (3.5% -> 2.5%), planning-residual MAE improves (~11.2 -> 10.8),
  UPRN-side metrics roughly unchanged. Known caveat: unit-level extraction surfaces some
  genuinely messy individual PLD records (large single-line loss entries on applications
  that never reach `status='Completed'`) that need the project's existing
  `apply_outlier_exclusion` step to not distort results locally — already handled by that
  step, not a new gap, but worth knowing before using this file without that screening.
- Not yet done: full-London K-fold/finalization validation (the checklist AZ3 was
  originally selected against); a cross-check against PLD's own pre-aggregated
  `residential_details.total_no_proposed_residential_units` field; EPC (`epc_number`)
  fill-rate at full scale.
- **Status: a validated candidate, not yet substituted for the production file.** The
  production file above is still what `run-models`/`compare`/`report` actually use as of
  this tag.

## OS AddressBase / UPRN net change (`<DATA_PATH>/uprn/final_residential_uprn_net_changes_by_oa_fy (1).csv`)

Unchanged by this investigation — not re-extracted, no new checksum recorded here.
