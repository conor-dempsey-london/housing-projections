# What's in this folder

Estimates of how many homes were added or lost each financial year, for
almost every LSOA in London, covering April 2011 – March 2021. These come
from our current best model and may be refined in future versions.

## Files at a glance

- **`area_year_estimates.csv`** — the main file: estimated change + range, per area, per financial year.
- **`area_scenarios.csv`** — for areas with more than one plausible year-allocation story, the distinct scenarios and their likelihoods.
- **`area_tier_summary.csv`** — one row per area: its confidence tier and the evidence behind it.
- **`borough_london_totals.csv`** — the same estimates rolled up to borough and London-wide totals.
- **`borough_boundaries.geojson`** — borough map shapes, for plotting the above.

## Background

Two administrative sources try to track housing-stock change year to year —
planning-permission completion records and OS AddressBase net UPRN changes —
but both are noisy and sometimes disagree. 

The 10-yearly Census gives a much
more reliable total for how much an area changed *over the whole decade*,
but not *which year*. The model currently treats the Census total as fixed and uses
the noisier annual sources to work out the most plausible year-by-year path
consistent with it — so every estimate comes with a **90% credible
interval**: given the data and the model's assumptions, a 90% probability
the true figure falls in that range (not a survey margin-of-error — it comes
from the spread of values the model finds plausible).

**Coverage:** London has 4,994 LSOAs; this run covers 4,987. The other 7 had
at least one year of planning/OS AddressBase data so extreme it was treated
as a data error and excluded — a modelling choice that could change in a
future run.

**Financial years:** `year` values are labelled by the calendar year they end
in — `2012` = April 2011–March 2012, ..., `2021` = April 2020–March 2021.

## Confidence tiers

Every area is one of:

- **Tier 1 — Confident.** A single clear year-by-year path. Use the estimate and range as given.
- **Tier 2 — Ambiguous.** The 10-year total is unaffected, but more than one story fits which year(s) absorbed the change.
  - **Resolved** — a small number of distinct, labelled scenarios (`area_scenarios.csv`), each with a likelihood.
  - **Unresolved** — no clean split. Check `frac_flagged_magnitude` in `area_tier_summary.csv`
    first: if it's low, the ambiguity is confined to a small part of the decade's change and
    the rest of the year-by-year story (including the dominant year(s)) is still reliable;
    only if it's high does "treat the year-by-year figures as indicative only" really apply.
- **Tier 3 — Total only.** Next to no annual data for this area. Only the 10-year Census total is reported.

Roughly: most areas are Tier 1; ~30% are Tier 2 resolved.

## File details

### `area_year_estimates.csv`

One row per area per financial year (4,987 × 10 = 49,870 rows).

| Column | What it means |
|---|---|
| `area` | LSOA code |
| `year` | Financial year, e.g. `2012` = April 2011–March 2012 |
| `tier` | `tier1` / `tier2` / `tier3` |
| `tier_subtype` | `resolved` / `unresolved`, `tier2` only |
| `z_mean` | Best estimate of net homes added this year |
| `z_lo90` / `z_hi90` | 90% credible interval |

For `tier2`/`unresolved` rows, `z_mean` blends genuinely different possible
stories — the wide range is honest uncertainty, not noise.

### `area_scenarios.csv`

One row per scenario, `tier2`/`resolved` areas only (~1,460 areas; almost all have 2 scenarios, a few have 3).

| Column | What it means |
|---|---|
| `area` | LSOA code |
| `scenario_label` | `Scenario A`/`B`/`C`, most to least likely |
| `weight` | Likelihood of this scenario (an area's scenarios sum to 1.0) |
| `peak_year` | Year with the largest change under this scenario |
| `peak_year_z` | Estimated change in that peak year |
| `year_profile` | Full 10-year estimate for this scenario, as a list of numbers in year order |

### `area_tier_summary.csv`

One row per area — the tier and the evidence behind it.

| Column | What it means |
|---|---|
| `area` | LSOA code |
| `borough_name` / `borough` | Borough name / ONS code |
| `D` | Census-recorded total change, April 2011–March 2021 — fixed input, not model output |
| `tier` / `tier_subtype` | As above |
| `n_low_confidence_years` | Years showing year-allocation ambiguity |
| `n_multimodal_years` | Years flagged with more than one plausible story |
| `n_flagged_years` | Same, restricted to `tier2` areas |
| `flagged_years` | Which years those are, e.g. `2012,2013` — `tier2` only |
| `frac_flagged_magnitude` | Share of the area's total decade change that falls in the flagged years above — `tier2` only. **Low** (e.g. under ~25%) means the area's dominant year-by-year pattern is actually confident and only a minor year or two is unresolved; **high** means the ambiguity really does run through most of the decade's change |
| `has_active_year` | Any meaningful annual data at all — `False` ⇒ Tier 3 |
| `max_rhat`, `min_flagged_corr`, `flagged_concentration` | Technical scores behind the tier classification; not for direct interpretation |

### `borough_london_totals.csv`

Area estimates summed to borough and London level.

| Column | What it means |
|---|---|
| `geography` | `London` or a borough name |
| `year` | Financial year as above, or `total` for the full 10 years |
| `n_areas` | Number of LSOAs in this geography |
| `z_total_mean` | Best estimate of total net homes added |
| `z_total_lo90` / `z_total_hi90` | 90% credible interval |

The `total` row's interval is proportionally much tighter than any single
year's, since it's anchored directly by the Census.

### `borough_boundaries.geojson`

Standard GeoJSON shapes for London's 33 boroughs (works in QGIS, Tableau,
Power BI, web-mapping libraries). One property per shape:

| Property | What it means |
|---|---|
| `borough_name` | Matches `borough_name`/`geography` in the CSVs above, for joining |
