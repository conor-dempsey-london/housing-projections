# Producing and communicating year-by-year estimates — work plan

**Status doc for this round of work.** Like `az-family-work-plan.md` and
`model-finalization-work-plan.md`, this is a live checklist — check back here at the
start of each phase and update it immediately when a phase's status changes. Follows
directly from `docs/model-stopping-criteria-and-communication.md`'s decision that AZ3
is the model to stop iterating on and move to production reporting; this doc covers
that "production reporting" step itself.

Three deliverables, in dependency order:

1. **Numerical estimates** for every LSOA-year (Phase A).
2. **A dashboard** for exploring the data and results, for non-technical GLA
   stakeholders (Phase B) — depends on Phase A's CSVs.
3. **A concise written report** summarising findings and open questions for
   stakeholders, no r-hat/ESS/LOO/Pareto-k technical detail (Phase C) — depends on
   Phase A's aggregate CSV.

## Phase A — Full LSOA-year numerical estimates export

**Status: DONE.**

Built `scripts/az3_year_estimates.py` — opens `results/traces_full/AZ3.nc` exactly
once (read-only) and applies the three-tier scheme from
`docs/model-stopping-criteria-and-communication.md` Sec 4 to every one of the 4987
areas x 10 years, not just the 16 hand-picked examples that
`results/artifacts/az3_full_characterization/` covers:

- **Tier 1 (confident)** — `z_identifiability_summary.confident` AND 0 KDE-flagged
  multimodal years. Point mean + 90% CI.
- **Tier 2 (ambiguous)** — otherwise, if the area has any active P/E year.

  **Method note — a real bug found and fixed before trusting this.** The first
  version clustered each area's full 10-year standardized z vector via whole-draw
  k-means (the same method as `az3_deep_dive_followups.py`'s
  `characterize_area_modes` / `plots/core.py`'s `plot_z_area_modes`). Verified
  against this project's own two headline documented ground-truth areas before
  trusting it, and it got both **backwards**: E01002702 (confirmed genuine
  bimodality, `az-family-work-plan.md` Phase 3) came out "unresolved"; E01002794
  (confirmed spurious/diffuse, `az3-report-review-plan.md`'s own pairwise-correlation
  investigation) came out "resolved" with a confident 4-scenario split. Root cause:
  whole-vector k-means is dominated by whichever years have the most variance,
  which can easily NOT be the specific years the (separately validated) per-cell KDE
  scan actually flagged as ambiguous — a real bimodality localized to 2 small-ish
  years gets swamped by two large, unambiguous years elsewhere in the row; and
  conversely k-means will always partition even genuinely diffuse data into
  roughly-balanced wedges, which looks identical to a real split by cluster-balance
  alone. **This also means `mode_recharacterization.csv`'s "734/999 areas need 3+
  scenarios" finding (`az3-report-review-plan.md`, marked closed) used this same
  flawed method and is likely unreliable — flagged here as a known follow-up, not
  fixed, per user instruction to scope this fix to Phase A only.**

  Fixed method, implemented in `scripts/az3_year_estimates.py`'s `decompose_area`:
  restrict every check to exactly the years `multimodal_cells.csv` already flagged
  as per-cell multimodal for that area (trust the validated KDE scan to say WHICH
  years are ambiguous, then only look at those dimensions).
  - 1 flagged year — no cross-year clustering needed; reuse
    `diagnostics._detect_modes`'s own validated per-cell KDE modes directly.
  - >=2 flagged years — **resolved** only if BOTH (a) the flagged years contain a
    real anti-correlated pair (min pairwise correlation <= -0.3) AND (b) a small
    subset of those years dominates as the single largest value across draws
    (top-2 argmax-share >= 0.5, the same bar `plot_z_area_modes` already uses,
    just now computed on the flagged subset instead of the whole vector). (a) alone
    is not sufficient once an area has 5+ flagged years: E01035709 (9 flagged years,
    all near-identical ~0-or-~90 cells) has pairwise correlations that are just noise
    around the -1/(n-1) baseline the zero-sum constraint mechanically induces among
    ANY interchangeable candidate years (observed mean -0.12, matching -1/8 almost
    exactly) — one pair sampling to -0.55 by chance clears the correlation bar
    despite there being no real 2-group structure (no block pattern in the full
    9x9 correlation matrix; the *count* of "high" years per draw is tightly fixed at
    3-4 of 9, the signature of genuine combinatorial exchangeability — confident
    about how many years absorbed the change, uncertain about which, not 2
    distinguishable stories). Its flagged-year concentration is 31%, correctly below
    the bar. If resolved, a 2-way k-means restricted to just the flagged dimensions
    gives the scenario split, reported as N labelled scenarios with relative
    posterior mass and each scenario's peak year.
  - Verified against all four documented ground-truth areas after the fix:
    E01002702 (resolved, 63%/37% split, corr -0.89), E01002794 (unresolved, corr
    only -0.18), E01004686 (resolved, 73%/27% — a genuine but imbalanced minority
    scenario, matching the report-review's own "don't discard a real 3% mode"
    finding), E01035709 (unresolved, concentration 31%).
  - *unresolved* — ambiguity confirmed but no clean split clears both bars —
    reported as "ambiguous, no clean scenarios," never a misleading point estimate.
- **Tier 3 (diffuse)** — no active P/E year across the whole decade
  (`area_summary.csv`'s `has_active_year == False`). Reported as total `D` only, no
  year breakdown. Takes priority over the other two checks — 32 areas were found with
  both `has_active_year == False` and >=1 KDE-flagged multimodal year, i.e. per-cell
  multimodality detected on cells with no real data behind them; Tier 3 correctly
  overrides rather than reporting a spurious scenario split there.

Reuses `results/artifacts/az3_full_characterization/area_summary.csv` and
`multimodal_cells.csv` (already on disk) for the tier-classification inputs rather
than re-running the expensive per-cell KDE scan (`detect_z_multimodality`) a second
time — does NOT reuse `mode_recharacterization.csv` (built with the now-superseded
whole-vector method, see the method note above). Final counts on the full
4987-area run: 542 Tier 3 (no active year), 2318 Tier 1 (confident + unimodal +
active), 2127 Tier 2 — 1462 resolved (clean scenario split, 2941 scenario rows,
1445 areas with 2 scenarios / 17 with 3), 665 unresolved (ambiguity confirmed, no
clean split).

Outputs → `results/artifacts/az3_year_estimates/`:
- `area_year_estimates.csv` — (area, year) grain, all 49,870 rows: `z_mean`,
  `z_lo90`, `z_hi90`, `tier`, `tier_subtype`.
- `area_scenarios.csv` — (area, scenario) grain, Tier-2-resolved areas only:
  `scenario_label`, `weight`, `peak_year`, `peak_year_z`, `year_profile`.
- `area_tier_summary.csv` — one row per area: tier/subtype, `D`, borough name,
  `n_low_confidence_years`, `max_rhat`, `n_multimodal_years`, `n_flagged_years`,
  `min_flagged_corr`, `flagged_concentration`.
- `borough_london_totals.csv` — per-borough and London-wide total, per year plus the
  full-decade total, with 90% CI — the "always show the trusted total alongside"
  rule from Sec 4.
- `manifest.json`.

Borough names (not just `LAD22CD` codes) joined via `gla_data._ons.fetch_geography_lookup`
(the same ONS lookup `analysis.py`'s `uncertainty_by_geography` already uses) —
`data.py`'s own `gdf`/`make_borough_idx` only carry the code.

## Phase B — Dashboard

**Status: DONE.**

Built `notebooks/8.0-cd-az3_estimates_dashboard.py`, a marimo app: headline stats
(London total ± 90% CI, tier breakdown %), a borough choropleth (colourable by
% Tier 1 / % Tier 3 / total change) built from a dedicated borough-level geometry
export, a borough dropdown driving a year-by-year total+CI trend chart, and an
LSOA search/borough-filter table whose selected row renders the per-area panel —
point+CI table for Tier 1, labelled scenarios with relative mass for Tier
2-resolved, an explicit "ambiguous, no clean scenarios"/"diffuse, total only"
callout otherwise. Never publishes a bare point estimate for a Tier 2/3 area
without its qualifier, per Sec 4's "what not to do."

**Zero runtime dependency on this repo's private data pipeline** — the notebook
never imports `housing_projections`/`gla_data`/`geopandas`; Phase A's 4 CSVs plus a
one-time borough-geometry export (`scripts/export_borough_boundaries.py`, dissolves
LSOAs to 33 borough polygons via geopandas at BUILD time only, simplified +
reprojected to WGS84, 269 KB GeoJSON) are gzip+base64-embedded directly into the
notebook's own source by `scripts/build_dashboard.py` (re-run that script, not the
notebook by hand, whenever Phase A's CSVs change). This was necessary because the
audience is non-technical GLA stakeholders (confirmed) — the export target
(`marimo export html-wasm`) runs entirely client-side via Pyodide, which cannot
reach a private git-based package or a local filesystem path at all.

Added `altair` as a project dependency (`pyproject.toml`) for the interactive
charts/choropleth — the marimo-notebook skill's own recommended plotting library
for this exact use case (declarative, renders via vega-lite, well-supported in
Pyodide), not previously used anywhere else in this codebase.

Exported artifact: `results/artifacts/az3_estimates_dashboard/` (`index.html` +
marimo's own JS/WASM runtime assets — **not** literally one portable file; per
marimo's own CLI help, "it must be served over HTTP, and cannot be opened directly
from the file system" — needs plain static hosting, e.g.
`python -m http.server --directory results/artifacts/az3_estimates_dashboard`
locally, or wherever GLA already hosts internal static content; no backend/Python
server logic required).

**Verification, and an honest limitation.** Caught 2 real runtime bugs (a missing
`altair` dependency; `mo.ui.altair_chart` not supporting `mark_geoshape` selection)
by running the notebook for real via `marimo export html-wasm`'s server-side sibling,
`marimo export html`, which executes every cell with this repo's normal Python
environment before baking in the initial output — confirmed the London total
(353,963 dwellings) and all section headings render correctly. Could **not**
visually verify the actual browser/Pyodide-rendered page (no browser-automation tool
available in this environment) — the underlying Python logic is the same either way
(no Pyodide-specific code paths), but per this project's own standing instruction to
check UI changes in a real browser before calling them done, this is a real gap:
open `results/artifacts/az3_estimates_dashboard/index.html` (via the http.server
command above) in an actual browser before sharing it with stakeholders.

**Added two further panels** (per follow-up request): "Which kinds of areas tend to
be less certain?" — a bar chart of confidence-tier % by size-of-change band
(confirmed on the data: confident-tier share drops from ~70-80% for small changes to
under 2% for the largest, non-monotonically at the very smallest band because that
band also holds most of the no-activity Tier 3 areas) plus a per-borough confidence
bar chart shaded by total change, showing the same size-of-change pattern holds
within boroughs too, not just across them — and "Does uncertainty add up
consistently across scales?" — for the currently-selected borough (or London),
compares the model's actual 90% range against what naively summing every individual
area's own range would give. Confirmed directly from the CSVs before building the
chart: the actual range is ~10-20% of the naive sum at borough scale and ~3% at
London scale — aggregating over more, largely-independent areas shrinks relative
uncertainty further, exactly as it should if the model's uncertainty is genuinely
coherent across scales rather than recombined incorrectly.

**Distribution.** Per marimo's own constraint (WASM apps must be served over HTTP,
not opened via `file://`), and since a local double-click launcher was declined:
this needs *some* static HTTP host, but no backend/Python server logic — a plain
network share (`\\server\path`) does **not** work, because that's still the
`file://` scheme from the browser's point of view even though it looks like a
normal folder to Windows. Options, roughly in order of how little new
infrastructure they need:
1. Any existing GLA-internal static web hosting (an intranet site, a SharePoint
   document library configured to serve files directly rather than via its own
   viewer UI, or an existing internal file/web server with a spare directory) —
   copy `results/artifacts/az3_estimates_dashboard/` there as-is.
2. A small internal server (a spare VM, or even a colleague's machine on the GLA
   network) running a one-line static file server — `python -m http.server` (needs
   Python) or any equivalent — kept running for as long as the dashboard needs to
   stay reachable. Fine for a short-lived share, not for something meant to be
   permanently available.
3. If GLA has any approved external static-hosting service already in use
   elsewhere (e.g. an S3-compatible bucket, Azure Static Web Apps) — worth checking
   with IT before standing up something new, since this dashboard's content
   (aggregated dwelling-count estimates) is unlikely to be especially sensitive, but
   that's a judgement call for whoever owns that decision, not something to assume.

Whichever is chosen, the whole `results/artifacts/az3_estimates_dashboard/`
directory needs to move together (the exported `index.html` plus its JS/WASM/font
assets) — it's a small static site, not a single portable file.

## Phase C — Concise stakeholder report

**Status: DONE.** `docs/az3-stakeholder-summary.md`.

Plain-language document, separate from `html_report.py` (which stays as the
internal technical model-comparison artifact). Leads with the finding that,
*given this round's modelling choice to treat the Census counts as exactly
correct* (a simplifying assumption, not a claim the Census itself is error-free —
flagged explicitly as revisitable, not fixed here per user correction after an
earlier draft overclaimed this), the model adds no further uncertainty of its own
to the London/borough decade **total** — the zero-sum census-anchored z-prior
forces every area's yearly figures to sum to its own Census-recorded total by
construction, confirmed directly from `borough_london_totals.csv`: every
`year == 'total'` row has `z_total_lo90 == z_total_mean == z_total_hi90`, zero
width, while every per-year row has a genuine, substantial CI. The model's own
uncertainty is entirely about the year-by-year split, on top of whatever
uncertainty already exists in the Census figure itself (not modelled this round).
Reports the tier breakdown in plain English (47% confident / 29% narrow to a small
number of labelled possibilities / 24% genuinely unclear — the last figure
collapses Tier 2-unresolved and Tier 3 into one stakeholder-facing category, since
both cash out to the same practical statement: "total [under this round's Census
assumption] is solid, year is not"), 3 concrete example areas (one per tier, plain
language, no jargon), borough-level framing
using `area_tier_summary.csv`'s `D`/tier aggregates, and an open-questions section.
No r-hat, ESS, divergences, LOO, ELPD, or Pareto-k anywhere in this document.

**Open-questions section extended** (per follow-up request) with: known potential
data-quality issues in both source datasets (planning completions and OS
AddressBase-derived records) and a note that an investigation of both is already
underway separately, plus that specific expert knowledge of known issues in either
source could be built into a future model version directly; a note that the same
kind of model could be extended to draw on further data sources (EPC records
floated as one candidate); and an assessment of temporal lag and spatial spillover
— both tested during the AZ-family round and not included in AZ3, grounded in
existing evidence rather than a fresh claim: the spatial-misallocation mechanism
(M5) was decisively beaten by AZ3 on genuine K-fold CV (Task 2, `elpd_diff/dse`
-10.2), and the lag mechanism (AZ1b), combined with AZ3's own pieces into AZ4/AZ4b,
bought at best a marginal, noise-level domain-fitness improvement while making
year-allocation confidence worse (`model-stopping-criteria-and-communication.md`
§5) — so neither is expected to meaningfully change the headline numbers if
revisited, though neither has been conclusively ruled out under a different
implementation.

**Exported to PDF** via `scripts/export_stakeholder_pdf.py` → `docs/az3-stakeholder-
summary.pdf` — Markdown → HTML (via the `markdown` package) → PDF via headless Edge's
built-in `--print-to-pdf` (ships with Windows; no new heavy dependency like
weasyprint/wkhtmltopdf/pandoc). Re-run after any edit to the source Markdown.
