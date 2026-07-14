# Full-dataset trace characterization — reusable method

**Purpose**: how to produce a full-scale characterization report (like AZ3's, on all
4987 London LSOAs) for any future finalist trace, without re-deriving the pipeline or
re-reading a multi-GB trace file more than once. Written after building and running this
pipeline against `results/traces_full/AZ3.nc` — see
`docs/model-finalization-work-plan.md` Task 1 for the request this answers.

## The two-script split, and why

- **`scripts/full_dataset_characterization.py`** — the only script that ever opens the
  trace file. Loads it once, computes every summary table the report needs, and writes
  them as CSVs (plus a handful of pre-rendered PNGs) to `--output-dir`. Reusable against
  any model's full-scale trace via `--trace-path`/`--model-name` — nothing in it is
  AZ3-specific except the default paths.
- **`results/artifacts/az3_full_characterization/build_report.py`** — reads only the
  CSVs/JSON/PNGs already written by the script above and re-renders `report.html`. It
  never opens the trace. Run this after a wording/layout/plot-styling change; only
  re-run the heavy script if the underlying trace changed or a new summary column is
  needed.

This split exists because the trace itself (AZ3's full run is 24 GB) is the expensive
resource — re-reading it for every small report tweak would make iteration on the report
itself painfully slow. The heavy script's own final step calls `build_report.build_report()`
directly, so a single invocation still produces the finished HTML — the split only matters
when you want to regenerate the report *without* touching the trace again.

## Running it on a new trace

```bash
pixi run python scripts/full_dataset_characterization.py \
    --trace-path results/traces_full/<Model>.nc \
    --output-dir results/artifacts/<model>_full_characterization \
    --model-name <Model> \
    --thin-multimodality 4 \
    --n-example-areas 6
```

To rebuild just the report from an existing output directory (no trace access):

```bash
pixi run python results/artifacts/<model>_full_characterization/build_report.py \
    results/artifacts/<model>_full_characterization
```

Note the reproduction script lives *inside* the output directory it was built for
(alongside its CSVs) — the canonical copy of the HTML-assembly logic itself is read from
`results/artifacts/az3_full_characterization/build_report.py` by the heavy script
regardless of `--output-dir` (so a scratch/smoke-test run still exercises the real report
builder), but once an output directory has its own CSVs, its own copy of
`build_report.py` is the one to run directly for pure reproduction.

## What gets computed, and the tradeoffs behind each choice

### Trace loading (`load_trace_no_warmup`)

Uses `az.from_netcdf` — in the installed arviz version (1.2) this is a thin wrapper around
`xr.open_datatree`, which is **lazy by default**: a variable's bytes are only read off disk
when something calls `.values`/`.compute()` on it. Because this pipeline never touches
`trace.warmup_posterior`/`trace.warmup_sample_stats` (irrelevant to every diagnostic here),
their ~50% share of the file's size is never actually paid for in memory or read time,
despite `from_netcdf` nominally "opening" every group. Confirmed practically: AZ3's full
trace is 24 GB on disk (including warmup), but the pipeline's resident memory stays in the
~15-20 GB range — the six `(chain, draw, area, year)`-shaped posterior variables it does
touch (`z`, `delta`, `P_like_pointwise`, `E_like_pointwise`, `resp_noise_P`, `resp_noise_E`)
plus the `log_likelihood` group's `P_like`/`E_like`, each ~2.4 GB at 4 chains × 1500 draws ×
4987 areas × 10 years. Comfortably inside the machine's 128 GB, but worth keeping in mind if
a future finalist trace is bigger still, or has more `(chain, draw, area, year)`-shaped
variables — don't call `.load()`/`.values` on anything not needed for a specific summary.

### Area/borough summary

Reuses existing `diagnostics.py` functions directly rather than reimplementing anything:
`z_flatness_summary` (does z track P/E activity at all), `z_identifiability_summary`
(per-area year-allocation confidence), `_check_calibration`/`_check_census_constraint`/
`_check_morans_i` (aggregate scalars). `compute_resp_noise_summary` and
`compute_pareto_k_summary` (this script's own additions) degrade gracefully (return `None`)
for models without a noise-mixture likelihood or usable `log_likelihood` group — the
pipeline works on any registered model, not just AZ3-shaped ones.

### Multimodality scan at full scale — thinning tradeoff

`detect_z_multimodality` runs one KDE fit per `(area, year)` cell. At 4987 areas × 10 years
= 49,870 cells, running it at full draw count (4 chains × 1500 draws = 6000 samples/cell)
would be ~25× the 200-area development sample's already-nontrivial cost. `--thin-multimodality`
(default 4) keeps every Nth draw before scanning — trades some per-cell KDE resolution for a
large constant-factor speedup. This is a scale-driven compute tradeoff, not an untested
shortcut: the *method* (`_detect_modes`'s prominence-filtered KDE) was already validated at
full draw count on the 200-area sample in `az-family-work-plan.md` Phase 3 (E01002702's known
bimodal/unimodal cells). If a future run needs finer resolution on specific flagged cells, rerun
`detect_z_multimodality` directly (unthinned) against just those `(area, year)` pairs rather
than lowering the thinning factor for the whole scan.

### Deep-dive example selection

Reuses `select_spike_tracking_areas` (category-based: under-tracked spikes, worst Pareto-k,
P/E spike-year disagreement, well-tracked contrast) plus `REFERENCE_AREAS`, then adds two
full-scale-specific contrasts not covered by that selector: the most-multimodal area, and a
unimodal-but-heavily-noise-flagged area (a genuine confident-noise case, distinct from
genuine ambiguity). The whole-draw mode-decomposition plot (`plot_z_area_modes`) for each
selected example is rendered once, at generation time, and saved as a static PNG — it needs
the full per-draw `z` array, which is deliberately **not** carried into the CSVs (would
defeat the "reproduce without the trace" goal), so regenerating it requires rerunning the
heavy script, not `build_report.py`.

### What's in each CSV

| File | Grain | Key columns |
|---|---|---|
| `area_summary.csv` | 1 row/area | `range_z`, `is_flat`, `flat_despite_active`, `max_rhat`, `n_low_confidence_years`, `confident`, `mean_resp_noise_{P,E}`, `max_pareto_k_{P,E}`, `n_multimodal_years` |
| `borough_summary.csv` | 1 row/borough | means of the above over member areas |
| `multimodal_cells.csv` | 1 row per (area, year) with ≥2 modes | `n_modes`, `mode_locations` |
| `morans_i_resp_noise_by_year.csv` | 1 row/year | Moran's I, p-value, z-score on mean `resp_noise_P` |
| `example_areas.csv` / `example_areas_timeseries.csv` | 1 row/example area / 1 row per (example area, year) | selection reason, `P_obs`/`E_obs`/`z_mean`/`z_lo`/`z_hi`/`resp_noise_{P,E}` — everything `build_report.py` needs to redraw each panel without the trace |
| `scalar_summary.json` | whole run | flatness/coverage/census/Moran's-I aggregates, shared hyperparameter posterior mean/sd |
| `manifest.json` | whole run | trace path, data path, thinning factor used, file listing — read by `build_report.py` to know where to reload the GeoDataFrame from |

## Known limitations / honest gaps

- The multimodality scan's thinning factor trades resolution for speed (see above) —
  treat `n_multimodal_years`/`multimodal_cells.csv` as a full-scale-appropriate estimate,
  not as precise as the 200-area development sample's own (unthinned) scan.
- `compute_resp_noise_summary`'s per-area means use a fixed active-cell threshold
  (`|P_obs|`/`|E_obs| > 3.0`, matching `z_flatness_summary`'s own convention) — areas with
  no active cells get `NaN`, not zero, so aggregate means/plots should account for that.
- Borough-level choropleth maps in `build_report.py` reload the GeoDataFrame via
  `load_data(data_path)` — fast, but does re-run outlier exclusion; if that logic changes
  between the heavy run and a report-only rebuild, the two could show slightly different
  excluded-area sets. Not currently guarded against.
