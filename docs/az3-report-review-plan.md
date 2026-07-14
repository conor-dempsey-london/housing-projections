# AZ3 full-characterization report review — findings & follow-ups

Review of `results/artifacts/az3_full_characterization/report.html`, started
2026-07-14. All items below are now resolved — this doc is kept as the record
of what was found and why, not as an open task list.

## Round 1 (no trace access — CSV/PNG-only changes in `build_report.py`)

- **#2 — noise-branch responsibility not shown.** Deep-dive P_obs/E_obs bars
  coloured by per-year `resp_noise_P`/`resp_noise_E` (already in
  `example_areas_timeseries.csv`), with a callout explaining the mechanism via
  E01033926. **Superseded by round 2, item B1 below** — the red/green
  colouring was hard to read and was replaced with plain colours + hatching.
- **#4 — D missing from example plots.** Added a `D/n_years` axhline and
  `D=...` in the title of the main deep-dive bar-chart panel, pulled from
  `area_summary.csv`.
- **#5 — mode-decomposition plot shown for unimodal areas.** `build_report.py`
  skips embedding `mode_decomposition_{area}.png` when `n_multimodal_years==0`.
- **#6 — confidence breakdown section.** New "Area-Level Confidence &
  Reliability" section: active-areas-only, ranked top/bottom-50 tables.
- **#7 — unused CSV columns surfaced.** Added a `max_rhat` vs
  `n_multimodal_years` scatter (coloured by `mean_resp_noise_P`), a
  `frac_bad_k_P` finding referencing `docs/model-evaluation-methods.md`, and
  two new borough choropleths (`max_pareto_k_P`, `frac_active_P_high_noise`).

## Round 2 (2026-07-14, second pass)

### B1 — noise marker was hard to read

Replaced the red/green `RdYlGn_r` bar colouring (round 1's #2 fix) with plain
`steelblue`/`coral` bars plus a black diagonal **hatch** (`///`) on any bar
whose `resp_noise_{P,E}` exceeds 50% — a binary "trust vs. noise-flagged"
marker instead of a continuous colour scale to decode, with a one-line legend
entry (`hatched = P(noise) > 50%`). Verified visually on E01033926 and the new
confident-example panels — much cleaner, no colorbar needed.

### B2 — confidence broken down by D band

Added a grouped bar chart + table to the confidence section: active areas
binned into `D<0` / `0-10` / `10-25` / `25-50` / `50-100` / `100-250` / `250+`,
showing fraction fully-confident and fraction with ≥1 multimodal year per
band. **Finding**: confidence collapses sharply with activity — `D` 0–25 bands
are ~100% confident and ~0–7% multimodal; `D>250` areas are only 4% confident
and 97% multimodal. Bigger intercensal change means more P/E activity to
reconcile, which is exactly where the year-allocation ambiguity this whole
report is about actually bites.

### B3 — confident, high-activity example areas

Added 4 new example areas (selected live from the trace, no pre-existing
selector matched this criterion): `has_active_year & n_low_confidence_years==0
& n_multimodal_years==0`, ranked by peak `max(|P_obs|, |E_obs|)`:
`E01034220` (D=719, peak activity 1227), `E01000943` (D=-29, peak activity
1037), `E01033098` (D=795, peak activity 836), `E01035496` (D=978, peak
activity 813). E01034220's panel is a clean illustration of the whole
noise/confidence mechanism working as intended: z tracks the un-hatched
(trusted) P/E bars closely and ignores the hatched (noise-flagged) spikes,
with zero residual year-allocation ambiguity.

### Deferred items — now resolved

- **#1 — "two modes" claimed, plot shows one (E01004686).** Confirmed:
  rerunning `plot_z_area_modes` with `min_cluster_frac=0` (nothing dropped)
  shows a genuine but small second scenario — **3% of draws** at k=2
  (`top-2-year concentration=62%`); at k=3 the split refines to 89%/8%/3%
  (`73%` concentration). So the per-cell KDE scan wasn't wrong to flag 3
  bimodal years — there IS a second story, it's just a minority one that sits
  right at the `min_cluster_frac=0.03` cutoff, which is why it flickered
  in/out of the rendered plot. Fixed at the source: `plot_mode_decomposition_
  if_available` no longer overwrites `plot_z_area_modes`'s own title, so the
  concentration score (and therefore this ambiguity) is now visible directly
  on the regenerated PNG instead of silently discarded.
- **#3 (narrow) — E01002794 mode misalignment.** Investigated with
  `n_clusters=2/3/4` (`min_cluster_frac=0`) and a pairwise draw-correlation
  check across the 5 individually-flagged years (2013/2016/2017/2020/2021).
  Result: k=2's concentration is only **46%** — `plot_z_area_modes`'s own
  diagnostic correctly flags this as `LOW: scenarios may not be real`, which
  the old title-overwrite bug had been hiding. The 5 flagged years are all
  weakly *anti*-correlated with each other (−0.07 to −0.18, no strong pair),
  consistent with them being largely independent, mutually-exclusive ways to
  absorb the same zero-sum slack rather than 2 (or any small number of)
  coherent stories. Concentration keeps climbing with k (61% at k=3, 71% at
  k=4) — the honest read is that this area's uncertainty is closer to
  diffuse/near-exchangeable across ~5 candidate years than a clean 2-corner
  (or even 3-corner) decomposition, echoing the E01035709 cautionary case
  already documented in `plot_z_area_modes`'s own docstring. Regenerated PNG
  now shows the `LOW` warning directly in the title.
- **#3 (broad) — richer multimodality characterization at full scale.**
  Built an adaptive re-clustering pass (`scripts/az3_deep_dive_followups.py`,
  written to `mode_recharacterization.csv`): for every area with ≥3
  individually-flagged multimodal years (999 candidates), tests whole-draw
  k-means at k=2/3/4 and records the largest k where every resulting cluster
  still clears a 5%-of-draws floor (a real scenario, not a k-means sliver).
  **Finding: 734 of 999 areas (73.5%) genuinely need 3+ scenarios** — the
  report's blanket 2-scenario default was understating the real structure for
  most heavily-multimodal areas, not just the 1-2 hand-picked examples this
  review started from. Surfaced in the report as a new "Areas needing more
  than 2 scenarios" subsection under Multimodality at Full Scale. The
  per-example deep-dive plots still default to k=2 (unchanged) — bumping
  `n_clusters` per-area from this table to the deep-dive plots themselves
  wasn't done and would be the natural next step if this becomes a recurring
  need, not a one-off.
- **#4 (remainder) — mode-decomposition PNGs lacked D/P/E.** Fixed at the
  source in `plot_mode_decomposition_if_available` (now threads `P_obs`,
  `E_obs`, `D`, `n_years` through to `plot_z_area_modes`) and regenerated all
  14 multimodal example PNGs. Deliberately did **not** pass
  `resp_noise_P`/`resp_noise_E` into this call — that would reintroduce the
  same red/green colour-by-noise style B1 just moved away from; these panels'
  P/E markers are plain, matching the main deep-dive chart's new hatch-based
  convention instead.
- **#5 (remainder) — redundant PNG generation at the source.** `main()` in
  `scripts/full_dataset_characterization.py` now gates
  `plot_mode_decomposition_if_available` on `n_multimodal_years > 0` before
  generating the PNG at all (previously only `build_report.py` skipped
  *embedding* it — the heavy script still rendered it unconditionally on every
  full rerun). Also fixed a latent inefficiency found while touching this
  code: the function used to call `trace.posterior['z'].values` itself on
  every invocation (a fresh multi-GB materialization per example area); it now
  accepts an optional pre-loaded `z_post` so the caller loads it once.

## Method note

All round-2 trace access went through one script
(`scripts/az3_deep_dive_followups.py`) that opens
`results/traces_full/AZ3.nc` exactly once, read-only, reuses
`area_summary.csv`/`example_areas.csv` already on disk instead of recomputing
anything cached there, and writes its own log to
`results/scratch/az3_deep_dive_followups_run.log`.
