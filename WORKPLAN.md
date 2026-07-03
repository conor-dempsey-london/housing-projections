# Housing Projections — Work Plan

*Status as of 2026-07-03. All items below are proposed; none have been started unless noted.*

---

## Part 1 — Code and package improvements

### 1.1 Separate I/O from computation in `load_data`

`load_data` currently does I/O (four external calls: `load_csv`, two `gla_data.load_census_dwellings`, `gla_data.load_boundaries`, `gla_data.aggregate`) interleaved with ~60 lines of joins, reshaping, and column renaming. This means the transform logic can only be tested by mocking the I/O.

**Action:** Extract a pure function `_build_gdf(completions, dwellings_2011_xw, dwellings_2021, df_ben, lsoa_gdf)` that takes the already-loaded DataFrames and does the merge/reshape/column-insertion work. `load_data` becomes a thin shell that calls the four external sources and passes their results to `_build_gdf`. This makes the merge logic unit-testable without any mocking.

### 1.2 Lint and typecheck tasks in pixi.toml

No `lint` or `typecheck` tasks are configured. The audit checklist treats these as prerequisites; currently they can't be run.

**Action:** Add `pixi run lint` (ruff) and `pixi run typecheck` (mypy or pyright) tasks to `pyproject.toml`. Add type annotations to all public functions as a separate pass once typecheck is wired up.

### 1.3 Test coverage for EDA subpackage

`eda/` has four modules with no tests. The plot functions are hard to unit-test meaningfully, but the compute functions are not:
- `compute_agreement_stats(gdf)` → returns a dict
- `compute_overall_correlation(gdf)` → returns a dict
- `classify_lsoas(gdf)` → returns a DataFrame

**Action:** Add `tests/test_eda.py` with unit tests for these three functions using `synthetic_gdf` from conftest.

### 1.4 Docstring consistency pass

Several public functions have no docstring (`load_data` now has one, but `make_data_dict` just got its first). Others (`compute_morans_i`, `build_weights_libpysal`, etc.) have Google-style `Parameters/Returns` sections but some are missing `Raises:` entries where they clearly raise.

**Action:** Audit all public functions against the Google docstring style. Fill in missing `Parameters`, `Returns`, and `Raises` sections. Specifically: `morans_i`-family, `build_weights_*`, `select_spatial_sample` (partial), `full_report`, `run_comparison_reports`.

### 1.5 Builder functions module-level visibility in `models.py`

`models.py` contains several module-level builder functions (`build_z_prior`, `build_lag`, `build_spatial_misallocation`, etc.) that are used by the model classes but aren't prefixed with `_`. They are not in `__init__.py` or `__all__`, so not part of the public API. But they are technically importable and appear alongside the public model classes in the module namespace.

**Action:** Prefix these builder functions with `_` (`_build_z_prior`, `_build_lag`, etc.) and add `__all__` to `models/models.py` listing only the model classes.

### 1.6 `data_path` convention and configuration

`load_data(data_path)` requires the caller to know the on-disk directory layout. The directory structure (subfolders `pld/`, `ben/`) is implicit and undocumented. If the path changes, callers get unhelpful `FileNotFoundError`s.

**Action:** Document the expected directory structure in `load_data`'s docstring. Optionally, add a `validate_data_path(data_path)` helper that checks for the expected files and raises a clear `FileNotFoundError` with a description of what's missing before the load attempt.

---

## Part 2 — Model extensions

The current model family (M0 → M6) has progressively added: hierarchical structure (M0h), sparsity (M1), per-source noise (M2), temporal lag (M3), zero-inflation (M4), asymmetric missingness (M5), two-component noise (M5b), and spatial misallocation (M6). The next natural extensions fall into three themes: **prior structure**, **observation model**, and **projection beyond the intercensal window**.

### 2.1 M7: Temporal random walk prior on z

**What it addresses:** Currently z is i.i.d. across years within an area, given the global mean and spread. In reality, dwelling change is autocorrelated year-on-year — a development site takes several years to complete; a large scheme may produce a spike that then returns to zero. The i.i.d. assumption means the model cannot represent this structure.

**Proposal:** Replace the year-i.i.d. Normal prior with an AR(1) or random walk prior:

```
z[a, 0]   ~ Normal(mu_area, sigma_init)
z[a, t]   ~ Normal(rho * z[a, t-1] + (1-rho) * mu_area, sigma_innov)
```

where `rho ~ Beta(8, 2)` (prior mean 0.8, strong autocorrelation) and `sigma_innov ~ HalfNormal(sigma=10)`.

This would substantially change the posterior shape for areas with multi-year development patterns. It is a significant change to the core prior and should be compared against M5b/M6 using LOO.

**Diagnostics needed:** Time-series plot of `z` posterior per area (currently `plot_sample_areas` shows this but the i.i.d. prior makes the draws look noisy year-to-year — AR(1) would smooth them).

### 2.2 M8: Hierarchical Borough-level structure

**What it addresses:** London has 33 boroughs, each containing ~150 LSOAs. Currently all LSOAs share a single global mu_global and sigma_slab — the model treats Havering and Hackney identically at the hyperprior level. Borough membership is a strong prior signal: development pressure, planning policy, and housing market dynamics vary dramatically by borough.

**Proposal:** Two-level hierarchy: global hyperprior → borough-level means → LSOA-level z.

```
mu_global           ~ Normal(D_full_mean / n_years, sigma=5)
sigma_borough       ~ HalfNormal(sigma=5)
mu_borough[b]       ~ Normal(mu_global, sigma_borough)       # per borough mean
sigma_slab          ~ HalfNormal(sigma=20)
z[a, t]             ~ Normal(mu_borough[borough[a]], sigma_slab)
```

`borough[a]` is a static integer array mapping LSOAs to their borough index, derivable from `gla_data.load_boundaries(geography='lsoa')` joined on borough LSOA-to-borough crosswalk.

**Data requirement:** Need `LSOA21CD → LAD21CD` mapping (available from `gla_data`).

**Value:** Produces borough-level summaries directly from the posterior (posterior mean of `mu_borough`), which are directly useful to planners. Currently, borough aggregates are computed post-hoc by summing LSOA z posteriors, which loses the borrowing-of-strength.

### 2.3 M9: Time-varying observation noise

**What it addresses:** Planning data quality is not uniform across the inference window (2012–2021). The PLD system underwent changes in reporting around 2013–2016, and COVID-19 affected completions reporting in 2019–2021. A fixed sigma_obs over all years is likely too rigid.

**Proposal:** Model `sigma_obs_plan` as a year-level random effect:

```
sigma_base_plan    ~ HalfNormal(sigma=5)
sigma_year_offset  ~ HalfNormal(sigma=2, shape=n_years)
sigma_obs_plan[t]  = sigma_base_plan + sigma_year_offset[t]
```

The year-indexed sigma propagates through to the planning likelihood.

**Simpler alternative:** Fit year-fixed-effect sigmas — treat each year's observation noise as a free parameter with a shared HalfNormal prior. This is equivalent but avoids the additive decomposition.

**Diagnostic check:** After fitting, inspect sigma_obs_plan by year — a spike in 2019–2021 would confirm the COVID signal. This could be a standalone diagnostic added to `full_report`.

### 2.4 M10: Separate lag for BEN source

**What it addresses:** BEN is currently assumed lag-free (z_t directly predicts E_t). But BEN is derived from UPRN net changes which are compiled quarterly and may have their own short recording delay, particularly for conversions and demolitions which take longer to register in UPRN.

**Proposal:** Add a short (max_lag=1 or 2) lag for BEN in parallel with the planning lag. The BEN lag weights `kappa ~ Dirichlet([3, 1])` would express a prior that most BEN observations are contemporaneous (lag 0) with a small probability of lag 1.

**Risk:** Adds identifiability tension — both P and E would now be explaining z through a lagged relationship, creating more room for the model to fit noise by adjusting lag weights. Would need to verify with a prior predictive check that the joint lag structure doesn't create degenerate posteriors.

### 2.5 M11: Robust outlier model (replacing pre-filtering)

**What it addresses:** Currently outlier areas are excluded before modelling (`apply_outlier_exclusion`). This is a pragmatic choice but it is a prior decision made outside the model. Areas that are excluded might contain real signal (e.g., a large regeneration scheme that genuinely produces a 3000-dwelling year).

**Proposal:** Replace outlier exclusion with a contamination component in the observation model. Add a fourth mixture component to the planning likelihood alongside the zero-inflation and tight/loose StudentT components:

```
P_like[a,t] ~ pi_outlier * Normal(0, sigma_outlier)      # extreme contamination
             + (1-pi_outlier) * [existing M5b mixture]
```

where `sigma_outlier` is large (~500 dwellings) and `pi_outlier ~ Beta(1, 99)` (prior mean 0.01 — very rare).

This allows the model to self-identify and down-weight anomalous observations without excluding areas entirely. `apply_outlier_exclusion` would remain as a utility but become optional rather than a required preprocessing step.

### 2.6 LOO-based model comparison

**What it addresses:** Model comparison between M3–M6 is currently visual and informal (residual plots, calibration checks). No information criterion is computed.

**Action:** Add `compute_model_comparison(traces)` to `diagnostics.py` using `az.compare(traces, ic='loo')`. This runs LOO-CV using the posterior log-likelihood groups (which are already saved in traces via `idata_kwargs={'log_likelihood': True}`). The comparison table directly quantifies how much each model extension improves out-of-sample predictive accuracy.

**Output:** DataFrame with `loo`, `se`, `p_loo`, `d_loo` for each model — directly interpretable and worth adding to `full_report` when multiple traces are available.

### 2.7 Projection extension (beyond 2021)

**What it addresses:** All current models are retrospective — they estimate z for 2012–2021 given the two census anchor points. Producing housing projections requires forecasting z beyond 2021 under some assumptions about future trends.

**Proposal (two-stage approach):**

1. **Fit the interpolation model** on the 2012–2021 window to get a posterior over z and all model parameters (as now).
2. **Extend the generative model** to produce posterior predictive draws of z for 2022–2030 by:
   - Drawing from the fitted prior structure (e.g., AR(1) from M7 if implemented, or sampling from the posterior `mu_slab`, `sigma_slab`)
   - Applying planning pipeline assumptions (a planning lag applied to future PLD completions data where available, or scenario-based assumptions where not)
   - Constraining to a new census or mid-year estimates when available

This is a substantial research task that would likely live in a new notebook (`3.0-sd-projections.py`) and require careful thought about how to represent scenario uncertainty alongside model parameter uncertainty.

---

## Suggested sequencing

| Priority | Item | Effort | Value |
|---|---|---|---|
| High | 1.2 Lint/typecheck tasks | Small | Unblocks future quality checks |
| High | 1.1 I/O separation in `load_data` | Small | Improves testability |
| High | 2.6 LOO model comparison | Small | Immediate analytical value |
| High | 1.3 EDA tests | Small | Fills last test coverage gap |
| Medium | 2.1 M7 Random walk prior | Medium | Addresses fundamental prior assumption |
| Medium | 2.2 M8 Borough hierarchy | Medium | Highest direct planning value |
| Medium | 1.5 Builder functions prefixed with `_` | Small | Tidying |
| Medium | 1.4 Docstring consistency | Small | Quality |
| Medium | 1.6 `data_path` validation helper | Small | Developer experience |
| Low | 2.3 M9 Time-varying sigma_obs | Medium | Interesting but secondary |
| Low | 2.4 M10 BEN lag | Medium | Risk of identifiability issues |
| Low | 2.5 M11 Robust outlier model | Large | Principled but complex |
| Low | 2.7 Projection extension | Large | Major research task |
