# housing-projections

Bayesian (PyMC) models that infer, with uncertainty, the annual dwelling-stock change (`z`,
per LSOA per year) underlying two noisy, disagreeing observation sources — planning
completions and OS Address Base net uprn changes — for London LSOAs. Both sources are recorded
every year but are individually unreliable; the Census dwelling count is trustworthy but only
available once a decade, so it can pin down the *total* change over ten years without saying
anything about which years that change happened in. Every model in this repo uses the Census
total to anchor `z`'s sum across years, and the two model families differ in how hard that
anchor is enforced: the `M0`-`M16` progression treats it as a soft constraint (a likelihood
term pulling `sum(z)` *close to* the Census total), while the `AZ0`-`AZ5` family builds it into
the prior so `sum(z)` matches the Census total *exactly*, by construction.

## Results

- [Stakeholder summary](docs/az3-stakeholder-summary.md) — write-up of the AZ3 findings
- [Output files](results/artifacts/az3_year_estimates/) — the underlying per-area/per-year estimates (CSV/GeoJSON, see the folder's own README)
- [Live dashboard](https://conor-dempsey-london.github.io/dwelling-change-dashboard/) — interactive map/chart dashboard, hosted on GitHub Pages

## Requirements

- Python >= 3.12.9
- [pixi](https://pixi.sh) for environment management
- Access to the private [pld-database-live](https://github.com/JamesB686/pld_database_live)
  repo (pulled automatically by `pixi install` once you have git access to it)
- Raw input data (see [Data](#data) below) — **not included in this repo**

## Installation

```bash
git clone https://github.com/conor-dempsey-london/housing-projections
cd housing-projections
pixi install
cp .env.example .env   # then fill in the paths described below
```

`.env` variables (read by `src/housing_projections/config.py`):

| Variable       | Required | Description                                              |
|----------------|----------|-----------------------------------------------------------|
| `PROJECT_ROOT` | yes      | Repo root; anchors the default `results`/`traces` paths   |
| `DATA_PATH`    | yes      | Root directory of the raw input data files (see below)    |
| `RESULTS_DIR`  | no       | Defaults to `PROJECT_ROOT/results`                         |
| `TRACES_DIR`   | no       | Defaults to `PROJECT_ROOT/results/traces`                  |
| `REPORT_PATH`  | no       | Defaults to `PROJECT_ROOT/results/report.html`             |

## Data

`DATA_PATH` must point to a directory containing two files that are **not checked into this
repo** and must be sourced separately:

```
<DATA_PATH>/
├── pld/
│   └── lsoa_completions_time_series_pivot.csv        # LSOA-level planning completions
└── ben/
    └── final_residential_uprn_net_changes_by_oa_fy (1).csv   # OA-level OS Address Base net uprn changes
```

`housing-projections run-models` (and every other command that loads data) calls
`validate_data_path()` first and will fail fast with a clear message naming whichever of these
two files is missing.

Census dwelling counts, the 2011→2021 geography crosswalk, LSOA boundaries, and the
LSOA→borough lookup are **not** static files — they're pulled at runtime through the `gla_data`
package (`gla_data.load_census_dwellings`, `.crosswalk`, `.load_boundaries`,
`.load_geography_lookup`), which is installed as a normal pixi/PyPI dependency and handles its
own data access.

## Development

```bash
pixi run test          # pytest — synthetic fixtures, no data access or sampling needed
pixi run lint          # ruff check src/housing_projections tests
pixi run typecheck     # mypy src/housing_projections
```

Run a single test, or re-enable the sampling-heavy tests that are deselected by default:

```bash
pixi run pytest tests/test_models.py::test_name
pixi run pytest -m slow      # includes tests that actually call pm.sample
```

## Usage

`pyproject.toml` defines a pixi task with the same name as each `housing-projections`
subcommand (`run-models`, `compare`, `diagnose`, `multimodality` → `check-multimodality`,
`report`) as a shorthand for running it with every default — e.g. `pixi run run-models` is
exactly `housing-projections run-models` with no flags. Both forms accept extra CLI flags
appended after the command/task name (`pixi run run-models --models AZ0a --n-areas 50` works
just as well as calling `housing-projections` directly), so which one to type down to taste —
`pixi run <task>` guarantees the pixi environment is activated first, while `housing-projections
<command>` assumes you're already inside `pixi shell` or have otherwise activated it. The
examples below use whichever form makes the specific flags being demonstrated clearest; treat
them as interchangeable. All subcommands accept `--help` for the full flag list.

### `run-models` — sample models and save traces

```bash
# Sample every registered model on the default 200-LSOA spatial sample
pixi run run-models

# Sample specific models only, on a smaller subsample, skipping nutpie
housing-projections run-models --models M0,M1,AZ0a --n-areas 50 --no-nutpie

# Save to a non-default traces directory
housing-projections run-models --models AZ0a --traces-dir results/traces/experiment_1
```

### `compare` — LOO comparison + z sensitivity across saved traces

```bash
pixi run compare

# Restrict to specific models, or a non-default traces directory
housing-projections compare --models M0,M1,AZ0a --traces-dir results/traces/experiment_1
```

### `diagnose` — r-hat / divergence / calibration summary per model

```bash
pixi run diagnose

# Tighten the r-hat threshold, restrict to specific models
housing-projections diagnose --models AZ0a,AZ1b --rhat-threshold 1.005

# Fold in the multimodality classification before reporting headline r-hat/ESS
housing-projections diagnose --adjust-for-multimodality
```

### `check-multimodality` — classify per-area/per-scalar multimodality

```bash
pixi run multimodality

# Restrict to one model and one lag-weights variable
housing-projections check-multimodality --models AZ1b --lag-var lag_P_lambda_weights

# Also attempt the informed-init reseeding fix for stuck_fixable areas
housing-projections check-multimodality --models AZ1b \
  --lag-var lag_P_lambda_weights --resolve --resolve-chains 16
```

### `report` — self-contained HTML analysis report

```bash
pixi run report

# Custom title/output path, restricted to specific models
housing-projections report --models AZ0a,AZ1b,AZ4b \
  --output results/report_az_round.html --title "AZ round comparison"
```

### Example workflow: iterating on a model with the pixi tasks

A typical loop when developing or tweaking a model (e.g. testing a new `AZ*` variant) chains
the pixi tasks together, checking convergence and spike-tracking before moving on to
comparison:

```bash
# 1. Sample just the model(s) you're iterating on, on a small subsample for a fast cycle
pixi run run-models --models AZ1c --n-areas 50

# 2. Check convergence first — bad r-hat/divergences before anything else
pixi run diagnose --models AZ1c

# 3. If diagnose flags elevated r-hat/low ESS, classify whether it's genuine
#    multimodality before chasing it as a bug
pixi run multimodality --models AZ1c

# 4. Once convergence looks sound, resample on the full 200-area set
pixi run run-models --models AZ1c

# 5. Compare against the existing baseline and neighbouring variants
pixi run compare --models AZ0a,AZ1c

# 6. Generate the HTML report and eyeball spike-tracking plots for the reference LSOAs
#    before calling the iteration accepted or rejected
pixi run report --models AZ0a,AZ1c --output results/report_az1c.html
```
