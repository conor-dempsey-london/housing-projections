# Plan: CLI and Analysis Report

## Overview

Three deliverables on top of the existing model family (M0–M9):

1. **`sensitivity.py`** — new diagnostics for z model sensitivity
2. **`cli.py`** — `housing-projections` CLI with `run-models`, `compare`, `report` commands
3. **`html_report.py`** — self-contained HTML report generator

---

## 1. New module: `src/housing_projections/sensitivity.py`

The key research question is: _how much do our z estimates depend on model choice?_

### Functions

**`compute_z_model_sensitivity(traces)`**
- Input: `dict[str, az.InferenceData]` keyed by model name
- For each model, extract `z.posterior.mean(axis=(chain, draw))` → shape `(n_areas, n_years)`
- Output: `pd.DataFrame` with columns `model`, `LSOA_idx`, `year`, `z_mean`
  - Plus summary DataFrame with per-LSOA columns: `z_mean_across_models`, `z_std_across_models`, `z_range_across_models`

**`compute_model_agreement_matrix(traces)`**
- Compute correlation matrix of flattened z mean vectors across models
- Output: `pd.DataFrame` (model × model), symmetric, diagonal=1

**`compute_z_ensemble(traces, comparison_df=None)`**
- LOO-weight-averaged z posterior mean. If `comparison_df` is None, equal weights.
- Output: `np.ndarray` of shape `(n_areas, n_years)`

**`plot_z_sensitivity_map(gdf, sensitivity_df, col='z_std_across_models', title='')`**
- Choropleth map of z sensitivity (std across model posterior means)

**`plot_model_agreement_matrix(corr_df, title='')`**
- Annotated heatmap of model-to-model z correlation

**`plot_z_range_distribution(sensitivity_df, title='')`**
- Histogram of per-LSOA z-range (max_model_mean - min_model_mean) across all years

**`plot_sensitivity_vs_disagreement(sensitivity_df, gdf, title='')`**
- Scatter: source disagreement (PLD vs BEN divergence) vs model sensitivity

---

## 2. CLI: `src/housing_projections/cli.py`

Entry point for `pyproject.toml`'s `[project.scripts]` `housing-projections = "housing_projections.cli:main"`.

### Commands

```
housing-projections run-models
    --data-path PATH          raw data directory (required)
    --models M0,M3,M5,M6     comma-separated model names (default: all)
    --n-areas N               subsample for speed (default: all)
    --traces-dir DIR          where to save .nc files (default: results/traces)
    --no-nutpie               force PyMC sampler

housing-projections compare
    --traces-dir DIR          where to load .nc files from (default: results/traces)
    --models M0,M3,M5         which models to include (default: all found in dir)

housing-projections report
    --data-path PATH          raw data directory (required for EDA)
    --traces-dir DIR          (default: results/traces)
    --output PATH             HTML output path (default: results/report.html)
    --models M0,M3,M5,M6      models to include in walk-through (default: all found)
    --title STR               report title
```

### Pixi task aliases
```toml
run-models = "housing-projections run-models"
compare    = "housing-projections compare"
report     = "housing-projections report"
```

---

## 3. HTML report: `src/housing_projections/html_report.py`

Generates a self-contained `.html` file. Matplotlib figures are embedded as base64 PNG.
No external JS/CSS dependencies — plain HTML with inline styles.

### Report sections (in order)

#### 0. Executive summary table
| Model | LOO | ΔLOO | Divergences | z sensitivity (mean std) |

#### 1. Data and EDA
- Source comparison: `plot_cumulative_vs_intercensal`, `plot_annual_p_vs_e`
- Source agreement distribution: `compute_agreement_stats`, `plot_total_agreement`
- Spatial autocorrelation: `plot_morans_i_by_year`
- Temporal patterns: `plot_mean_trends`, `plot_crosscorrelations`
- Lag candidates: `plot_lag_candidates`

#### 2. Problem statement
Static prose section explaining:
- What `z` is (latent true annual dwelling change per LSOA)
- Why two imperfect sources (planning, BEN)
- What the census constraint is
- Why model choice matters

#### 3. Model walk-through (one sub-section per model in the comparison set)

For each model `Mk`:
- Short description (from `model.description`)
- Key diagnostic plot(s) that show what `Mk` adds:
  - M0: prior predictive z distribution
  - M1: z posterior mean by year (shows temporal variation)
  - M2: z posterior spread by area (shows area heterogeneity)
  - M3: lag weight posterior + planning residuals before/after lag
  - M4: zero-inflation probabilities + negative tail comparison
  - M5: asymmetric pi_miss (pos vs neg)
  - M6: alpha_spatial posterior + spatial z change map
  - M7: rho posterior + z trajectory for selected areas
  - M8: mu_borough by borough + between-borough variance
  - M9: sigma_obs_plan by year (shows time-varying noise)
- LOO improvement vs previous model (ΔLOO, standard errors)

#### 4. Full model comparison
- LOO table (`compute_model_comparison`)
- Model agreement matrix heatmap
- z sensitivity map (std of posterior z across models)
- z range distribution

#### 5. Summary and conclusions
- Which model is preferred (LOO)?
- How sensitive are z estimates to model choice?
  - Mean z std across models (aggregated over LSOAs/years)
  - Areas of high vs low sensitivity
  - Interpretation
- Ensemble estimate vs individual model estimates

### Key helper functions

```python
def fig_to_base64(fig) -> str:
    """Save matplotlib figure to base64 PNG string for embedding."""

def html_section(title, content_html, level=2) -> str:
    """Wrap content in a styled <section> block."""

def html_figure(fig, caption='') -> str:
    """Return <figure> HTML with embedded base64 PNG and caption."""

def generate_report(data, traces, post_preds, output_path,
                    model_classes=None, comparison_df=None, title=''):
    """
    Main entry point. data=make_data_dict output, traces=dict of InferenceData,
    post_preds=dict of posterior predictive InferenceData.
    Writes self-contained HTML to output_path.
    """
```

---

## 4. Tests

- `tests/test_sensitivity.py` — unit tests for compute functions (no traces required; use synthetic z arrays)
- `tests/test_cli.py` — argparse tests using `parse_args` directly, not subprocess
- No tests for `html_report.py` (pure rendering, no logic to test)

---

## 5. Execution order

1. `sensitivity.py` + tests  
2. `cli.py` (wire up `run-models` and `compare` first)  
3. `html_report.py` section by section  
4. Wire `report` command in CLI  
5. Update `__init__.py`, pyproject.toml tasks  
6. Lint pass + test run  
7. Commit  
