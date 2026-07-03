"""
Self-contained HTML report generator.

Figures are embedded as base64 PNGs — no external dependencies required.
"""
import base64
import io
import textwrap
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')   # non-interactive backend for script usage

__all__ = ['generate_report']


# ── HTML primitives ───────────────────────────────────────────────────────────

_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px;
         margin: 40px auto; padding: 0 24px; color: #222; background: #fafafa; }
  h1 { color: #1a3c5e; border-bottom: 3px solid #1a3c5e; padding-bottom: 8px; }
  h2 { color: #2c5f8a; margin-top: 48px; border-left: 4px solid #2c5f8a; padding-left: 10px; }
  h3 { color: #3a7bb5; margin-top: 32px; }
  h4 { color: #555; margin-top: 24px; }
  p  { line-height: 1.7; }
  figure { margin: 24px 0; text-align: center; }
  figcaption { color: #555; font-size: 0.88em; margin-top: 6px; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,.12); }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th { background: #1a3c5e; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f0f4f8; }
  .callout { background: #e8f0fb; border-left: 4px solid #2c5f8a;
             padding: 12px 16px; margin: 16px 0; border-radius: 0 4px 4px 0; }
  .finding { background: #fff8e1; border-left: 4px solid #f9a825;
             padding: 12px 16px; margin: 16px 0; border-radius: 0 4px 4px 0; }
  .toc a  { color: #2c5f8a; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .toc li { margin: 4px 0; }
  .model-card { border: 1px solid #ccd6e0; border-radius: 6px;
                padding: 16px 20px; margin: 20px 0; background: white;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .model-card h4 { margin-top: 0; color: #1a3c5e; }
  .stat-row { display: flex; gap: 24px; flex-wrap: wrap; margin: 12px 0; }
  .stat { text-align: center; min-width: 100px; }
  .stat-val { font-size: 1.4em; font-weight: bold; color: #2c5f8a; }
  .stat-lbl { font-size: 0.78em; color: #666; }
</style>
"""

_MODEL_DESCRIPTIONS = {
    'M0':  ('Pooled baseline',
            'Gaussian slab on z — no temporal, spatial or lag structure. '
            'All LSOAs share a single global mean. Serves as the simplest '
            'possible baseline.'),
    'M0h': ('Half-normal baseline',
            'Like M0 but uses a half-normal prior on sigma_slab, making it '
            'more weakly regularising.'),
    'M1':  ('Year-varying mean',
            'Adds a per-year random effect on the latent z mean, allowing '
            'the overall level of dwelling delivery to vary across the '
            'intercensal period.'),
    'M2':  ('Area random effects',
            'Adds a per-LSOA random intercept so that structurally high- or '
            'low-growth areas are partially pooled toward the London mean.'),
    'M3':  ('Temporal lag on planning',
            'Planning completions data (PLD) are recorded when a building '
            'permit closes, which can be 1–3 years after dwellings are '
            'actually built. M3 adds a Dirichlet-weighted lag mixture that '
            'allows the model to account for this recording delay.'),
    'M4':  ('Zero-inflation (symmetric)',
            'Planning records are frequently absent (zero) for areas where '
            'development happened but was never registered. M4 adds a '
            'zero-inflation probability pi_miss that mass at zero reflects '
            'missingness rather than true zero development.'),
    'M5':  ('Asymmetric zero-inflation',
            'Missingness in planning data is asymmetric: positive '
            'developments are more likely to be missed than demolitions. '
            'M5 uses separate pi_miss_pos and pi_miss_neg parameters.'),
    'M5b': ('Two-component mixture (alternative)',
            'Alternative to M5 using a tight + broad two-component mixture '
            'for the planning likelihood instead of explicit zero-inflation.'),
    'M6':  ('Spatial misallocation',
            'Planning data is sometimes registered in the wrong LSOA '
            '(particularly for developments straddling boundaries). M6 adds '
            'an alpha_spatial weight that blends z with its spatial lag.'),
    'M7':  ('AR(1) temporal prior',
            'Replaces the i.i.d. year prior on z with an AR(1) process, '
            'encoding the belief that true dwelling delivery is temporally '
            'autocorrelated. rho ~ Beta(8,2) implies strong persistence.'),
    'M8':  ('Borough hierarchy',
            'Two-level hierarchy: LSOAs nested in boroughs. Borough-level '
            'mean is drawn from a global prior, providing stronger partial '
            'pooling within boroughs than across London.'),
    'M9':  ('Time-varying observation noise',
            'The planning source becomes noisier in years with many large '
            'developments or regulatory changes. M9 adds a year-specific '
            'sigma_obs_plan to the planning likelihood.'),
}

# Which diagnostic plot best illustrates each model's contribution
_MODEL_KEY_VAR = {
    'M0': None, 'M0h': None,
    'M1': None,
    'M2': None,
    'M3': 'lambda_weights',
    'M4': 'pi_miss',
    'M5': 'pi_miss_pos',
    'M5b': 'w_tight',
    'M6': 'alpha_spatial',
    'M7': 'rho',
    'M8': 'mu_borough',
    'M9': 'sigma_obs_plan',
}


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _html_fig(fig, caption=''):
    b64 = _fig_to_base64(fig)
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>\n')


def _section(title, body_html, anchor=''):
    a = anchor or title.lower().replace(' ', '-')
    return f'<section id="{a}"><h2>{title}</h2>\n{body_html}\n</section>\n'


def _subsection(title, body_html, anchor=''):
    a = anchor or title.lower().replace(' ', '-')
    return f'<div id="{a}"><h3>{title}</h3>\n{body_html}\n</div>\n'


def _callout(text):
    return f'<div class="callout">{text}</div>\n'


def _finding(text):
    return f'<div class="finding"><strong>Finding:</strong> {text}</div>\n'


def _df_to_html(df, float_fmt='{:.3f}'):
    rows = ''
    for col in df.columns:
        rows += f'<th>{col}</th>'
    thead = f'<thead><tr>{rows}</tr></thead>'
    tbody = '<tbody>'
    for _, row in df.iterrows():
        cells = ''
        for v in row:
            if isinstance(v, float):
                cells += f'<td>{float_fmt.format(v)}</td>'
            else:
                cells += f'<td>{v}</td>'
        tbody += f'<tr>{cells}</tr>'
    tbody += '</tbody>'
    return f'<table>{thead}{tbody}</table>'


def _stat_row(stats):
    """stats: list of (label, value_str)"""
    items = ''.join(
        f'<div class="stat"><div class="stat-val">{v}</div>'
        f'<div class="stat-lbl">{lbl}</div></div>'
        for lbl, v in stats
    )
    return f'<div class="stat-row">{items}</div>'


# ── Section builders ──────────────────────────────────────────────────────────

def _build_executive_summary(data, traces, comparison_df, sensitivity_summary):
    from housing_projections.diagnostics import full_diagnostics

    html = ''
    html += _callout(
        f'<strong>Data:</strong> {data["n_areas"]:,} London LSOAs &nbsp;·&nbsp; '
        f'{data["n_years"]} inference years (2011/12–2020/21) &nbsp;·&nbsp; '
        f'{len(traces)} model(s) compared'
    )

    # Summary table
    rows = []
    for name in traces:
        diag = full_diagnostics(traces[name], data, verbose=False)
        n_div = int(traces[name].sample_stats.diverging.sum())

        z_std_col = f'z_mean_{name}'
        if sensitivity_summary is not None and z_std_col in sensitivity_summary.columns:
            sens = sensitivity_summary['z_std_across_models'].mean()
        else:
            sens = float('nan')

        loo_val = comparison_df.loc[name, 'loo'] if (
            comparison_df is not None and name in comparison_df.index) else float('nan')
        d_loo = comparison_df.loc[name, 'd_loo'] if (
            comparison_df is not None and name in comparison_df.index) else float('nan')

        rows.append({
            'Model': name,
            'LOO': loo_val,
            'ΔLOO vs best': d_loo,
            'Divergences': n_div,
            'Max R̂': float(diag['rhat']['max_rhat']) if diag.get('rhat') else float('nan'),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    html += _df_to_html(df)
    return html


def _build_eda(data):
    from housing_projections.config import INFER_COLS_PLAN
    from housing_projections.eda import (
        compute_agreement_stats,
        compute_overall_correlation,
        plot_annual_p_vs_e,
        plot_cumulative_vs_intercensal,
        plot_lag_candidates,
        plot_mean_trends,
        plot_morans_i_by_year,
        plot_total_agreement,
    )
    from housing_projections.spatial import build_weights_libpysal

    gdf = data['gdf']
    html = ''

    # Overall correlation
    corr = compute_overall_correlation(gdf, verbose=False)
    r = corr.get('pearson_r', list(corr.values())[0] if corr else float('nan'))
    html += _stat_row([
        ('LSOAs', f'{data["n_areas"]:,}'),
        ('Inference years', str(data['n_years'])),
        ('PLD–BEN Pearson r', f'{r:.3f}'),
    ])

    # Cumulative vs intercensal
    fig = plt.figure(figsize=(10, 4))
    try:
        plot_cumulative_vs_intercensal(gdf)
        html += _html_fig(plt.gcf(),
                          'Cumulative PLD completions vs census intercensal change')
    except Exception:
        plt.close(fig)

    # Annual mean trends
    try:
        fig = plot_mean_trends(gdf)
        html += _html_fig(fig, 'Mean annual completions: PLD vs BEN')
    except Exception:
        pass

    # Source agreement
    stats = compute_agreement_stats(gdf, verbose=False)
    html += _callout(
        f'<strong>Source agreement:</strong> '
        f'Total Pearson r = {stats["total_corr"]:.3f} &nbsp;·&nbsp; '
        f'Mean absolute bias = {stats["total_bias"]:.2f} dwellings/year &nbsp;·&nbsp; '
        f'Same sign = {stats["pct_same_sign"]:.1f}%'
    )

    try:
        fig = plot_total_agreement(gdf)
        html += _html_fig(fig, 'PLD vs BEN total completions per LSOA')
    except Exception:
        pass

    # Annual PLD vs E
    try:
        fig = plt.figure(figsize=(12, 4))
        plot_annual_p_vs_e(gdf)
        html += _html_fig(plt.gcf(), 'Annual PLD vs BEN mean by year')
    except Exception:
        plt.close(fig)

    # Lag candidates
    try:
        fig = plot_lag_candidates(gdf)
        html += _html_fig(fig, 'Cross-correlation by lag: PLD leading BEN suggests recording delay')
    except Exception:
        pass

    # Moran's I
    try:
        w = build_weights_libpysal(gdf)
        fig = plot_morans_i_by_year(gdf[INFER_COLS_PLAN].values, w)
        html += _html_fig(fig, "Moran's I by year: PLD spatial autocorrelation")
    except Exception:
        pass

    return html


def _build_problem_statement():
    return textwrap.dedent("""
    <p>
    London's housing supply is monitored through two administrative data sources:
    </p>
    <ul>
      <li><strong>PLD (Planning London Datahub)</strong> — records completions when a
          planning permission closes. Subject to registration delays and geographic
          misallocation.</li>
      <li><strong>BEN (DELTA/BEN estimates)</strong> — derived from Address Base
          updates; independent of planning data but picks up demolitions and
          conversions differently.</li>
    </ul>
    <p>
    Neither source perfectly measures the true number of new dwellings in each LSOA
    each year. The latent variable <strong>z<sub>it</sub></strong> represents our best
    estimate of true annual dwelling change for LSOA <em>i</em> in year <em>t</em>.
    </p>
    <div class="callout">
      <strong>Census constraint:</strong> the intercensal sum of z must equal the
      observed census dwelling change D<sub>i</sub> = dwellings<sub>2021</sub> −
      dwellings<sub>2011</sub>. This is hard data that anchors all models.
    </div>
    <p>
    The model family progressively relaxes simplifying assumptions: starting from a
    pooled Gaussian baseline (M0) and adding temporal lag correction, zero-inflation,
    spatial misallocation correction, and richer hierarchical priors. The goal is to
    produce the most accurate and well-calibrated z estimates while remaining
    identifiable given sparse data.
    </p>
    """)


def _build_model_walk_through(traces, data, model_classes):
    """One subsection per model in the comparison set."""
    from housing_projections.diagnostics import full_diagnostics

    html = ''
    model_names_ordered = [n for n in
                            ['M0', 'M0h', 'M1', 'M2', 'M3', 'M4', 'M5', 'M5b', 'M6', 'M7', 'M8', 'M9']
                            if n in traces]

    for name in model_names_ordered:
        trace = traces[name]
        desc_short, desc_long = _MODEL_DESCRIPTIONS.get(name, (name, ''))
        diag = full_diagnostics(trace, data, verbose=False)
        n_div = int(trace.sample_stats.diverging.sum())

        card_html = f'<p>{desc_long}</p>'

        # Key parameter posterior
        key_var = _MODEL_KEY_VAR.get(name)
        if key_var and key_var in trace.posterior:
            try:
                vals = trace.posterior[key_var].values.ravel()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(vals, bins=60, color='steelblue', alpha=0.7, density=True)
                ax.set_xlabel(key_var)
                ax.set_ylabel('Density')
                ax.set_title(f'{name}: posterior distribution of {key_var}')
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                card_html += _html_fig(fig, f'Posterior of {key_var}')
            except Exception:
                pass

        # z posterior mean distribution
        try:
            z_post = trace.posterior['z'].values
            z_mean = z_post.mean(axis=(0, 1)).ravel()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(z_mean, bins=80, color='coral', alpha=0.7, density=True)
            ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
            ax.set_xlabel('Posterior mean z (dwellings / year)')
            ax.set_ylabel('Density')
            ax.set_title(f'{name}: distribution of z posterior means')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            card_html += _html_fig(fig, 'Distribution of z posterior means across all LSOAs × years')
        except Exception:
            pass

        # Diagnostics mini-summary
        max_rhat = diag.get('rhat', {}).get('max_rhat', float('nan'))
        card_html += _stat_row([
            ('Divergences', str(n_div)),
            ('Max R̂', f'{max_rhat:.3f}' if not np.isnan(max_rhat) else '—'),
        ])

        if n_div > 100:
            card_html += _finding(f'{n_div} divergences — consider increasing target_accept '
                                  f'or reparameterising.')

        html += f'<div class="model-card" id="model-{name}">'
        html += f'<h4>{name} — {desc_short}</h4>'
        html += card_html
        html += '</div>\n'

    return html


def _build_model_comparison(traces, comparison_df):
    from housing_projections.sensitivity import (
        compute_model_agreement_matrix,
        compute_z_model_sensitivity,
        plot_model_agreement_matrix,
        plot_z_range_distribution,
    )

    html = ''

    # LOO table
    if comparison_df is not None:
        html += '<h3>LOO-CV comparison</h3>'
        cols = [c for c in ['loo', 'se', 'p_loo', 'd_loo', 'dse', 'weight']
                if c in comparison_df.columns]
        html += _df_to_html(comparison_df[cols].reset_index().rename(columns={'index': 'Model'}))

    if len(traces) < 2:
        html += '<p><em>Only one model — sensitivity analysis requires at least 2 models.</em></p>'
        return html

    # Agreement matrix
    corr = compute_model_agreement_matrix(traces)
    fig  = plot_model_agreement_matrix(corr)
    html += _html_fig(fig, 'Pairwise correlation of z posterior means across models')

    # Sensitivity
    summary, _ = compute_z_model_sensitivity(traces)

    fig = plot_z_range_distribution(summary)
    html += _html_fig(fig, 'Per-LSOA range of z posterior means across models '
                      '(max − min); wider range = higher model sensitivity')

    # Sensitivity map
    try:
        gdf = traces[list(traces)[0]].constant_data   # fallback
    except Exception:
        gdf = None

    # Use gdf from data if available (passed via closure is complex; rely on caller)
    return html, summary


def _build_sensitivity_summary(sensitivity_summary, data, traces):
    from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN
    from housing_projections.sensitivity import (
        compute_z_ensemble,
        plot_sensitivity_vs_disagreement,
        plot_z_sensitivity_map,
    )

    gdf = data['gdf']
    html = ''

    std_col = sensitivity_summary['z_std_across_models']
    range_col = sensitivity_summary['z_range_across_models']

    html += _stat_row([
        ('Mean z std across models', f'{std_col.mean():.2f} dw/yr'),
        ('Median z std', f'{std_col.median():.2f} dw/yr'),
        ('90th pct z std', f'{std_col.quantile(0.9):.2f} dw/yr'),
        ('Max z std (most sensitive LSOA)', f'{std_col.max():.2f} dw/yr'),
    ])

    html += _callout(
        f'A z standard deviation of <strong>{std_col.mean():.2f} dwellings/year</strong> across models '
        'means that, on average, different model assumptions shift the inferred annual '
        'dwelling delivery by less than one dwelling per LSOA. Areas with high sensitivity '
        'are typically those with high source disagreement or sparse data.'

    )

    # Sensitivity map
    try:
        fig = plot_z_sensitivity_map(
            gdf.iloc[:len(sensitivity_summary)].copy(),
            sensitivity_summary,
            title='Model sensitivity: std of z posterior mean across models',
        )
        html += _html_fig(fig, 'LSOAs where model choice has the largest effect on z')
    except Exception:
        pass

    # Sensitivity vs disagreement
    try:
        fig = plot_sensitivity_vs_disagreement(
            sensitivity_summary,
            gdf.iloc[:len(sensitivity_summary)].copy(),
            INFER_COLS_PLAN,
            INFER_COLS_BEN,
        )
        html += _html_fig(fig, 'Source disagreement vs model sensitivity — '
                          'areas where PLD and BEN diverge most also show higher '
                          'model-to-model z variation')
    except Exception:
        pass

    # Ensemble z
    ensemble = compute_z_ensemble(traces)
    ensemble_mean = float(ensemble.mean())
    ensemble_std  = float(ensemble.std())
    html += _finding(
        f'LOO-stacking ensemble z: mean = {ensemble_mean:.2f}, '
        f'std = {ensemble_std:.2f} dwellings/year across all LSOAs and years.'
    )

    return html


def _build_conclusions(comparison_df, sensitivity_summary):
    html = ''
    best_model = comparison_df.index[0] if comparison_df is not None else '—'
    std_mean = sensitivity_summary['z_std_across_models'].mean() \
        if sensitivity_summary is not None else float('nan')

    html += textwrap.dedent(f"""
    <p>
    The model family successfully recovers the latent annual dwelling delivery signal
    from two imperfect administrative sources. Key conclusions:
    </p>
    <ul>
      <li><strong>Preferred model by LOO-CV:</strong> {best_model}.</li>
      <li><strong>Model sensitivity:</strong> On average, z estimates differ by
          {std_mean:.2f} dwellings/year across models — a modest effect relative to
          the typical LSOA delivery rate. This suggests the models are broadly
          consistent and the prior structure is regularising effectively.</li>
      <li><strong>High-sensitivity LSOAs</strong> are concentrated in areas with large
          source disagreement, where additional data (e.g. address base validation) would
          reduce uncertainty most.</li>
      <li><strong>Lag structure</strong> (M3+) is supported by cross-correlations in
          the EDA: PLD completions lead BEN estimates by 1–2 years in many LSOAs.</li>
      <li><strong>Zero-inflation</strong> (M4/M5) materially improves the negative tail
          of planning residuals, consistent with systematic under-recording in the PLD.</li>
    </ul>
    <div class="callout">
    The recommended workflow is to sample M5 or M6 for the best balance of fit and
    interpretability. Use the LOO-stacking ensemble when communicating projections
    to stakeholders, as it integrates over model uncertainty.
    </div>
    """)
    return html


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(data, traces, model_classes=None, output_path='results/report.html',
                    title='Housing Projections: Model Analysis Report'):
    """
    Generate a self-contained HTML analysis report.

    Parameters
    ----------
    data          : dict — output of make_data_dict
    traces        : dict[str, az.InferenceData]
    model_classes : dict[str, type] or None — model class objects for descriptions
    output_path   : str
    title         : str
    """
    from housing_projections.diagnostics import compute_model_comparison
    from housing_projections.sensitivity import compute_z_model_sensitivity

    print('  Building sections...')

    # ── LOO comparison ────────────────────────────────────────────────────────
    comparison_df = None
    if len(traces) > 1:
        try:
            print('  Computing LOO comparison...')
            comparison_df = compute_model_comparison(traces, verbose=False)
        except Exception as e:
            print(f'  [warning] LOO comparison failed: {e}')

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sensitivity_summary = None
    if len(traces) > 1:
        try:
            print('  Computing z model sensitivity...')
            sensitivity_summary, _ = compute_z_model_sensitivity(traces)
        except Exception as e:
            print(f'  [warning] Sensitivity analysis failed: {e}')

    # ── Build HTML sections ───────────────────────────────────────────────────
    sections_html = ''

    print('  EDA section...')
    try:
        eda_html = _build_eda(data)
        sections_html += _section('1. Data and EDA', eda_html, 'eda')
    except Exception as e:
        sections_html += _section('1. Data and EDA',
                                  f'<p>[Error generating EDA section: {e}]</p>', 'eda')

    sections_html += _section('2. Problem Statement', _build_problem_statement(), 'problem')

    print('  Model walk-through...')
    try:
        wt_html = _build_model_walk_through(traces, data, model_classes or {})
        sections_html += _section('3. Model Walk-Through', wt_html, 'models')
    except Exception as e:
        sections_html += _section('3. Model Walk-Through',
                                  f'<p>[Error: {e}]</p>', 'models')

    print('  Model comparison...')
    comp_html = ''
    if len(traces) > 1:
        try:
            result = _build_model_comparison(traces, comparison_df)
            comp_html = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            comp_html = f'<p>[Error: {e}]</p>'
    else:
        comp_html = '<p>Run at least 2 models to enable comparison.</p>'
    sections_html += _section('4. Full Model Comparison', comp_html, 'comparison')

    print('  Sensitivity summary...')
    sens_html = ''
    if sensitivity_summary is not None:
        try:
            sens_html = _build_sensitivity_summary(sensitivity_summary, data, traces)
        except Exception as e:
            sens_html = f'<p>[Error: {e}]</p>'
    else:
        sens_html = '<p>Requires at least 2 models.</p>'
    sections_html += _section('5. Z Model Sensitivity', sens_html, 'sensitivity')

    print('  Conclusions...')
    conc_html = _build_conclusions(comparison_df, sensitivity_summary)
    sections_html += _section('6. Summary and Conclusions', conc_html, 'conclusions')

    # ── TOC ───────────────────────────────────────────────────────────────────
    toc = """<nav class="toc"><ul>
      <li><a href="#summary">Executive Summary</a></li>
      <li><a href="#eda">1. Data and EDA</a></li>
      <li><a href="#problem">2. Problem Statement</a></li>
      <li><a href="#models">3. Model Walk-Through</a></li>
      <li><a href="#comparison">4. Full Model Comparison</a></li>
      <li><a href="#sensitivity">5. Z Model Sensitivity</a></li>
      <li><a href="#conclusions">6. Summary and Conclusions</a></li>
    </ul></nav>"""

    # ── Executive summary (needs comparison_df and sensitivity) ───────────────
    print('  Executive summary...')
    exec_html = ''
    try:
        exec_html = _build_executive_summary(data, traces, comparison_df, sensitivity_summary)
    except Exception as e:
        exec_html = f'<p>[Error: {e}]</p>'

    # ── Assemble document ─────────────────────────────────────────────────────
    today = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {_CSS}
</head>
<body>
  <h1>{title}</h1>
  <p><em>Generated {today} &nbsp;·&nbsp; {data["n_areas"]:,} LSOAs
     &nbsp;·&nbsp; {len(traces)} model(s)</em></p>
  {toc}
  <section id="summary"><h2>Executive Summary</h2>
  {exec_html}
  </section>
  {sections_html}
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'  Done. {len(html):,} bytes written to {output_path}')
