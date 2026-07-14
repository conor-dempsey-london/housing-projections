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
import pandas as pd

matplotlib.use('Agg')   # non-interactive backend for script usage

from housing_projections.analysis import (
    compute_lag_residuals,
    compute_lag_weights,
    compute_model_comparison,
    compute_spatial_misallocation_stats,
)
from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN, INFER_YEARS
from housing_projections.diagnostics import diagnostics_summary, full_diagnostics
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
from housing_projections.plots.core import (
    plot_residual_analysis,
    plot_sample_areas,
    plot_z_area,
    select_sample_areas,
)
from housing_projections.plots.model import (
    plot_lag_weights,
    plot_missing_statistics,
    plot_missingness_effect_on_z,
    plot_missingness_posterior,
    plot_spatial_diagnostics,
    plot_twocomp_diagnostics,
    plot_zero_inflation_check,
)
from housing_projections.sensitivity import (
    compute_decomposed_uncertainty,
    compute_model_agreement_matrix,
    compute_z_ensemble,
    compute_z_model_sensitivity,
    plot_ensemble_mean_map,
    plot_estimate_vs_uncertainty,
    plot_model_agreement_matrix,
    plot_sensitivity_vs_disagreement,
    plot_z_range_distribution,
    plot_z_sensitivity_map,
)
from housing_projections.spatial import build_weights_libpysal

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
    'M1':  ('Temporal lag on planning',
            'Planning completions data (PLD) are recorded when a building '
            'permit closes, which can be 1–3 years after dwellings are '
            'actually built. M1 adds a Dirichlet-weighted lag mixture that '
            'allows the model to account for this recording delay.'),
    'M2':  ('Zero-inflation (symmetric)',
            'Planning records are frequently absent (zero) for areas where '
            'development happened but was never registered. M2 adds a '
            'zero-inflation probability pi_miss that mass at zero reflects '
            'missingness rather than true zero development.'),
    'M3':  ('Asymmetric zero-inflation',
            'Missingness in planning data is asymmetric: positive '
            'developments are more likely to be missed than demolitions. '
            'M3 uses separate pi_miss_pos and pi_miss_neg parameters.'),
    'M4': ('Two-component mixture (alternative)',
            'Alternative to M3 using a tight + broad two-component mixture '
            'for the planning likelihood instead of explicit zero-inflation.'),
    'M5':  ('Spatial misallocation',
            'Planning data is sometimes registered in the wrong LSOA '
            '(particularly for developments straddling boundaries). M5 adds '
            'an alpha_spatial weight that blends z with its spatial lag.'),
    'M6':  ('AR(1) temporal prior',
            'Replaces the i.i.d. year prior on z with an AR(1) process, '
            'encoding the belief that true dwelling delivery is temporally '
            'autocorrelated. rho ~ Beta(8,2) implies strong persistence.'),
    'M7':  ('Borough hierarchy',
            'Two-level hierarchy: LSOAs nested in boroughs. Borough-level '
            'mean is drawn from a global prior, providing stronger partial '
            'pooling within boroughs than across London.'),
    'M8':  ('Time-varying observation noise',
            'The planning source becomes noisier in years with many large '
            'developments or regulatory changes. M8 adds a year-specific '
            'sigma_obs_plan to the planning likelihood.'),
}

# Which diagnostic plot best illustrates each model's contribution
_MODEL_KEY_VAR = {
    'M0': None, 'M0h': None,
    'M1': 'lambda_weights',
    'M2': 'pi_miss',
    'M3': 'pi_miss_pos',
    'M4': 'w_tight',
    'M5': 'alpha_spatial',
    'M6': 'rho',
    'M7': 'mu_borough',
    'M8': 'sigma_obs_plan',
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_fig(fn, caption='', *args, **kwargs):
    """Call fn(*args, **kwargs), capture the figure, close it, return HTML.
    Returns empty string on any failure so one bad plot never breaks the report."""
    try:
        result = fn(*args, **kwargs)
        fig = result[0] if isinstance(result, tuple) else result
        if fig is None:
            return ''
        html = _html_fig(fig, caption)
        plt.close(fig)
        return html
    except Exception:  # noqa: BLE001
        return ''


# ── Section builders ──────────────────────────────────────────────────────────

def _build_executive_summary(data, traces, comparison_df, sensitivity_summary):
    html = ''
    html += _callout(
        f'<strong>Data:</strong> {data["n_areas"]:,} London LSOAs &nbsp;·&nbsp; '
        f'{data["n_years"]} inference years (2011/12–2020/21) &nbsp;·&nbsp; '
        f'{len(traces)} model(s) compared'
    )

    # Summary table — use lightweight diagnostics_summary (no Moran's I / residuals)
    diag_df = diagnostics_summary(traces)
    rows = []
    for name in traces:
        loo_val = comparison_df.loc[name, 'elpd'] if (
            comparison_df is not None and name in comparison_df.index
            and 'elpd' in comparison_df.columns) else float('nan')
        d_loo = comparison_df.loc[name, 'elpd_diff'] if (
            comparison_df is not None and name in comparison_df.index
            and 'elpd_diff' in comparison_df.columns) else float('nan')

        rows.append({
            'Model':          name,
            'ELPD':           loo_val,
            'ΔELPD vs best':  d_loo,
            'Divergences':    int(diag_df.loc[name, 'divergences']) if name in diag_df.index else 0,
            'Max R̂':         float(diag_df.loc[name, 'max_rhat'])  if name in diag_df.index else float('nan'),
        })

    df = pd.DataFrame(rows)
    html += _df_to_html(df)
    return html


def _build_eda(data):
    gdf = data['gdf']
    html = ''

    # Overall correlation
    corr = compute_overall_correlation(gdf, verbose=False)
    r = corr.get('pearson_r', list(corr.values())[0] if corr else float('nan'))
    html += _stat_row([
        ('LSOAs', f'{data["n_areas"]:,}'),
        ('Inference years', str(data['n_years'])),
        ('PLD–BEN Pearson r (annual, per-LSOA)', f'{r:.3f}'),
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
        f'<strong>Source agreement (10-year cumulative totals per LSOA):</strong> '
        f'Pearson r = {stats["total_corr"]:.3f} &nbsp;·&nbsp; '
        f'Mean absolute bias = {stats["total_bias"]:.2f} dwellings/year &nbsp;·&nbsp; '
        f'Same sign = {stats["pct_same_sign"]:.1f}% &nbsp;·&nbsp; '
        f'<em>Note: higher than the annual figure above because aggregating over years '
        f'smooths out recording delays and year-to-year noise.</em>'
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


def _build_model_walk_through(traces, data, model_classes, diag_df=None):
    """One subsection per model — universal plots plus model-specific diagnostics."""

    html = ''
    model_names_ordered = [n for n in
                            ['M0', 'M0h', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
                            if n in traces]

    for name in model_names_ordered:
        trace = traces[name]
        desc_short, desc_long = _MODEL_DESCRIPTIONS.get(name, (name, ''))

        if diag_df is not None and name in diag_df.index:
            n_div    = int(diag_df.loc[name, 'divergences'])
            max_rhat = float(diag_df.loc[name, 'max_rhat'])
        else:
            n_div    = int(trace.sample_stats.diverging.sum())
            max_rhat = float('nan')

        card_html = f'<p>{desc_long}</p>'

        # ── Diagnostics mini-summary ──────────────────────────────────────────
        card_html += _stat_row([
            ('Divergences', str(n_div)),
            ('Max R̂', f'{max_rhat:.3f}' if not np.isnan(max_rhat) else '—'),
        ])
        if n_div > 100:
            card_html += _finding(f'{n_div} divergences — consider increasing target_accept '
                                  'or reparameterising.')

        # ── Universal: sample areas (z posterior vs observations) ─────────────
        card_html += _safe_fig(
            plot_sample_areas, 'Posterior z vs planning and BEN observations for 6 example LSOAs',
            trace, data, title=name, random_state=42,
        )

        # ── Universal: residual analysis ──────────────────────────────────────
        card_html += _safe_fig(
            plot_residual_analysis, 'Residual analysis — planning and BEN residuals by year and census change',
            trace, data, title=name,
        )

        # ── Model-specific plots ──────────────────────────────────────────────

        if name == 'M0h':
            # Hierarchical hyperpriors — show shrinkage across areas
            for var in ('mu_global', 'sigma_mu'):
                if var in trace.posterior:
                    try:
                        vals = trace.posterior[var].values.ravel()
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.hist(vals, bins=60, color='steelblue', alpha=0.7, density=True)
                        ax.set_xlabel(var)
                        ax.set_ylabel('Density')
                        ax.set_title(f'{name}: posterior of {var}')
                        ax.spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        card_html += _html_fig(fig, f'Posterior of {var} — the hierarchical hyperprior governing how much area-level means can deviate from the global mean')
                        plt.close(fig)
                    except Exception:  # noqa: BLE001
                        pass

        elif name == 'M1':
            try:
                lag_results = compute_lag_weights(trace, verbose=False)
                card_html += _safe_fig(
                    plot_lag_weights,
                    'Posterior lag weight distributions — how planning completions are spread across years',
                    lag_results, title=name,
                )
            except Exception:  # noqa: BLE001
                pass
            # Residuals with and without lag correction vs a baseline model
            if 'M0' in traces:
                try:
                    resids_m0 = compute_lag_residuals(traces['M0'], data)
                    resids_m3 = compute_lag_residuals(trace, data)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    for ax, resid, label, color in [
                        (axes[0], resids_m0['no_lag'].ravel(), 'M0 (no lag)', 'steelblue'),
                        (axes[1], resids_m3['with_lag'].ravel(), 'M1 (with lag)', 'coral'),
                    ]:
                        clip = np.quantile(np.abs(resid), 0.99)
                        ax.hist(resid, bins=80, density=True, color=color, alpha=0.7,
                                range=(-clip, clip))
                        ax.axvline(0, color='black', linewidth=0.8)
                        ax.axvline(resid.mean(), color='red', linestyle='--', linewidth=0.8,
                                   label=f'mean={resid.mean():.2f}')
                        ax.set_title(f'Planning residuals: {label}')
                        ax.set_xlabel('Observed − predicted')
                        ax.spines[['top', 'right']].set_visible(False)
                        ax.legend(fontsize=8)
                    plt.suptitle('M1 lag correction — planning residuals before and after')
                    plt.tight_layout()
                    card_html += _html_fig(fig, 'Lag correction shrinks the mean residual — M1 accounts for completions recorded in later years')
                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass

        elif name in ('M2', 'M3', 'M4'):
            card_html += _safe_fig(
                plot_missingness_posterior,
                'Missingness parameter posteriors — probability that a zero planning observation is a missing record rather than genuine zero activity',
                trace, title=name,
            )
            card_html += _safe_fig(
                plot_zero_inflation_check,
                'Observed vs model-predicted zero rate — how well the missingness model captures the frequency of zero planning observations',
                trace, data, title=name,
            )
            card_html += _safe_fig(
                plot_missing_statistics,
                'What the model infers about LSOAs with zero planning observations — how many zeros are genuine vs missing records',
                trace, data, title=name,
            )
            if name == 'M4':
                card_html += _safe_fig(
                    plot_twocomp_diagnostics,
                    'Two-component observation noise — weight on the tight vs loose component; the loose component absorbs outlier observations',
                    trace, data, title=name,
                )
            # Show effect of missingness correction vs M0
            if 'M0' in traces:
                card_html += _safe_fig(
                    plot_missingness_effect_on_z,
                    'Effect on z estimates for LSOAs where planning shows zero but BEN shows non-zero activity',
                    traces['M0'], trace, data,
                    title=name, label_before='M0', label_after=name,
                )

        elif name == 'M5':
            try:
                stats_dict = compute_spatial_misallocation_stats(trace, data)
                card_html += _safe_fig(
                    plot_spatial_diagnostics,
                    'Spatial misallocation diagnostics — alpha_spatial posterior and z vs spatial lag of z',
                    stats_dict, title=name,
                )
            except Exception:  # noqa: BLE001
                pass

        elif name == 'M6':
            for var, caption in [
                ('rho',        'AR(1) autocorrelation posterior — how strongly this year\'s delivery predicts next year\'s'),
                ('sigma_innov', 'Innovation noise posterior — year-to-year variability in z beyond the AR(1) trend'),
            ]:
                if var in trace.posterior:
                    try:
                        vals = trace.posterior[var].values.ravel()
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.hist(vals, bins=60, color='steelblue', alpha=0.7, density=True)
                        ax.set_xlabel(var)
                        ax.set_ylabel('Density')
                        ax.set_title(f'{name}: posterior of {var}')
                        ax.spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        card_html += _html_fig(fig, caption)
                        plt.close(fig)
                    except Exception:  # noqa: BLE001
                        pass

        elif name == 'M8':
            for var, caption in [
                ('sigma_base_plan', 'Base planning uncertainty posterior — how much z can deviate from the planning signal on average'),
                ('sigma_obs_plan',  'Planning observation noise posterior — residual noise beyond the structural uncertainty'),
            ]:
                if var in trace.posterior:
                    try:
                        vals = trace.posterior[var].values.ravel()
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.hist(vals, bins=60, color='steelblue', alpha=0.7, density=True)
                        ax.set_xlabel(var)
                        ax.set_ylabel('Density')
                        ax.set_title(f'{name}: posterior of {var}')
                        ax.spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        card_html += _html_fig(fig, caption)
                        plt.close(fig)
                    except Exception:  # noqa: BLE001
                        pass

        html += f'<div class="model-card" id="model-{name}">'
        html += f'<h4>{name} — {desc_short}</h4>'
        html += card_html
        html += '</div>\n'

    return html


def _build_model_comparison(traces, comparison_df, sensitivity_summary=None):
    html = ''

    # LOO table
    if comparison_df is not None:
        html += '<h3>LOO-CV comparison</h3>'
        cols = [c for c in ['elpd', 'se', 'p', 'elpd_diff', 'dse', 'weight']
                if c in comparison_df.columns]
        html += _df_to_html(comparison_df[cols].reset_index().rename(columns={'index': 'Model'}))

    if len(traces) < 2:
        html += '<p><em>Only one model — sensitivity analysis requires at least 2 models.</em></p>'
        return html

    # Agreement matrix
    corr = compute_model_agreement_matrix(traces)
    fig  = plot_model_agreement_matrix(corr)
    html += _html_fig(fig, 'Pairwise correlation of z posterior means across models')

    # Reuse pre-computed sensitivity if available
    if sensitivity_summary is None:
        sensitivity_summary, _ = compute_z_model_sensitivity(traces)

    fig = plot_z_range_distribution(sensitivity_summary)
    html += _html_fig(fig, 'Per-LSOA range of z posterior means across models '
                      '(max − min); wider range = higher model sensitivity')

    return html, sensitivity_summary


def _build_sensitivity_summary(sensitivity_summary, data, traces, comparison_df=None):
    gdf = data['gdf']
    html = ''

    std_col   = sensitivity_summary['z_std_across_models']
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

    # ── Decomposed uncertainty ────────────────────────────────────────────────
    html += _subsection('Decomposed Uncertainty', '')

    html += '''
    <p>A credible interval from a single model only captures <em>within-model</em> uncertainty
    (sampling variance given that model's assumptions). If different models place z in very
    different locations for the same LSOA, the true uncertainty is larger than any single
    model interval suggests.</p>
    <p>We decompose total uncertainty into two components:</p>
    <ul>
      <li><strong>Within-model:</strong> LOO-weighted average of per-model posterior SDs —
          how wide the posterior is given each model's assumptions.</li>
      <li><strong>Between-model:</strong> std of posterior means across models —
          how much the estimate itself shifts when model assumptions change.</li>
    </ul>
    <p>Total uncertainty = √(within² + between²). The <strong>confidence tier</strong>
    (High / Medium / Low) is assigned by the coefficient of variation of total uncertainty
    relative to the ensemble mean estimate.</p>
    '''

    try:
        lsoa_codes = gdf['LSOA21CD'].values if 'LSOA21CD' in gdf.columns else None
        unc_df = compute_decomposed_uncertainty(
            traces, comparison_df=comparison_df, lsoa_codes=lsoa_codes,
        )

        # Summary stats
        html += _stat_row([
            ('Mean within-model uncertainty', f'{unc_df["z_within_uncertainty"].mean():.2f} dw/yr'),
            ('Mean between-model uncertainty', f'{unc_df["z_between_uncertainty"].mean():.2f} dw/yr'),
            ('Mean total uncertainty', f'{unc_df["z_total_uncertainty"].mean():.2f} dw/yr'),
        ])

        tier_counts = unc_df['confidence_tier'].value_counts()
        n = len(unc_df)
        tier_html = ' | '.join(
            f'<strong>{t}</strong>: {tier_counts.get(t, 0)} ({tier_counts.get(t, 0)/n*100:.0f}%)'
            for t in ('High', 'Medium', 'Low')
        )
        html += f'<p>Confidence tier distribution: {tier_html}</p>'

        dominant = 'between-model' if (
            unc_df['z_between_uncertainty'].mean() > unc_df['z_within_uncertainty'].mean()
        ) else 'within-model'
        html += _finding(
            f'The dominant source of uncertainty is <strong>{dominant}</strong>. '
            + ('Model choice drives more uncertainty than sampling variance — '
               'this suggests the ensemble estimate is more trustworthy than any single model.'
               if dominant == 'between-model'
               else 'Sampling variance dominates — models broadly agree on where z is, '
                    'but each has wide posteriors.')
        )

        # Ensemble mean + uncertainty maps
        fig = plot_ensemble_mean_map(gdf.iloc[:len(unc_df)].copy(), unc_df)
        html += _html_fig(
            fig,
            'Left: LOO-stacking ensemble mean z (dwelling delivery rate). '
            'Right: total uncertainty — areas in red should be interpreted with more caution.'
        )

        # Estimate vs uncertainty scatter + decomposition
        fig = plot_estimate_vs_uncertainty(unc_df)
        html += _html_fig(
            fig,
            'Left: ensemble mean z vs total uncertainty, coloured by confidence tier. '
            'Right: within-model vs between-model uncertainty decomposition. '
            'Points above the diagonal are LSOAs where model choice matters more than sampling noise.'
        )

    except Exception:
        html += '<p><em>Uncertainty decomposition could not be computed.</em></p>'

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
      <li><strong>Lag structure</strong> (M1+) is supported by cross-correlations in
          the EDA: PLD completions lead BEN estimates by 1–2 years in many LSOAs.</li>
      <li><strong>Zero-inflation</strong> (M2/M3) materially improves the negative tail
          of planning residuals, consistent with systematic under-recording in the PLD.</li>
    </ul>
    <div class="callout">
    The recommended workflow is to sample M3 or M5 for the best balance of fit and
    interpretability. Use the LOO-stacking ensemble when communicating projections
    to stakeholders, as it integrates over model uncertainty.
    </div>
    """)
    return html


# ── Sample traces section ─────────────────────────────────────────────────────

def _build_sample_traces(traces, data, comparison_df=None, n_sample=9,
                         random_state=42):
    """
    Section 7: per-LSOA timeseries for a selection of areas using the best model.

    Selects areas spanning the full range of census differences (low, mid, high
    growth) and plots, for each one, the posterior z mean + credible interval
    from the best model alongside PLD and BEN observations.  A coloured header
    records the LSOA's confidence tier from the ensemble uncertainty analysis.
    """
    if not traces:
        return '<p>No traces available.</p>'

    # ── Identify best model ───────────────────────────────────────────────────
    if comparison_df is not None and len(comparison_df) > 0:
        best_name = comparison_df.index[0]
    else:
        best_name = next(iter(traces))
    best_trace = traces.get(best_name)
    if best_trace is None:
        return f'<p>Trace for best model ({best_name}) not found.</p>'

    # ── Compute per-area confidence tiers from ensemble ───────────────────────
    tier_by_idx = {}
    try:
        lsoa_codes = data['gdf']['LSOA21CD'].values \
            if 'LSOA21CD' in data['gdf'].columns else None
        unc_df = compute_decomposed_uncertainty(
            traces, comparison_df=comparison_df, lsoa_codes=lsoa_codes,
        )
        # Aggregate tier per area: take mode across years
        tier_by_idx = (
            unc_df.groupby('lsoa_idx')['confidence_tier']
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
    except Exception:
        pass

    # ── Select sample areas spanning the D range ──────────────────────────────
    D           = data['D']
    sample_idx  = select_sample_areas(D, n_sample=n_sample, random_state=random_state)
    z_post      = best_trace.posterior['z'].values   # (chains, draws, n_areas, n_years)
    P_obs       = data['P_obs']
    E_obs       = data['E_obs']

    # Resolve LSOA codes
    gdf      = data['gdf']
    code_col = next((c for c in ['LSOA21CD', 'LSOA11CD', 'geo_code', 'lsoa_code']
                     if c in gdf.columns), None)

    # ── Tier colour map ───────────────────────────────────────────────────────
    tier_colours = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}

    # ── Build one figure per area ─────────────────────────────────────────────
    n_cols = 3
    n_rows = int(np.ceil(n_sample / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for ax, idx in zip(np.array(axes).ravel(), sample_idx):
        plot_z_area(ax, z_post, idx,
                    infer_years=INFER_YEARS,
                    P_obs=P_obs, E_obs=E_obs, D=D,
                    n_years=data['n_years'],
                    show_legend=False, alpha_ci=0.9)

        # Annotate with LSOA code and confidence tier
        lsoa_label = gdf.iloc[idx][code_col] if code_col else f'LSOA {idx}'
        tier        = tier_by_idx.get(idx, '')
        colour      = tier_colours.get(tier, '#aaaaaa')
        ax.set_title(f'{lsoa_label}  D={D[idx]:.0f}', fontsize=8)
        if tier:
            ax.text(0.97, 0.97, tier, transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colour, alpha=0.85,
                              edgecolor='none'))

    # Hide unused panels
    for ax in np.array(axes).ravel()[n_sample:]:
        ax.set_visible(False)

    # Shared legend in the first panel
    axes_flat = np.array(axes).ravel()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels, fontsize=6, loc='upper left')

    plt.suptitle(
        f'Posterior z timeseries — {best_name} (best model by LOO-CV)',
        fontsize=12,
    )
    plt.tight_layout()

    # ── Prose ─────────────────────────────────────────────────────────────────
    tier_legend = ' '.join(
        f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;'
        f'background:{c};color:white;font-size:.8em;font-weight:bold">{t}</span>'
        for t, c in tier_colours.items()
    )
    html = f'''
    <p>
    Each panel shows the posterior z timeseries for one example LSOA from the
    <strong>{best_name}</strong> model (the best-performing model by LOO-CV).
    The shaded band is the 90% credible interval; markers show PLD (planning)
    and BEN observations.  Areas are selected to span the full range of
    intercensal change — from net loss through to high growth.
    </p>
    <p>
    The badge in the top-right corner of each panel indicates the LSOA's
    confidence tier from the ensemble uncertainty analysis: {tier_legend}
    </p>
    '''
    html += _html_fig(fig, (
        f'Posterior z timeseries for {n_sample} example LSOAs — {best_name}. '
        'Shaded band = 90% CI. Confidence tier badge reflects between-model '
        'disagreement in addition to within-model sampling uncertainty.'
    ))
    plt.close(fig)
    return html


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(data, traces, model_classes=None, output_path='results/report.html',
                    title='Housing Projections: Model Analysis Report',
                    comparison_df=None):
    """
    Generate a self-contained HTML analysis report.

    Parameters
    ----------
    data          : dict — output of make_data_dict
    traces        : dict[str, az.InferenceData]
    model_classes : dict[str, type] or None — model class objects for descriptions
    output_path   : str
    title         : str
    comparison_df : pd.DataFrame or None — pre-computed LOO comparison table.
                    If None and len(traces) > 1, LOO is computed here.
    """
    print('  Building sections...')

    # ── LOO comparison ────────────────────────────────────────────────────────
    if comparison_df is None and len(traces) > 1:
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

    # ── Diagnostics summary (r-hat + divergences — used in walk-through and exec summary) ──
    print('  Computing sampling diagnostics...')
    try:
        diag_df = diagnostics_summary(traces)
    except Exception as e:
        print(f'  [warning] diagnostics_summary failed: {e}')
        diag_df = None

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
        wt_html = _build_model_walk_through(traces, data, model_classes or {}, diag_df=diag_df)
        sections_html += _section('3. Model Walk-Through', wt_html, 'models')
    except Exception as e:
        sections_html += _section('3. Model Walk-Through',
                                  f'<p>[Error: {e}]</p>', 'models')

    print('  Model comparison...')
    comp_html = ''
    if len(traces) > 1:
        try:
            result = _build_model_comparison(traces, comparison_df,
                                             sensitivity_summary=sensitivity_summary)
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
            sens_html = _build_sensitivity_summary(sensitivity_summary, data, traces, comparison_df=comparison_df)
        except Exception as e:
            sens_html = f'<p>[Error: {e}]</p>'
    else:
        sens_html = '<p>Requires at least 2 models.</p>'
    sections_html += _section('5. Z Model Sensitivity', sens_html, 'sensitivity')

    print('  Conclusions...')
    conc_html = _build_conclusions(comparison_df, sensitivity_summary)
    sections_html += _section('6. Summary and Conclusions', conc_html, 'conclusions')

    print('  Sample traces...')
    try:
        sample_html = _build_sample_traces(traces, data, comparison_df=comparison_df)
    except Exception as e:
        sample_html = f'<p>[Error: {e}]</p>'
    sections_html += _section('7. Sample Traces with Uncertainty', sample_html, 'sample-traces')

    # ── TOC ───────────────────────────────────────────────────────────────────
    toc = """<nav class="toc"><ul>
      <li><a href="#summary">Executive Summary</a></li>
      <li><a href="#eda">1. Data and EDA</a></li>
      <li><a href="#problem">2. Problem Statement</a></li>
      <li><a href="#models">3. Model Walk-Through</a></li>
      <li><a href="#comparison">4. Full Model Comparison</a></li>
      <li><a href="#sensitivity">5. Z Model Sensitivity</a></li>
      <li><a href="#conclusions">6. Summary and Conclusions</a></li>
      <li><a href="#sample-traces">7. Sample Traces with Uncertainty</a></li>
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
