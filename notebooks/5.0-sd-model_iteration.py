import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import housing_projections.data as data_utils
    import housing_projections.outliers as outliers
    from housing_projections.config import DATA_PATH, INFER_YEARS, TRACES_DIR
    from housing_projections.diagnostics import diagnostics_summary, full_diagnostics

    return (
        DATA_PATH,
        INFER_YEARS,
        TRACES_DIR,
        az,
        data_utils,
        diagnostics_summary,
        full_diagnostics,
        mo,
        np,
        outliers,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Model iteration

    Watches `results/traces/` for changes. Pick a set of models below — traces
    are (re)loaded whenever the selection changes or the underlying `.nc` file
    on disk is updated (e.g. after a resample), and diagnostics re-run
    automatically whenever traces are (re)loaded.
    """)
    return


@app.cell(hide_code=True)
def _(DATA_PATH, data_utils, outliers):
    gdf_raw = data_utils.load_data(DATA_PATH)
    gdf_clean, _excluded = outliers.apply_outlier_exclusion(gdf_raw)
    return (gdf_clean,)


@app.cell(hide_code=True)
def _(TRACES_DIR, mo):
    # Re-runs when files are added to / removed from the traces directory.
    traces_dir_watch = mo.watch.directory(TRACES_DIR)
    available_models = sorted(p.stem for p in traces_dir_watch.glob("*.nc"))
    return (available_models,)


@app.cell(hide_code=True)
def _(mo):
    # Explicit state (rather than relying on mo.ui.multiselect's own
    # frontend-state preservation across reactive reruns) so a
    # newly-appeared trace file can be deliberately auto-added to the
    # comparison set without depending on marimo's implicit "don't
    # clobber user interaction" behaviour, which is the right default in
    # general but would otherwise leave new models unselected until
    # manually clicked. get_seen_models tracks every model name ever
    # observed in available_models, so "new" can be distinguished from
    # "the user deliberately deselected this one".
    get_selected_models, set_selected_models = mo.state(None)
    get_seen_models, set_seen_models = mo.state(set())
    return (
        get_seen_models,
        get_selected_models,
        set_seen_models,
        set_selected_models,
    )


@app.cell(hide_code=True)
def _(
    available_models,
    get_seen_models,
    get_selected_models,
    set_seen_models,
    set_selected_models,
):
    # Reconciliation: models newly present in available_models (a trace
    # file just appeared) are auto-added to the selection; models no
    # longer present (a trace file was deleted) are dropped. Models the
    # user has manually deselected, but that still exist, stay deselected.
    reconcile_seen     = get_seen_models()
    reconcile_selected = get_selected_models()

    reconcile_new_models     = [m for m in available_models if m not in reconcile_seen]
    reconcile_removed_models = reconcile_seen - set(available_models)

    if reconcile_selected is None:
        reconcile_result = list(available_models)  # first run: select everything
    else:
        reconcile_result = (
            [m for m in reconcile_selected if m not in reconcile_removed_models]
            + reconcile_new_models
        )

    if reconcile_selected is None or reconcile_new_models or reconcile_removed_models:
        set_selected_models(reconcile_result)
    set_seen_models(set(available_models))
    return


@app.cell(hide_code=True)
def _(available_models, get_selected_models, mo, set_selected_models):
    model_selector = mo.ui.multiselect(
        options=available_models,
        value=get_selected_models() or [],
        on_change=set_selected_models,
        label="Models to compare",
    )
    model_selector
    return (model_selector,)


@app.cell(hide_code=True)
def _():
    # Persistent, mutated-in-place cache of loaded traces, keyed by model
    # name -> {mtime, trace}. mo.watch.file below makes this cell re-run
    # in full whenever ANY watched trace file changes (marimo re-executes
    # whole cells, not sub-expressions) — without this cache that would
    # mean re-reading every currently selected trace's netCDF (each
    # 300-400MB) whenever just one of them changes or a new one is added.
    # This cache lets each model's expensive az.from_netcdf be skipped
    # unless that specific model's file mtime has actually changed.
    trace_cache = {}
    return (trace_cache,)


@app.cell(hide_code=True)
def _(TRACES_DIR, az, mo, model_selector, trace_cache):
    def load_selected_traces(names, traces_dir, cache):
        loaded, paths = {}, {}
        for name in names:
            path = traces_dir / f"{name}.nc"
            if not path.exists():
                continue
            # Re-runs this cell when the individual trace file's content
            # changes on disk (e.g. a model is resampled and re-saved).
            watched_path = mo.watch.file(path)
            mtime = watched_path.stat().st_mtime
            cached = cache.get(name)
            if cached is None or cached["mtime"] != mtime:
                cache[name] = {"mtime": mtime, "trace": az.from_netcdf(str(watched_path))}
            loaded[name] = cache[name]["trace"]
            paths[name] = path

        for name in list(cache):
            if name not in names:
                del cache[name]

        return loaded, paths

    traces, trace_paths = load_selected_traces(model_selector.value, TRACES_DIR, trace_cache)

    mo.md(f"Loaded traces: {', '.join(traces) or '*(none selected)*'}")
    return trace_paths, traces


@app.cell(hide_code=True)
def _(data_utils, gdf_clean, traces):
    def data_matching_traces(gdf, traces):
        first_trace = next(iter(traces.values()))
        z_posterior = first_trace.posterior["z"]
        if "area" in z_posterior.coords:
            lsoa_codes = z_posterior.coords["area"].values.tolist()
            subset = gdf[gdf["LSOA21CD"].isin(lsoa_codes)].copy()
            subset = subset.set_index("LSOA21CD").loc[lsoa_codes].reset_index()
            return data_utils.make_data_dict(subset)
        return data_utils.make_data_dict(gdf, n_areas=z_posterior.shape[2])

    data = data_matching_traces(gdf_clean, traces) if traces else None
    return (data,)


@app.cell
def _(mo):
    mo.md("""
    ## Diagnostics
    """)
    return


@app.cell(hide_code=True)
def _():
    # A plain dict, mutated (never reassigned) by the diagnostics cell below.
    # Runs once at notebook start — marimo doesn't track mutations, so this
    # persists across reruns of the diagnostics cell instead of resetting.
    diagnostics_cache = {}
    return (diagnostics_cache,)


@app.cell(hide_code=True)
def _(
    data,
    diagnostics_cache,
    diagnostics_summary,
    mo,
    pd,
    trace_paths,
    traces,
):
    # Depends on `traces`, so it re-runs every time the load-traces cell above
    # re-runs (new selection or an updated file on disk) — but only actually
    # recomputes diagnostics for models whose trace file mtime (or the
    # underlying `data` sample) has changed since the last run; everything
    # else is served from `diagnostics_cache`.
    mo.stop(not traces, mo.md("*Select at least one model to see diagnostics.*"))

    def refresh_diagnostics(traces, trace_paths, data, cache):
        # Cheap fingerprint so a switch to a differently-sampled area set
        # invalidates stale cached rows even if mtimes didn't change.
        data_fingerprint = tuple(data["gdf"]["LSOA21CD"]) if data is not None else None

        recomputed = []
        for name, trace in traces.items():
            mtime = trace_paths[name].stat().st_mtime
            cached = cache.get(name)
            if (cached is not None
                    and cached["mtime"] == mtime
                    and cached["data_fingerprint"] == data_fingerprint):
                continue
            row = diagnostics_summary({name: trace}, data=data, rhat_threshold=1.01).iloc[0]
            cache[name] = {"mtime": mtime, "data_fingerprint": data_fingerprint, "row": row}
            recomputed.append(name)

        for name in list(cache):
            if name not in traces:
                del cache[name]

        return recomputed

    recomputed = refresh_diagnostics(traces, trace_paths, data, diagnostics_cache)

    diag = pd.DataFrame({name: diagnostics_cache[name]["row"] for name in traces}).T

    mo.vstack([
        mo.md(f"Recomputed diagnostics for: {', '.join(recomputed) or '*(none — all cached)*'}"),
        diag,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Structural diagnostics

    Checks for models that replace generic observation noise with a
    structural mechanism — shown per selected model where applicable,
    silently skipped for models without the relevant parameters (e.g.
    M0-M8 have neither).

    - **sigma_slab vs disagreement** (M9, M10): does per-area `sigma_slab`
      track genuine, lag-corrected P/E disagreement — computed by shifting
      each source's full observation series by its own learned mean lag —
      rather than something spurious like raw area scale? Expect both
      correlations clearly negative.
    - **kappa vs recording rate** (M10): does per-area capture-rate
      `kappa` track the empirical log(P/E) recording-rate bias, rather
      than absorbing generic residual noise? Expect a strongly positive
      correlation.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Same mtime-based caching pattern as diagnostics_cache/loo_cache —
    # full_diagnostics is the heavier call (includes calibration, census,
    # Moran's I), so avoid recomputing it here on every reactive rerun.
    structural_cache = {}
    return (structural_cache,)


@app.cell(hide_code=True)
def _(data, full_diagnostics, mo, structural_cache, trace_paths, traces):
    mo.stop(not traces, mo.md("*Select at least one model to see structural diagnostics.*"))

    def refresh_structural(traces, trace_paths, data, cache):
        data_fingerprint = tuple(data["gdf"]["LSOA21CD"]) if data is not None else None
        recomputed = []
        for name, trace in traces.items():
            mtime = trace_paths[name].stat().st_mtime
            cached = cache.get(name)
            if (cached is not None
                    and cached["mtime"] == mtime
                    and cached["data_fingerprint"] == data_fingerprint):
                continue
            result = full_diagnostics(trace, data, verbose=False)
            cache[name] = {
                "mtime": mtime,
                "data_fingerprint": data_fingerprint,
                "sigma_slab": result.get("sigma_slab_vs_disagreement"),
                "kappa":      result.get("kappa_vs_recording_rate"),
            }
            recomputed.append(name)

        for name in list(cache):
            if name not in traces:
                del cache[name]

        return recomputed

    def format_sigma_slab_check(check):
        return mo.vstack([
            mo.hstack([
                mo.stat(value=f"{check['mean_lag_P']:.2f} yr", label="Mean lag — P"),
                mo.stat(value=f"{check['mean_lag_E']:.2f} yr", label="Mean lag — E"),
                mo.stat(value=f"{check['corr_agreement_vs_sigma_slab']:.3f}",
                        label="corr(agreement, sigma_slab)"),
                mo.stat(value=f"{check['corr_agreement_vs_sigma_slab_scale_controlled']:.3f}",
                        label="...scale-controlled"),
            ]),
            mo.md(f"Based on {check['n_areas_valid']} areas with valid "
                  "agreement statistics."),
        ])

    def format_kappa_check(check):
        return mo.vstack([
            mo.stat(value=f"{check['corr_kappa_ratio_vs_empirical']:.3f}",
                    label="corr(log(kappa_P/kappa_E), empirical log(P/E))"),
            mo.md(f"Based on {check['n_areas_valid']} areas with valid "
                  "empirical ratios."),
        ])

    recomputed_structural = refresh_structural(traces, trace_paths, data, structural_cache)

    structural_sections = []
    for name in traces:
        cached = structural_cache[name]
        blocks = []
        if cached["sigma_slab"] is not None:
            blocks.append(mo.vstack([
                mo.md(f"**{name} — sigma_slab vs disagreement**"),
                format_sigma_slab_check(cached["sigma_slab"]),
            ]))
        if cached["kappa"] is not None:
            blocks.append(mo.vstack([
                mo.md(f"**{name} — kappa vs recording rate**"),
                format_kappa_check(cached["kappa"]),
            ]))
        if blocks:
            structural_sections.append(mo.vstack(blocks))

    mo.vstack([
        mo.md(f"Recomputed structural diagnostics for: "
              f"{', '.join(recomputed_structural) or '*(none — all cached)*'}"),
        (mo.vstack(structural_sections) if structural_sections
         else mo.md("*No selected model exposes a structural diagnostic.*")),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Model comparison
    """)
    return


@app.cell(hide_code=True)
def _():
    # Per-model LOO (ELPDData), cached the same way as diagnostics_cache —
    # mutated in place so it survives reruns of the comparison cell below.
    loo_cache = {}
    return (loo_cache,)


@app.cell(hide_code=True)
def _(az, loo_cache, mo, trace_paths, traces):
    # Depends on `traces`, so it re-runs whenever the load-traces cell
    # re-runs — but only recomputes LOO (the expensive PSIS step) for models
    # whose trace file mtime has changed; az.compare() itself is cheap once
    # each model's ELPDData is in hand, so the ranking is always freshly
    # recombined from (mostly cached) per-model results.
    mo.stop(len(traces) < 2, mo.md("*Select at least 2 models to compare.*"))

    def refresh_loo(traces, trace_paths, cache):
        recomputed = []
        for name, trace in traces.items():
            mtime = trace_paths[name].stat().st_mtime
            cached = cache.get(name)
            if cached is not None and cached["mtime"] == mtime:
                continue
            elpd = az.loo(trace, var_name="P_like")
            cache[name] = {"mtime": mtime, "elpd": elpd}
            recomputed.append(name)

        for name in list(cache):
            if name not in traces:
                del cache[name]

        return recomputed

    recomputed_loo = refresh_loo(traces, trace_paths, loo_cache)

    comparison = az.compare(
        {name: loo_cache[name]["elpd"] for name in traces},
    )

    mo.vstack([
        mo.md(f"Recomputed LOO for: {', '.join(recomputed_loo) or '*(none — all cached)*'}"),
        comparison,
    ])
    return (comparison,)


@app.cell(hide_code=True)
def _(az, comparison, plt):
    az.plot_compare(comparison)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Visual inspection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_inspect_model, set_inspect_model = mo.state(None)
    return get_inspect_model, set_inspect_model


@app.cell
def _(get_inspect_model, mo, set_inspect_model, traces):
    mo.stop(not traces, mo.md("*Select at least one model to inspect.*"))

    # Keep the current inspection target if it's still among the loaded
    # traces; otherwise (first run, or that model's trace disappeared)
    # default to the first available — avoids jumping to a different
    # model just because an unrelated new trace was added elsewhere.
    inspect_current = get_inspect_model()
    inspect_default = (inspect_current if inspect_current in traces
                       else sorted(traces)[0])

    inspect_model_selector = mo.ui.dropdown(
        options=sorted(traces),
        value=inspect_default,
        on_change=set_inspect_model,
        label="Model to inspect",
    )
    inspect_model_selector
    return (inspect_model_selector,)


@app.cell(hide_code=True)
def _(INFER_YEARS, data, inspect_model_selector, mo, np, plt, traces):
    mo.stop(inspect_model_selector.value is None)

    N_SAMPLE_AREAS = 6

    def select_diverse_areas(P_obs, n):
        """
        Return n area indices spanning the range of P_obs temporal
        variability, stratified by CV(P_obs) so the sample includes quiet,
        steady, and bursty areas rather than n random ones.
        """
        means = np.abs(P_obs).mean(axis=1)
        stds = P_obs.std(axis=1)
        cv = np.where(means > 0.5, stds / means, 0.0)
        quantiles = np.linspace(0, 100, n + 2)[1:-1]
        thresholds = np.percentile(cv, quantiles)
        return [int(np.argmin(np.abs(cv - t))) for t in thresholds]

    trace = traces[inspect_model_selector.value]
    z_post = trace.posterior["z"].values           # (chain, draw, area, year)
    n_chains, n_draws, n_areas, n_years = z_post.shape
    z_flat = z_post.reshape(n_chains * n_draws, n_areas, n_years)

    z_mean = z_flat.mean(axis=0)
    z_lo = np.percentile(z_flat, 5, axis=0)
    z_hi = np.percentile(z_flat, 95, axis=0)

    lsoa_codes = (trace.posterior["z"].coords["area"].values.tolist()
                  if "area" in trace.posterior["z"].coords else list(range(n_areas)))

    area_idx = select_diverse_areas(data["P_obs"], N_SAMPLE_AREAS)
    years = np.array(INFER_YEARS)

    ncols = 3
    nrows = int(np.ceil(N_SAMPLE_AREAS / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharey=False)
    axes = np.array(axes).ravel()

    for plot_i, area_i in enumerate(area_idx):
        ax = axes[plot_i]
        code = lsoa_codes[area_i]

        ax.fill_between(years, z_lo[area_i], z_hi[area_i],
                         alpha=0.25, color="steelblue", label="z 90% CI")
        ax.plot(years, z_mean[area_i], color="steelblue", linewidth=1.5, label="z mean")
        ax.plot(years, data["P_obs"][area_i], "x", color="darkorange",
                markersize=5, label="P_obs")
        ax.plot(years, data["E_obs"][area_i], "o", color="forestgreen",
                markersize=4, fillstyle="none", label="E_obs")

        p_cv = (data["P_obs"][area_i].std() /
                max(abs(data["P_obs"][area_i].mean()), 0.5))
        ax.set_title(f"{str(code)[:12]}  CV={p_cv:.2f}", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(years[::2])
        ax.tick_params(labelsize=7)

    for ax in axes[N_SAMPLE_AREAS:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(
        f"{inspect_model_selector.value} — z posterior vs observations "
        f"(areas ordered low→high temporal variability)",
        fontsize=11,
    )
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
