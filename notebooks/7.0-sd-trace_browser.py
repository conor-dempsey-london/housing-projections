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
    import housing_projections.models as models_pkg
    import housing_projections.outliers as outliers
    from housing_projections.config import DATA_PATH, INFER_YEARS, TRACES_DIR
    from housing_projections.diagnostics import diagnostics_summary

    return (
        DATA_PATH,
        INFER_YEARS,
        TRACES_DIR,
        az,
        data_utils,
        diagnostics_summary,
        mo,
        models_pkg,
        np,
        outliers,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Trace browser

    Pick one model to see its example posterior traces. A separate summary
    table below covers every model with a saved trace in `results/traces/`,
    and only recomputes rows for traces it hasn't seen before.
    """)
    return


@app.cell(hide_code=True)
def _(DATA_PATH, data_utils, outliers):
    gdf_raw = data_utils.load_data(DATA_PATH)
    gdf_clean, _excluded = outliers.apply_outlier_exclusion(gdf_raw, verbose=False)
    return (gdf_clean,)


@app.cell(hide_code=True)
def _(data_utils):
    def data_matching_trace(gdf, trace):
        """Data dict whose rows match the LSOAs a trace was sampled on."""
        z_posterior = trace.posterior["z"]
        if "area" in z_posterior.coords:
            lsoa_codes = z_posterior.coords["area"].values.tolist()
            subset = gdf[gdf["LSOA21CD"].isin(lsoa_codes)].copy()
            subset = subset.set_index("LSOA21CD").loc[lsoa_codes].reset_index()
            return data_utils.make_data_dict(subset)
        return data_utils.make_data_dict(gdf, n_areas=z_posterior.shape[2])

    return (data_matching_trace,)


@app.cell(hide_code=True)
def _(TRACES_DIR, mo):
    # Re-runs (and the dropdown below along with it) whenever a trace file
    # is added to / removed from the traces directory.
    traces_dir_watch = mo.watch.directory(TRACES_DIR)
    available_models = sorted(p.stem for p in traces_dir_watch.glob("*.nc"))
    return (available_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Example traces
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Remembers the selected model across unrelated reruns (e.g. a new trace
    # appearing elsewhere) instead of resetting to the first option.
    get_selected_model, set_selected_model = mo.state(None)
    return get_selected_model, set_selected_model


@app.cell(hide_code=True)
def _(available_models, get_selected_model, mo, set_selected_model):
    mo.stop(not available_models,
            mo.md("*No saved traces found in `results/traces/`.*"))

    selected_current = get_selected_model()
    selected_default = (selected_current if selected_current in available_models
                        else available_models[0])

    model_dropdown = mo.ui.dropdown(
        options=available_models,
        value=selected_default,
        on_change=set_selected_model,
        label="Model",
    )
    model_dropdown
    return (model_dropdown,)


@app.cell(hide_code=True)
def _(
    TRACES_DIR,
    az,
    data_matching_trace,
    gdf_clean,
    model_dropdown,
    models_pkg,
):
    # Loaded once per model selection and shared by the z-timeseries panel
    # and the scalar-parameter trace plot below — avoids reading the same
    # (sometimes 1GB+) trace file twice.
    selected_trace = az.from_netcdf(str(TRACES_DIR / f"{model_dropdown.value}.nc"))
    selected_data  = data_matching_trace(gdf_clean, selected_trace)
    selected_model = getattr(models_pkg, model_dropdown.value)(selected_data)
    return selected_data, selected_model, selected_trace


@app.cell(hide_code=True)
def _(mo, model_dropdown, selected_model):
    mo.md(f"""
    **{model_dropdown.value}**: {selected_model.description}
    """)
    return


@app.cell(hide_code=True)
def _(INFER_YEARS, mo, model_dropdown, np, plt, selected_data, selected_trace):
    def select_diverse_areas(P_obs, n):
        """n area indices spanning low->high P_obs temporal variability (CV)."""
        means = np.abs(P_obs).mean(axis=1)
        stds  = P_obs.std(axis=1)
        cv    = np.where(means > 0.5, stds / means, 0.0)
        thresholds = np.percentile(cv, np.linspace(0, 100, n + 2)[1:-1])
        return [int(np.argmin(np.abs(cv - t))) for t in thresholds]

    N_SAMPLE_AREAS = 9

    z_post      = selected_trace.posterior["z"]
    n_areas     = z_post.sizes["area"]
    lsoa_codes  = (z_post.coords["area"].values.tolist()
                  if "area" in z_post.coords else None)

    area_idx = select_diverse_areas(selected_data["P_obs"], min(N_SAMPLE_AREAS, n_areas))

    # az.from_netcdf is lazy (a MemoryCachedArray, not a materialised numpy
    # array) until something touches .values — slicing to just the areas
    # we're about to plot BEFORE that happens means only those areas are
    # ever read off disk, regardless of how large the full trace is.
    z_sub  = z_post.isel(area=area_idx).values  # (chain, draw, len(area_idx), year)
    n_chains, n_draws, _, n_years = z_sub.shape
    z_flat = z_sub.reshape(n_chains * n_draws, len(area_idx), n_years)
    z_mean = z_flat.mean(axis=0)
    z_lo   = np.percentile(z_flat,  5, axis=0)
    z_hi   = np.percentile(z_flat, 95, axis=0)

    years = np.array(INFER_YEARS[:n_years])

    ncols = 3
    nrows = int(np.ceil(len(area_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).ravel()

    for plot_i, (ax, area_i) in enumerate(zip(axes, area_idx)):
        ax.fill_between(years, z_lo[plot_i], z_hi[plot_i],
                        alpha=0.25, color="steelblue", label="z 90% CI")
        ax.plot(years, z_mean[plot_i], color="steelblue", linewidth=1.5, label="z mean")
        ax.plot(years, selected_data["P_obs"][area_i], "x", color="darkorange",
                markersize=5, label="P_obs")
        ax.plot(years, selected_data["E_obs"][area_i], "o", color="forestgreen",
                markersize=4, fillstyle="none", label="E_obs")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        code = lsoa_codes[area_i] if lsoa_codes is not None else area_i
        ax.set_title(f"{str(code)[:12]}  D={selected_data['D'][area_i]:.0f}  "
                     f"Σz={z_mean[plot_i].sum():.0f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(area_idx):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(f"{model_dropdown.value} — example z timeseries vs observations", fontsize=11)
    plt.tight_layout()

    mo.vstack([plt.gcf()])
    return


@app.cell(hide_code=True)
def _(az, mo, model_dropdown, plt, selected_model, selected_trace):
    trace_var_names = [v for v in selected_model.var_names
                       if v in selected_trace.posterior]

    az.plot_trace(selected_trace, var_names=trace_var_names,
                 figure_kwargs={"figsize": (12, 2 * len(trace_var_names))})
    plt.suptitle(f"{model_dropdown.value} — scalar parameter traces")
    plt.tight_layout()

    mo.vstack([plt.gcf()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Diagnostics & LOO summary — all saved traces
    """)
    return


@app.cell(hide_code=True)
def _():
    # Plain dict, mutated in place, keyed by model name -> {trace, diag_row,
    # elpd}. Never reassigned, so it survives reruns of the summary cell
    # below and only grows/shrinks for traces that actually appeared or
    # disappeared on disk.
    summary_cache = {}
    return (summary_cache,)


@app.cell(hide_code=True)
def _(
    TRACES_DIR,
    available_models,
    az,
    data_matching_trace,
    diagnostics_summary,
    gdf_clean,
    mo,
    models_pkg,
    pd,
    summary_cache,
):
    mo.stop(not available_models, mo.md("*No saved traces to summarise yet.*"))

    def refresh_summary(names, cache):
        # All saved traces are assumed to share the same area sample (the
        # convention this whole repo already relies on — see
        # diagnostics_summary's docstring and cli.py's _data_matching_traces)
        # so one `data` dict, built from whichever model happens to be
        # first, is reused for every model's diagnostics row below.
        new_names = [n for n in names if n not in cache]
        data = None
        if new_names:
            first_trace = az.from_netcdf(str(TRACES_DIR / f"{sorted(names)[0]}.nc"))
            data = data_matching_trace(gdf_clean, first_trace)

        recomputed, failed = [], {}
        for name in new_names:
            try:
                trace = az.from_netcdf(str(TRACES_DIR / f"{name}.nc"))

                # Restrict r-hat/ESS to this model's own scalar var_names
                # instead of every posterior variable (z, delta, resp_*,
                # *_pointwise, ...) — unrestricted, this is ~98% of
                # diagnostics_summary's runtime (see its docstring).
                model_var_names = None
                ModelClass = getattr(models_pkg, name, None)
                if ModelClass is not None:
                    vn = ModelClass.var_names
                    if isinstance(vn, property):  # e.g. M5 — needs an instance
                        vn = ModelClass(data).var_names if data is not None else None
                    model_var_names = {name: vn} if vn is not None else None

                diag_row = diagnostics_summary({name: trace}, data=data,
                                               rhat_threshold=1.01,
                                               var_names=model_var_names).iloc[0]
                elpd = az.loo(trace, var_name="P_like")
                cache[name] = {"diag_row": diag_row, "elpd": elpd}
                recomputed.append(name)
            except Exception as exc:  # noqa: BLE001 — surface, don't crash the cell
                failed[name] = str(exc)

        for name in list(cache):
            if name not in names:
                del cache[name]

        return recomputed, failed

    recomputed_summary, failed_summary = refresh_summary(available_models, summary_cache)

    diag_df = pd.DataFrame({
        name: summary_cache[name]["diag_row"] for name in available_models
        if name in summary_cache
    }).T

    elpd_by_model = {
        name: summary_cache[name]["elpd"] for name in available_models
        if name in summary_cache
    }
    if len(elpd_by_model) >= 2:
        comparison = az.compare(elpd_by_model)
        summary_df = diag_df.join(
            comparison[["elpd", "se", "p", "elpd_diff", "weight"]])
    else:
        summary_df = diag_df

    mo.vstack([
        mo.md(f"Recomputed: {', '.join(recomputed_summary) or '*(none — all cached)*'}"
              + (f"  \nFailed: {', '.join(failed_summary)}" if failed_summary else "")),
        summary_df,
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
