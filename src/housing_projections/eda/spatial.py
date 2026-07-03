import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local

from housing_projections.config import INFER_COLS_BEN, INFER_COLS_PLAN, INFER_YEARS
from housing_projections.spatial import build_weights_libpysal, compute_morans_i

# ── Moran's I ─────────────────────────────────────────────────────────────────

def plot_morans_i_by_year(gdf):
    """
    Compute and plot Moran's I for planning and BEN separately for each year.
    """

    w = build_weights_libpysal(gdf)

    results_plan = []
    results_ben  = []

    for col_p, col_b, yr in zip(INFER_COLS_PLAN, INFER_COLS_BEN, INFER_YEARS):
        results_plan.append(compute_morans_i(gdf[col_p].values, w))
        results_ben.append(compute_morans_i(gdf[col_b].values,  w))

    df_plan = pd.DataFrame(results_plan, index=INFER_YEARS)
    df_ben  = pd.DataFrame(results_ben,  index=INFER_YEARS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, df, label, color in zip(
        axes,
        [df_plan, df_ben],
        ['Planning', 'BEN'],
        ['steelblue', 'coral']
    ):
        ax.plot(INFER_YEARS, df['I'], marker='o', color=color)
        ax.axhline(0, color='black', linewidth=0.5)

        # Mark significant years
        sig = df[df['p_value'] < 0.05]
        ax.scatter(sig.index, sig['I'], color='red', zorder=5, s=50,
                   label='p < 0.05')

        ax.set_xlabel('Year')
        ax.set_ylabel("Moran's I")
        ax.set_title(f"Moran's I by year: {label}")
        ax.set_xticks(INFER_YEARS)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    plt.suptitle("Moran's I — spatial autocorrelation by year")
    plt.tight_layout()
    plt.show()

    return df_plan, df_ben

def plot_spatial_distribution(gdf, col, title='', cmap='RdBu',
                               symmetric=True, quantile_clip=0.95):
    values    = gdf[col].values
    clip      = np.quantile(np.abs(values), quantile_clip)
    clipped   = np.clip(values, -clip, clip)

    vmin      = -clip if symmetric else 0
    vmax      = clip

    gdf_plot  = gdf.copy()
    gdf_plot[col] = clipped

    fig, ax   = plt.subplots(figsize=(10, 12))

    norm      = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj  = mcm.get_cmap(cmap)

    gdf_plot.plot(
        column=col,
        cmap=cmap,
        legend=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )

    sm   = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label(col)

    ax.set_axis_off()
    ax.set_title(title or col)
    plt.tight_layout()
    plt.show()


def plot_mean_change_maps(gdf):
    gdf_plot              = gdf.copy()
    gdf_plot['mean_plan'] = gdf[INFER_COLS_PLAN].mean(axis=1)
    gdf_plot['mean_ben']  = gdf[INFER_COLS_BEN].mean(axis=1)

    all_vals = np.concatenate([
        gdf_plot['mean_plan'].values,
        gdf_plot['mean_ben'].values
    ])
    clip     = np.quantile(np.abs(all_vals), 0.95)

    norm     = mcolors.Normalize(vmin=-clip, vmax=clip)
    cmap_obj = mcm.get_cmap('RdBu')

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, col, label in zip(
        axes,
        ['mean_plan', 'mean_ben'],
        ['Planning',  'BEN']
    ):
        gdf_plot.assign(**{col: np.clip(gdf_plot[col].values, -clip, clip)}).plot(
            column=col, cmap='RdBu', legend=False,
            ax=ax, vmin=-clip, vmax=clip
        )
        sm   = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.02)
        cbar.set_label('Mean annual change')
        ax.set_axis_off()
        ax.set_title(f'Mean annual change: {label}')

    plt.suptitle('Spatial distribution of mean annual dwelling change')
    plt.tight_layout()
    plt.show()


def plot_source_disagreement_map(gdf, quantile_clip=0.95):
    P = gdf[INFER_COLS_PLAN].values
    E = gdf[INFER_COLS_BEN].values

    gdf_plot                      = gdf.copy()
    gdf_plot['mean_abs_disagree'] = np.abs(P - E).mean(axis=1)

    values = gdf_plot['mean_abs_disagree'].values
    clip   = np.quantile(values, quantile_clip)

    norm     = mcolors.Normalize(vmin=0, vmax=clip)
    cmap_obj = mcm.get_cmap('YlOrRd')

    fig, ax = plt.subplots(figsize=(10, 12))

    gdf_plot.assign(
        mean_abs_disagree=np.clip(values, 0, clip)
    ).plot(
        column='mean_abs_disagree',
        cmap='YlOrRd',
        legend=False,
        ax=ax,
        vmin=0,
        vmax=clip,
    )

    sm   = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label('Mean absolute disagreement')

    ax.set_axis_off()
    ax.set_title('Mean absolute disagreement between planning and BEN')
    plt.tight_layout()
    plt.show()

def plot_census_stock_maps(gdf,
                            col_2011='dwellings_2011',
                            col_2021='dwellings_2021'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, col, label in zip(
        axes,
        [col_2011, col_2021],
        ['2011 dwelling stock', '2021 dwelling stock'],
    ):
        gdf.plot(
            column=col,
            cmap='YlOrRd',
            legend=True,
            ax=ax,
            legend_kwds={
                'shrink':      0.4,
                'aspect':      20,
                'pad':         0.02,
                'orientation': 'vertical',
            }
        )
        ax.set_axis_off()
        ax.set_title(label)

    plt.suptitle('Census dwelling stocks')
    plt.tight_layout()
    plt.show()


def plot_intercensal_change_histogram_map(gdf,
                                           col_2011='dwellings_2011',
                                           col_2021='dwellings_2021',
                                           quantile_clip=0.95):
    D    = gdf[col_2021].values - gdf[col_2011].values
    clip = np.quantile(np.abs(D), quantile_clip)
    D_clipped = np.clip(D, -clip, clip)

    gdf_plot                       = gdf.copy()
    gdf_plot['intercensal_change'] = D_clipped   # use clipped values

    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios  = [2, 1],
        height_ratios = [1, 1],
        hspace        = 0.4,
        wspace        = 0.3,
    )

    ax_map = fig.add_subplot(gs[:, 0])

    norm = mcolors.Normalize(vmin=-clip, vmax=clip)
    cmap = mcm.get_cmap('RdBu')

    gdf_plot.plot(
        column='intercensal_change',
        cmap='RdBu',
        legend=False,
        ax=ax_map,
        vmin=-clip,
        vmax= clip,
    )

    sm   = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label('Net dwelling change')

    ax_map.set_axis_off()
    ax_map.set_title('Intercensal change (2011-2021)')

    # Growth histogram — use unclipped D for accurate distributions
    ax_pos  = fig.add_subplot(gs[0, 1])
    growth  = D[D > 0]
    ax_pos.hist(growth, bins=50, color='steelblue', alpha=0.7,
                range=(0, clip))
    ax_pos.axvline(growth.mean(), color='red', linestyle='--',
                   linewidth=0.8, label=f'mean={growth.mean():.1f}')
    ax_pos.set_title(f'Growth areas (n={len(growth):,})')
    ax_pos.set_xlabel('Net gain in dwellings')
    ax_pos.set_ylabel('Count')
    ax_pos.spines[['top', 'right']].set_visible(False)
    ax_pos.legend(fontsize=8)

    # Decline histogram
    ax_neg  = fig.add_subplot(gs[1, 1])
    decline = D[D < 0]
    ax_neg.hist(decline, bins=50, color='coral', alpha=0.7,
                range=(-clip, 0))
    ax_neg.axvline(decline.mean(), color='red', linestyle='--',
                   linewidth=0.8, label=f'mean={decline.mean():.1f}')
    ax_neg.set_title(f'Decline areas (n={len(decline):,})')
    ax_neg.set_xlabel('Net loss in dwellings')
    ax_neg.set_ylabel('Count')
    ax_neg.spines[['top', 'right']].set_visible(False)
    ax_neg.legend(fontsize=8)

    plt.suptitle('Spatial distribution of intercensal dwelling change')
    plt.tight_layout()
    plt.show()

    print("\n── Intercensal change summary ────────────────────────────────")
    print(f"  Total net change:   {D.sum():,.0f}")
    print(f"  Growth areas:       {(D > 0).sum():,} "
          f"({(D > 0).mean()*100:.1f}%)")
    print(f"  Decline areas:      {(D < 0).sum():,} "
          f"({(D < 0).mean()*100:.1f}%)")
    print(f"  No change:          {(D == 0).sum():,} "
          f"({(D == 0).mean()*100:.1f}%)")
    print(f"  Mean change:        {D.mean():.2f}")
    print(f"  Median change:      {np.median(D):.2f}")
    print(f"  Max gain:           {D.max():.0f}")
    print(f"  Max loss:           {D.min():.0f}")


def plot_change_hotspots(gdf, col_2011='dwellings_2011',
                          col_2021='dwellings_2021', n_quantiles=5):
    """
    Map showing areas classified into quantiles of intercensal change,
    highlighting hotspots of growth and decline.
    """

    D = gdf[col_2021].values - gdf[col_2011].values

    gdf_plot = gdf.copy()
    gdf_plot['intercensal_change'] = D

    # Classify into quantile bins
    labels_neg = ['Large decline', 'Small decline']
    labels_pos = ['Small growth',  'Medium growth', 'Large growth']

    neg_mask = D < 0
    pos_mask = D > 0

    quantile_class = np.full(len(D), 'No change', dtype=object)

    if neg_mask.sum() > 0:
        neg_quantiles = pd.qcut(
            D[neg_mask], q=2, labels=labels_neg)
        quantile_class[neg_mask] = neg_quantiles.astype(str)

    if pos_mask.sum() > 0:
        pos_quantiles = pd.qcut(
            D[pos_mask], q=3, labels=labels_pos)
        quantile_class[pos_mask] = pos_quantiles.astype(str)

    gdf_plot['change_class'] = quantile_class

    colour_map = {
        'Large decline':  '#d7191c',
        'Small decline':  '#fdae61',
        'No change':      '#ffffbf',
        'Small growth':   '#a6d96a',
        'Medium growth':  '#1a9641',
        'Large growth':   '#006837',
    }

    gdf_plot['colour'] = gdf_plot['change_class'].map(colour_map)

    fig, ax = plt.subplots(figsize=(10, 12))

    for cls in ['Large decline', 'Small decline', 'No change',
                'Small growth', 'Medium growth', 'Large growth']:
        subset = gdf_plot[gdf_plot['change_class'] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colour_map[cls],
                        edgecolor='white', linewidth=0.1)

    patches = [
        mpatches.Patch(color=colour_map[cls], label=cls)
        for cls in colour_map
        if cls in gdf_plot['change_class'].values
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=8)
    ax.set_axis_off()
    ax.set_title('Intercensal change hotspots (2011-2021)')
    plt.tight_layout()
    plt.show()


def plot_spatial_autocorrelation_change(gdf,
                                         col_2011='dwellings_2011',
                                         col_2021='dwellings_2021',
                                         quantile_clip=0.95):
    D     = gdf[col_2021].values - gdf[col_2011].values
    w     = build_weights_libpysal(gdf)
    moran = Moran(D, w)

    print("\n── Moran's I: intercensal change ─────────────────────────────")
    print(f"  I={moran.I:.4f}  p={moran.p_sim:.4f}")

    lisa  = Moran_Local(D, w)
    sig   = lisa.p_sim < 0.05
    quads = lisa.q

    labels = np.full(len(D), 'Not significant', dtype=object)
    labels[(quads == 1) & sig] = 'High-High'
    labels[(quads == 2) & sig] = 'Low-High'
    labels[(quads == 3) & sig] = 'Low-Low'
    labels[(quads == 4) & sig] = 'High-Low'

    colour_map = {
        'High-High':       '#d7191c',
        'Low-High':        '#abd9e9',
        'Low-Low':         '#2c7bb6',
        'High-Low':        '#fdae61',
        'Not significant': '#d9d9d9',
    }

    gdf_plot           = gdf.copy()
    gdf_plot['lisa']   = labels
    gdf_plot['colour'] = gdf_plot['lisa'].map(colour_map)

    fig, ax = plt.subplots(figsize=(10, 12))

    for cls in colour_map:
        subset = gdf_plot[gdf_plot['lisa'] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colour_map[cls],
                        edgecolor='white', linewidth=0.1)

    patches = [
        mpatches.Patch(color=colour_map[cls],
                       label=f'{cls} (n={(gdf_plot["lisa"] == cls).sum()})')
        for cls in colour_map
        if (gdf_plot['lisa'] == cls).any()
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=8)
    ax.set_axis_off()
    ax.set_title(f"LISA clusters: intercensal change "
                 f"(Moran's I={moran.I:.3f}, p={moran.p_sim:.4f})")
    plt.tight_layout()
    plt.show()


def plot_intercensal_change_map(gdf,
                                 col_2011='dwellings_2011',
                                 col_2021='dwellings_2021',
                                 quantile_clip=0.95):
    """
    Map of exact intercensal dwelling change with symmetric colormap
    centred at zero.
    """
    gdf_plot = gdf.copy()
    gdf_plot['intercensal_change'] = (
        gdf[col_2021].values - gdf[col_2011].values
    )

    plot_spatial_distribution(
        gdf_plot,
        col='intercensal_change',
        title='Exact intercensal dwelling change (2011-2021)',
        cmap='RdBu',
        symmetric=True,
        quantile_clip=quantile_clip,
    )
