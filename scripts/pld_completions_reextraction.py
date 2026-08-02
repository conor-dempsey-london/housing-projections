"""
Reproducible unit-level re-extraction of PLD completions (and starts, for cross-checking)
from the live Planning London Datahub, replacing the undocumented scheme-level cut in
``<DATA_PATH>/pld/lsoa_completions_time_series_pivot.csv``.

Background
----------
The existing cut was produced via ``pld_database_live``'s ``level='scheme'`` extraction
path (the CLI default). At scheme level, applications are filtered by their own
``actual_completion_date`` and ``status`` fields, but every ``residential_units`` child
record attached to a matching application is then summed in *regardless of that unit's
own completion date*. For a multi-phase scheme this dumps its entire lifetime unit count
into whichever single year the scheme's overall record happens to close in, silently
misattributing units that actually completed in other years. Traced on a concrete example
(``Islington-P2016_2994_S73``): 955 net units attributed entirely to FY2023/24 by the
scheme-level cut, when the per-unit completion dates show 322 in 2019, 116 in 2022, and
only 517 in 2023.

``pld_database_live``'s own ``level='unit'`` path avoids this — it queries each child
index directly on the unit's own date column — but has two rough edges worth knowing
before reusing this script:

- Its own ``residential_table_gen``/``spatial_loc_join`` step depends on GIS layer parquet
  files (``data/location/*.parquet``) that ship separately from the repo and were not
  present on this machine, so this script bypasses the tool's spatial join entirely and
  does its own LSOA assignment from the centroid coordinates the ``applications`` index
  already returns (see ``assign_lsoa``).
- Its per-unit FY-of-completion fix (``bep.py:_completion_date_protocol``) only runs for
  ``stage='completions'``, not ``stage='starts'`` — for starts this script computes the
  per-unit financial year itself directly from ``unit_commencement_date``.

See ``docs/pld-completions-reextraction-findings.md`` for the full writeup, including the
starts-vs-completions cross-check and the missing-centroid investigation.

Usage
-----
    pixi run python scripts/pld_completions_reextraction.py
"""

import time

import geopandas as gpd
import gla_data
import pandas as pd
from pld_database.bep import (
    gen_unit_level_table,
    get_bespoke_extraction,
    get_bespoke_transformation,
)
from shapely.geometry import Point

FULL_START_YEAR = 2009
FULL_END_YEAR = 2026

# A fixed placeholder value PLD substitutes for centroid.lat/centroid.lon on records that
# have no real geocoding at all (confirmed identical across many unrelated records/boroughs;
# not a real location — never treat this as a usable fallback coordinate).
_PLD_NULL_CENTROID = (49.766807, -7.55716)


def extract_unit_level(stage, start_year=FULL_START_YEAR, end_year=FULL_END_YEAR,
                        lpa='all', extract_type='residential', verbose=True):
    """
    Run the BEP extract -> transform -> unit-level-aggregate pipeline directly (bypassing
    ``run_bespoke_pipeline``'s hardcoded ``polygon=True``, which raises on malformed
    ``MultiPolygon`` geometries present in the live data at full London/multi-year scale).

    Parameters
    ----------
    stage        : 'completions' or 'starts'
    start_year   : int — first financial year (Apr-Mar) to include
    end_year     : int — last financial year to include
    lpa          : str — 'all' or a specific LPA name
    extract_type : str — 'residential', 'non_residential', or 'all'
    verbose      : bool — print BEP progress panels

    Returns
    -------
    pd.DataFrame — one row per netted unit-record group, with per-application columns
    (borough, centroid_easting/northing, centroid.lat/lon, ...) merged in.
    """
    extract_dict = get_bespoke_extraction(
        stage=stage, start_year=start_year, end_year=end_year, year_type='financial_year',
        lpa=lpa, level='unit', extract_type=extract_type, verbose=verbose,
    )
    transformed_dict = get_bespoke_transformation(
        extract_dict=extract_dict, layer=['lsoa'], year_gen='financial_year',
        polygon=False, centroid=True, os_poly=False, os_add=False, sam=False, buildings=False,
        verbose=verbose,
    )
    return gen_unit_level_table(
        extract_dict=transformed_dict, stage=stage, start_year=start_year, verbose=verbose,
    )


def add_unit_fy_commencement(df):
    """
    Compute a per-unit financial-year-of-commencement column directly from
    ``unit_commencement_date``, matching ``pld_database.utils.gen_year_col``'s FY rule
    (Apr-Mar). Needed because the tool's own per-unit FY fix only runs for completions.

    Parameters
    ----------
    df : pd.DataFrame — unit-level table with a ``unit_commencement_date`` column

    Returns
    -------
    pd.DataFrame — copy of ``df`` with an added ``fy_of_commencement_unit`` column
    """
    df = df.copy()
    dt = pd.to_datetime(df['unit_commencement_date'], dayfirst=True, errors='coerce')
    fy_year = dt.dt.year - (dt.dt.month < 4).astype(int)
    df['fy_of_commencement_unit'] = (
        fy_year.astype('Int64').astype(str) + '/' + ((fy_year + 1) % 100).astype('Int64').astype(str).str.zfill(2)
    )
    df.loc[dt.isna(), 'fy_of_commencement_unit'] = None
    return df


def assign_lsoa(df, easting_col='centroid_easting', northing_col='centroid_northing'):
    """
    Spatially join each row to a 2021 LSOA using its BNG centroid, replacing
    ``pld_database_live``'s own spatial join (unavailable without its GIS layer files).

    Rows with no centroid, or whose ``centroid.lat``/``centroid.lon`` equal the known PLD
    null-centroid placeholder, are dropped rather than joined — see module docstring.

    Parameters
    ----------
    df           : pd.DataFrame with BNG easting/northing columns
    easting_col  : str
    northing_col : str

    Returns
    -------
    gpd.GeoDataFrame — input rows with a usable centroid, plus an ``LSOA21CD`` column
        (``NaN`` where the point falls outside every LSOA polygon)
    """
    usable = df.dropna(subset=[easting_col, northing_col]).copy()
    if 'centroid_lat' in usable.columns:
        lat = pd.to_numeric(usable['centroid_lat'], errors='coerce')
        lon = pd.to_numeric(usable['centroid_lon'], errors='coerce')
        is_placeholder = (lat.round(6) == round(_PLD_NULL_CENTROID[0], 6)) & (
            lon.round(6) == round(_PLD_NULL_CENTROID[1], 6)
        )
        usable = usable[~is_placeholder]

    easting = pd.to_numeric(usable[easting_col], errors='coerce')
    northing = pd.to_numeric(usable[northing_col], errors='coerce')
    geometry = [Point(xy) for xy in zip(easting, northing)]
    gdf = gpd.GeoDataFrame(usable, geometry=geometry, crs='EPSG:27700')

    lsoa_gdf = gla_data.load_boundaries(geography='lsoa', year=2021)
    return gpd.sjoin(gdf, lsoa_gdf[['LSOA21CD', 'geometry']], how='left', predicate='within')


def build_pivot(joined, fy_col, value_col='units'):
    """
    Aggregate a spatially-joined unit-level table into the LSOA x financial-year wide
    format the model expects (``LSOA Cd`` plus one ``YYYY/YY`` column per year), covering
    every London LSOA even where net change is zero across the whole window.

    Parameters
    ----------
    joined    : gpd.GeoDataFrame — output of ``assign_lsoa``, with an ``LSOA21CD`` column
    fy_col    : str — financial-year column to pivot on
    value_col : str — column to sum per (LSOA, year) cell

    Returns
    -------
    pd.DataFrame — one row per LSOA, columns ``LSOA Cd`` + sorted ``YYYY/YY`` year columns
    """
    lsoa_gdf = gla_data.load_boundaries(geography='lsoa', year=2021)

    pivot = joined.groupby(['LSOA21CD', fy_col])[value_col].sum().unstack(fy_col, fill_value=0.0)
    pivot = pivot.reindex(lsoa_gdf['LSOA21CD']).fillna(0.0)
    pivot = pivot.reset_index().rename(columns={'LSOA21CD': 'LSOA Cd'})

    year_cols = sorted(c for c in pivot.columns if c != 'LSOA Cd')
    return pivot[['LSOA Cd'] + year_cols]


def main(output_dir='.'):
    t0 = time.time()

    print('Extracting unit-level completions...')
    completions = extract_unit_level('completions')
    completions_joined = assign_lsoa(completions)
    completions_pivot = build_pivot(completions_joined, fy_col='fy_of_completion')
    completions_pivot.to_csv(f'{output_dir}/lsoa_completions_time_series_pivot_unit_level.csv', index=False)
    print(f'  -> {completions_pivot.shape}, {time.time() - t0:.0f}s elapsed')

    print('Extracting unit-level starts (cross-check only, not a model input)...')
    starts = extract_unit_level('starts')
    starts = add_unit_fy_commencement(starts)
    starts_joined = assign_lsoa(starts)
    starts_pivot = build_pivot(starts_joined, fy_col='fy_of_commencement_unit')
    starts_pivot.to_csv(f'{output_dir}/lsoa_starts_time_series_pivot_unit_level.csv', index=False)
    print(f'  -> {starts_pivot.shape}, {time.time() - t0:.0f}s total elapsed')


if __name__ == '__main__':
    main()
