import os

import boto3
import geopandas as gpd
import gla_data
import numpy as np
import pandas as pd
from shapely.geometry import Point

from housing_projections.config import (
    ALL_COLS_BEN,
    ALL_COLS_PLAN,
    INFER_COLS_BEN,
    INFER_COLS_PLAN,
)

# Default centre: Islington, north central London
DEFAULT_CENTER_LATLON = (51.544, -0.103)

_BEN_FILENAME = 'final_residential_uprn_net_changes_by_oa_fy (1).csv'
_PLD_FILENAME = 'lsoa_completions_time_series_pivot.csv'


def validate_data_path(data_path):
    """
    Check that the expected raw data files are present under ``data_path``.

    Raises
    ------
    FileNotFoundError
        With a clear message listing every missing file, so the caller knows
        exactly what to provide rather than getting a cryptic read error.
    """
    required = {
        'PLD completions': os.path.join(data_path, 'pld', _PLD_FILENAME),
        'BEN estimates':   os.path.join(data_path, 'ben', _BEN_FILENAME),
    }
    missing = {label: path for label, path in required.items() if not os.path.exists(path)}
    if missing:
        lines = '\n'.join(f'  [{label}]  {path}' for label, path in missing.items())
        raise FileNotFoundError(
            f"load_data: {len(missing)} required file(s) not found under {data_path!r}:\n{lines}"
        )


def _load_csv(location, file_name, s3=False) -> pd.DataFrame:
    if not s3:
        df = pd.read_csv(os.path.join(location, file_name), low_memory=False)
    else:
        s3_client = boto3.client('s3')
        obj_data = s3_client.get_object(Bucket=location, Key=file_name)
        df = pd.read_csv(obj_data['Body'], low_memory=False)
    return df


def _build_gdf(
    completions: pd.DataFrame,
    dwellings_2011_xw: pd.DataFrame,
    dwellings_2021: pd.DataFrame,
    df_ben: pd.DataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Pure merge/reshape step — no I/O. Joins pre-loaded DataFrames into the
    single GeoDataFrame that models consume.

    Parameters
    ----------
    completions      : PLD completions with LSOA21CD and planning columns
    dwellings_2011_xw: 2011 census dwellings crosswalked to 2021 boundaries
    dwellings_2021   : 2021 census dwellings
    df_ben           : BEN estimates aggregated to LSOA level (wide format)
    lsoa_gdf         : GeoDataFrame with LSOA21CD and geometry

    Returns
    -------
    gpd.GeoDataFrame — one row per LSOA
    """
    completions = completions.copy()
    completions['intercensal_completions'] = completions[INFER_COLS_PLAN].sum(axis=1)

    total_dwellings_census = dwellings_2021.merge(dwellings_2011_xw, on='LSOA21CD', how='inner')
    total_dwellings_census['intercensal_change'] = (
        total_dwellings_census['dwellings_2021'] - total_dwellings_census['dwellings_2011']
    )

    dwellings = pd.merge(total_dwellings_census, completions, on='LSOA21CD', how='left')
    dwellings = dwellings.merge(df_ben, on='LSOA21CD', how='left').fillna(0)

    dwellings.insert(4, 'total_change_2011_to_2021_ben',
                     dwellings.pop('total_change_2011_to_2021_ben'))
    dwellings.insert(5, 'intercensal_completions',
                     dwellings.pop('intercensal_completions'))

    return gpd.GeoDataFrame(dwellings.merge(lsoa_gdf, on='LSOA21CD'))


def load_data(data_path):
    """
    Load, merge, and return a GeoDataFrame of London LSOAs with all data
    needed for modelling: census dwelling counts (2011 and 2021), planning
    completions (PLD), and BEN current estimates.

    Parameters
    ----------
    data_path : str or Path
        Root directory containing the raw data subdirectories:

        - ``pld/lsoa_completions_time_series_pivot.csv``
        - ``ben/final_residential_uprn_net_changes_by_oa_fy (1).csv``

    Returns
    -------
    gpd.GeoDataFrame
        One row per LSOA (2021 boundaries). Key columns:
        ``LSOA21CD``, ``dwellings_2011``, ``dwellings_2021``,
        ``intercensal_change``, planning columns (INFER_COLS_PLAN),
        BEN columns (INFER_COLS_BEN), plus ``geometry``.
    """
    validate_data_path(data_path)

    # ── I/O shell — four external reads ──────────────────────────────────────
    completions = _load_csv(
        os.path.join(data_path, 'pld'),
        'lsoa_completions_time_series_pivot.csv',
    )
    completions.rename(columns={'LSOA Cd': 'LSOA21CD'}, inplace=True)

    dwellings_2021 = gla_data.load_census_dwellings(year=2021).rename(
        columns={'dwellings_total': 'dwellings_2021'}
    )
    dwellings_2011 = gla_data.load_census_dwellings(year=2011).rename(
        columns={'dwellings_total': 'dwellings'}
    )
    dwellings_2011_xw = gla_data.crosswalk(
        dwellings_2011, from_year=2011, to_year=2021, value_cols=['dwellings']
    ).rename(columns={'dwellings': 'dwellings_2011'})

    df_ben_raw = pd.read_csv(
        os.path.join(data_path, 'ben', _BEN_FILENAME),
    )
    df_ben = gla_data.aggregate(
        df_ben_raw[['OA21CD', 'financial_year', 'uprn_net_change']],
        from_geography='oa',
        to_geography='lsoa',
        value_cols=['uprn_net_change'],
        year=2021,
    ).set_index(['LSOA21CD', 'financial_year'])
    df_ben = df_ben.unstack().fillna(0).droplevel(0, axis=1)
    df_ben.columns = [f'{x}_ben' for x in df_ben.columns]
    df_ben = df_ben.reset_index()
    df_ben['total_change_2011_to_2021_ben'] = df_ben[INFER_COLS_BEN].sum(axis=1)

    lsoa_gdf = gla_data.load_boundaries(geography='lsoa', year=2021)[['LSOA21CD', 'geometry']]

    # ── Pure transform ────────────────────────────────────────────────────────
    return _build_gdf(completions, dwellings_2011_xw, dwellings_2021, df_ben, lsoa_gdf)


def select_spatial_sample(gdf, n_areas=200, center_latlon=DEFAULT_CENTER_LATLON):
    """
    Select a spatially contiguous sample of LSOAs within a rough circle
    around a centre point. Used as the standard sampling function across
    all models to ensure consistent comparison.

    Parameters
    ----------
    gdf          : GeoDataFrame — clean GeoDataFrame of all LSOAs
    n_areas      : int — number of LSOAs to include
    center_latlon: (lat, lon) tuple in WGS84

    Returns
    -------
    GeoDataFrame of selected LSOAs, reset index
    """
    center_gdf = gpd.GeoDataFrame(
        geometry=[Point(center_latlon[1], center_latlon[0])],
        crs='EPSG:4326'
    ).to_crs(gdf.crs)

    center_point = center_gdf.geometry.iloc[0]
    distances    = gdf.geometry.centroid.distance(center_point)
    sample_idx   = distances.nsmallest(n_areas).index

    gdf_sample = gdf.loc[sample_idx].copy().reset_index(drop=True)

    print(f"Selected {len(gdf_sample)} LSOAs centred on {center_latlon}")
    print(f"Bounds: {gdf_sample.total_bounds}")

    return gdf_sample


def make_data_dict(gdf, n_areas=None):
    """
    Convert a GeoDataFrame into the dict format expected by all models
    and diagnostic functions.

    Parameters
    ----------
    gdf      : gpd.GeoDataFrame — output of ``load_data()`` or ``select_spatial_sample()``
    n_areas  : int or None — if given, take only the first ``n_areas`` rows (for quick tests)

    Returns
    -------
    dict with keys:
        ``D``            — census intercensal change (n_areas,)
        ``P_obs``        — planning completions (n_areas, n_years)
        ``E_obs``        — BEN estimates (n_areas, n_years)
        ``P_obs_full``   — planning over full time window (n_areas, n_years_full)
        ``E_obs_full``   — BEN over full time window (n_areas, n_years_full)
        ``n_years``      — number of inference years
        ``n_years_full`` — number of years in full time window
        ``n_areas``      — number of LSOAs
        ``gdf``          — GeoDataFrame (possibly subsetted)
        ``D_full_mean``  — mean intercensal change over the full (unsubsetted) dataset
    """
    D_full_mean = float(
        (gdf['dwellings_2021'] - gdf['dwellings_2011']).mean()
    )

    if n_areas is not None:
        gdf = gdf.iloc[:n_areas].copy().reset_index(drop=True)

    D     = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values.astype(float)
    P_obs = gdf[INFER_COLS_PLAN].values.astype(float)
    E_obs = gdf[INFER_COLS_BEN].values.astype(float)

    # Snap erroneous P records to zero: cells where P is non-zero but records
    # less than 10% of E are likely PLD data errors (e.g. geocoding or batch
    # mis-allocation) rather than genuine low completions. Treat as missing.
    erroneous = (P_obs > 0) & (E_obs > 0) & (P_obs < 0.1 * E_obs)
    P_obs[erroneous] = 0.0

    P_obs_full = gdf[ALL_COLS_PLAN].values.astype(float)
    E_obs_full = gdf[ALL_COLS_BEN].values.astype(float)

    # Empirical P-missingness rate per area, conditioned on E being active.
    # P=0 when E=0 reflects no completions, not a recording gap.
    # Areas with no active E years default to pi_miss=1 (nothing to record).
    e_active = E_obs > 0                          # (n_areas, n_years)
    n_active = e_active.sum(axis=1)               # (n_areas,)
    p_zero_when_active = ((P_obs == 0) & e_active).sum(axis=1)
    pi_miss_empirical = np.where(n_active > 0, p_zero_when_active / n_active, 1.0)
    # Clip away from exact 0/1: pi_miss=1 combined with any non-zero P (e.g.
    # in E=0 years) gives -inf log-likelihood in the zero-inflated mixture.
    pi_miss_empirical = np.clip(pi_miss_empirical, 0.01, 0.99)

    return {
        'D':                D,
        'P_obs':            P_obs,
        'E_obs':            E_obs,
        'P_obs_full':       P_obs_full,
        'E_obs_full':       E_obs_full,
        'pi_miss_empirical': pi_miss_empirical,
        'n_years':          len(INFER_COLS_PLAN),
        'n_years_full':     len(ALL_COLS_PLAN),
        'n_areas':          len(gdf),
        'gdf':              gdf,
        'D_full_mean':      D_full_mean,
    }


def make_borough_idx(gdf):
    """
    Derive a per-LSOA borough (LAD) index from the GLA LSOA-to-LAD crosswalk.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame with an 'LSOA21CD' column. Pass data['gdf']
          (post make_data_dict) rather than the pre-subset sample, so the
          returned array's row order always matches the data dict a model
          is actually built from.

    Returns
    -------
    borough_idx   : np.ndarray[int], shape (len(gdf),) — 0-indexed borough
                    id per row, same order as gdf.
    n_boroughs    : int
    borough_codes : np.ndarray[str], shape (n_boroughs,) — sorted LAD22CD
                    values; borough_codes[borough_idx[i]] recovers row i's LAD.

    Raises
    ------
    ValueError — lists any LSOA21CD in gdf with no match in the crosswalk.
    """
    lookup = gla_data.load_geography_lookup(year=2021, smallest_geography='lsoa')
    lookup = lookup[['LSOA21CD', 'LAD22CD']].drop_duplicates('LSOA21CD')

    merged = gdf[['LSOA21CD']].merge(lookup, on='LSOA21CD', how='left')
    assert len(merged) == len(gdf)

    unmapped = merged.loc[merged['LAD22CD'].isna(), 'LSOA21CD'].tolist()
    if unmapped:
        raise ValueError(
            f"make_borough_idx: {len(unmapped)} LSOA21CD code(s) not found "
            f"in the LAD crosswalk: {unmapped}")

    borough_codes, borough_idx = np.unique(merged['LAD22CD'].values, return_inverse=True)
    return borough_idx.astype(int), int(len(borough_codes)), borough_codes
