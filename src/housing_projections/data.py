import boto3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

import gla_data

from housing_projections.config import (
    INFER_COLS_PLAN, INFER_COLS_BEN,
    ALL_COLS_PLAN, ALL_COLS_BEN,
)


# Default centre: Islington, north central London
DEFAULT_CENTER_LATLON = (51.544, -0.103)

def load_csv(
    location,
    file_name,
    s3=False
) -> pd.DataFrame:

    if not s3:
        df = pd.read_csv(
            os.path.join(location, file_name),
            low_memory=False)
    else:
        # Load the data
        s3 = boto3.client('s3')
        obj_data = s3.get_object(Bucket= location, Key= file_name)

        # get object and file (key) from bucket
        df = pd.read_csv(
            obj_data['Body'],
            low_memory=False)

    return df


def load_data(data_path):

    # get completions
    completions = load_csv(
        os.path.join(data_path, 'pld'),
        'lsoa_completions_time_series_pivot.csv',
    )

    completions.rename(columns={
        'LSOA Cd': 'LSOA21CD',
    }, inplace=True)

    completions['intercensal_completions'] = completions[INFER_COLS_PLAN].sum(axis=1)

    # census dwelling counts via gla_data
    dwellings_2021 = gla_data.load_census_dwellings(year=2021).rename(
        columns={'dwellings_total': 'dwellings_2021'}
    )
    dwellings_2011 = gla_data.load_census_dwellings(year=2011).rename(
        columns={'dwellings_total': 'dwellings'}
    )

    # crosswalk 2011 dwellings onto 2021 LSOA boundaries
    dwellings_2011_xw = gla_data.crosswalk(
        dwellings_2011, from_year=2011, to_year=2021, value_cols=['dwellings']
    ).rename(columns={'dwellings': 'dwellings_2011'})

    # get current estimates supplied by Ben
    df_ben = pd.read_csv(
        os.path.join(data_path, 'ben', 'final_residential_uprn_net_changes_by_oa_fy (1).csv'),
    )

    # OA to LSOA 2021 lookup for aggregating Ben's estimates to LSOA level
    oa_lookup = gla_data.load_geography_lookup(year=2021, smallest_geography='oa')[['OA21CD', 'LSOA21CD']]

    df_ben = df_ben.merge(oa_lookup, on='OA21CD', how='left')

    df_ben = df_ben[['financial_year', 'uprn_net_change', 'LSOA21CD']].groupby(['LSOA21CD', 'financial_year']).sum()

    df_ben = df_ben.unstack().fillna(0).droplevel(0, axis=1)
    cols_ben = [f'{x}_ben' for x in df_ben.columns]
    df_ben.columns = cols_ben

    df_ben['total_change_2011_to_2021_ben'] = df_ben[INFER_COLS_BEN].sum(axis=1)

    # combine census, completions, and current estimates into a single dataframe
    total_dwellings_census = dwellings_2021.merge(
        dwellings_2011_xw, on='LSOA21CD', how='inner'
    )

    total_dwellings_census['intercensal_change'] = (
        total_dwellings_census['dwellings_2021'] - total_dwellings_census['dwellings_2011']
    )

    dwellings = pd.merge(
        total_dwellings_census,
        completions,
        on='LSOA21CD',
        how='left'
    )

    dwellings = dwellings.merge(df_ben, left_on='LSOA21CD', right_index=True, how='left').fillna(0)

    dwellings.insert(4, 'total_change_2011_to_2021_ben', dwellings.pop('total_change_2011_to_2021_ben'))
    dwellings.insert(5, 'intercensal_completions', dwellings.pop('intercensal_completions'))

    lsoa_gdf = gla_data.load_boundaries(geography='lsoa', year=2021)[['LSOA21CD', 'geometry']]

    dwellings = gpd.GeoDataFrame(
        dwellings.merge(lsoa_gdf, on='LSOA21CD')
    )

    return dwellings


def select_spatial_sample(gdf, n_areas=200,
                           center_latlon=DEFAULT_CENTER_LATLON):
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

    gdf_sample   = gdf.loc[sample_idx].copy().reset_index(drop=True)

    print(f"Selected {len(gdf_sample)} LSOAs centred on {center_latlon}")
    print(f"Bounds: {gdf_sample.total_bounds}")

    return gdf_sample

def make_data_dict(gdf, n_areas=None):

    D_full_mean = float(
        (gdf['dwellings_2021'] - gdf['dwellings_2011']).mean()
    )

    if n_areas is not None:
        gdf = gdf.iloc[:n_areas].copy().reset_index(drop=True)

    D     = (gdf['dwellings_2021'] - gdf['dwellings_2011']).values.astype(float)
    P_obs = gdf[INFER_COLS_PLAN].values.astype(float)
    E_obs = gdf[INFER_COLS_BEN].values.astype(float)

    # Full time series including out-of-window years
    P_obs_full = gdf[ALL_COLS_PLAN].values.astype(float)
    E_obs_full = gdf[ALL_COLS_BEN].values.astype(float)

    return {
        'D':            D,
        'P_obs':        P_obs,
        'E_obs':        E_obs,
        'P_obs_full':   P_obs_full,
        'E_obs_full':   E_obs_full,
        'n_years':      len(INFER_COLS_PLAN),
        'n_years_full': len(ALL_COLS_PLAN),
        'n_areas':      len(gdf),
        'gdf':          gdf,
        'D_full_mean':  D_full_mean,
    }
