import boto3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

from housing_projections.config import (
    INFER_COLS_PLAN, INFER_COLS_BEN,
    ALL_COLS_PLAN, ALL_COLS_BEN,
    LONDON_LAS
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

    # LSOA 2011-2021 exact lookup
    lsoa_11_21_lookup = pd.read_csv(
        os.path.join(data_path, 'ons' ,'lsoa_2011_to_2021_exact_fit_lookup.csv'),
    )

    # LSOA 2021 geometry
    df_lsoa= gpd.read_parquet(
        os.path.join(data_path, 'location', 'lsoa.parquet')
    )

    london_lsoas_lookup = lsoa_11_21_lookup.loc[lsoa_11_21_lookup['LAD22NM'].isin(LONDON_LAS),:]

    # read census dwellings data for 2011 and 2021
    dwellings_2011 = pd.read_csv("https://ukds-ckan.s3.eu-west-1.amazonaws.com/DWLTYP/DWLTYP_LSOADZ_England_Northern_Ireland_Scotland_Wales_Descriptions.csv")

    dwellings_2021 = pd.read_excel('https://ukds-ckan.s3.eu-west-1.amazonaws.com/2021/ONS/number-of-dwellings/RM204-Number-Of-Dwellings-2021-lsoa-ONS.xlsx', sheet_name='Dataset')

    dwellings_2011.rename(columns={
        'GEO_CODE':'LSOA11CD',
        r'Dwellings : Total\ Dwellings - Unit : Dwellings': 'dwellings'
        }, inplace=True)

    dwellings_2021.rename(columns={
        'Lower layer Super Output Areas Code':'LSOA21CD',
        'Observation': 'dwellings'
        }, inplace=True)

    dwellings_london_2011 = dwellings_2011.loc[
        dwellings_2011['LSOA11CD'].isin(london_lsoas_lookup['LSOA11CD']),
        ['LSOA11CD', 'dwellings']]

    dwellings_london_2021 = dwellings_2021.loc[
        dwellings_2021['LSOA21CD'].isin(london_lsoas_lookup['LSOA21CD']),
        ['LSOA21CD', 'dwellings']]

    dwellings_2011_w_lookup = (
        dwellings_london_2011
        .merge(london_lsoas_lookup, how='outer', on='LSOA11CD')
        [['LSOA11CD', 'dwellings', 'LSOA21CD', 'CHGIND']]
    )

    dwellings_2011_w_lookup['dwellings'] = dwellings_2011_w_lookup.groupby('LSOA11CD')['dwellings'].transform(lambda x: x.mean() / x.count())

    dwellings_2011_w_lookup = dwellings_2011_w_lookup.groupby('LSOA21CD')[['dwellings', 'CHGIND']].agg({'CHGIND': 'first', 'dwellings': 'sum'}).reset_index()

    # get currenet estimates supplied by Ben
    df_ben = pd.read_csv(
        os.path.join(data_path, 'ben', 'final_residential_uprn_net_changes_by_oa_fy (1).csv'),
    )

    # lookup from OA to LSOA 2021 for aggregating Ben's estimates to LSOA level
    lookup_df = pd.read_csv(
        os.path.join(data_path, 'ons', 'oa_lookup.csv'),
    )

    lookup_df = lookup_df[['oa21cd', 'lsoa21cd']].groupby('oa21cd').first().reset_index()

    df_ben = df_ben.merge(lookup_df, left_on='OA21CD', right_on='oa21cd', how='left')

    df_ben = df_ben[['financial_year', 'uprn_net_change', 'lsoa21cd']].groupby(['lsoa21cd', 'financial_year']).sum()

    df_ben = df_ben.unstack().fillna(0).droplevel(0, axis=1)
    cols_ben = [f'{x}_ben' for x in df_ben.columns]
    df_ben.columns = cols_ben

    df_ben['total_change_2011_to_2021_ben'] = df_ben[INFER_COLS_BEN].sum(axis=1)

    # combine census, completions, and current estimates into a single dataframe
    total_dwellings_census = pd.merge(
        dwellings_london_2021,
        dwellings_2011_w_lookup, 
        on='LSOA21CD',
        suffixes=('_2021', '_2011'),
        how='inner'
    )[['LSOA21CD', 'dwellings_2011', 'dwellings_2021']]

    total_dwellings_census['intercensal_change'] = (
        total_dwellings_census['dwellings_2021'] - total_dwellings_census['dwellings_2011']
    )

    dwellings = pd.merge(
        total_dwellings_census,
        completions,
        on='LSOA21CD',
        how='left'
    )

    dwellings = dwellings.merge(df_ben, left_on='LSOA21CD', right_on='lsoa21cd', how='left').fillna(0)

    dwellings.insert(4, 'total_change_2011_to_2021_ben', dwellings.pop('total_change_2011_to_2021_ben'))
    dwellings.insert(5, 'intercensal_completions', dwellings.pop('intercensal_completions'))

    dwellings = gpd.GeoDataFrame(
        pd.merge(
            dwellings, 
            df_lsoa[['target_id', 'geometry']], 
            left_on='LSOA21CD', 
            right_on='target_id',
        ).drop(columns='target_id'),
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