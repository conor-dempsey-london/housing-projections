# %%
import boto3
import pandas as pd 
import geopandas as gpd
from housing_projections.load import load_permissions_data_s3
from dotenv import load_dotenv
load_dotenv()

DATA_BUCKET             = os.getenv('DATA_BUCKET')
PERMISSIONS_DATA_FILE   = os.getenv('PERMISSIONS_DATA_FILE')
LSOA_GEOMETRY_FOLDER    = os.getenv('LSOA_GEOMETRY_FOLDER')

permissions_df = load_permissions_data_s3(DATA_BUCKET, PERMISSIONS_DATA_FILE)

# # Load the data
# s3 = boto3.client('s3') 
# obj_data = s3.get_object(Bucket= DATA_BUCKET, Key= f'{LSOA_GEOMETRY_FOLDER}/Bexley.shp') 

# # get object and file (key) from bucket
# bexley_lsoa_geom = gpd.read_file(obj_data['Body']) 
    
# %%

permissions_df['completed_date'] = pd.to_datetime(permissions_df[permissions_df['status_rc_per']=='COMPLETED']['completed_date_per'].astype(int).astype(str), format="%Y%m%d")

# %%

completed_small_df = permissions_df[['completed_date', 'no_of_bedrooms_line', 'no_of_prop_units_line', 'no_of_exist_units_line', 'no_of_beds_line', 'status_rc_per', 'post_code_per', 'LSOA11CD']][permissions_df['status_rc_per'] == 'COMPLETED']


# %%


