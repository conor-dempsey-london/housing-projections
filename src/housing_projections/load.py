import boto3
import pandas as pd

def load_csv_s3(
    bucket, file_name
) -> pd.DataFrame:

    # Load the data
    s3 = boto3.client('s3') 
    obj_data = s3.get_object(Bucket= bucket, Key= file_name) 

    # get object and file (key) from bucket
    df = pd.read_csv(
        obj_data['Body'], 
        low_memory=False) 

    return df