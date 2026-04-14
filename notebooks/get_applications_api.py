# %%

from elasticsearch import Elasticsearch
from pandas import json_normalize
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

# %%

client = Elasticsearch(
    hosts=["https://planningdata.london.gov.uk/api-guest/"],
    api_key="be2rmRnt&",
)

# %%

resp = client.search(
    index="applications",
    size=20,
    query={
        "bool": {
            "must": [
                {
                    "term": {"status.raw": "Completed"} 
                }
            ]
        }, 
    },
    source = [
        "actual_completion_date",
        "application_details.no_additional_bedrooms",
        "application_details.residential_details.residential_units.actual_completion_date",
        "application_details.residential_details.residential_units.change_type",
        "application_details.residential_details.residential_units.unit_type",
        "application_details.residential_details.residential_units.unit_development_type",
        "application_details.residential_details.residential_units.provider",
        "application_details.residential_details.residential_units.no_habitable_rooms",
        "application_details.residential_details.residential_units.unit_no",
        "application_details.residential_details.residential_units.unit_development_type.raw",
        "application_details.residential_details.residential_units.actual_commencement_date",
        "application_details.residential_details.residential_units.no_bedrooms",
        "application_details.residential_details.residential_units.tenure",
        "application_details.residential_details.residential_units.gia",
        "status",
        "epc_number",
        "postcode",
        "uprn",
        "cenroid",
        "centroid_easting",
        "centroid_northing",
        "wgs84_polygon"
    ],

)

# %%
resp_df = json_normalize(
    resp['hits']['hits'], 
)

# %%

resp_df = resp_df.explode(
    '_source.application_details.residential_details.residential_units'
    )

resp_df['_source.application_details.residential_details.residential_units'] = resp_df['_source.application_details.residential_details.residential_units'].where(resp_df['_source.application_details.residential_details.residential_units'].notna(), lambda x: [{}])

resp_df = pd.concat([resp_df.drop(columns=['_source.application_details.residential_details.residential_units']), pd.json_normalize(resp_df['_source.application_details.residential_details.residential_units'])], axis=1)


# %%
