# # Import US Energy Information Administration (EIA) daily oil price data
# - Author: Bryan Bravo
# - Created: 2026-03-18
# ## Import Libraries
########################## AWS Glue environment #######################################

# Libaries that are mainly used
import os
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import reduce
from pyspark.sql import (
    functions as F,
    Window as W,
    types as T,
    SparkSession,
    DataFrame
)

# AWS Glue spec. libraries
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'EIA_API_KEY'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()

#######################################################################################

## Variables
api_key = args['EIA_API_KEY']
end_date = (dt.now().replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
out_path = 's3a://ml-project-s3-bronze/input_folder/'

## Custom Functions
def get_oil_data(col_name, series_name, end_date, api_key):
    try:
        print(f"Importing from EIA API {col_name.upper()}[{series_name}]")
        # Get first batch of data = dates {'2006-01-01' through '2024-12-31'}
        response = requests.get(
            f"https://api.eia.gov/v2/petroleum/pri/spt/data/?frequency=daily&data[0]=value&facets[series][]={series_name}"+
            "&start=2006-01-01&end=2024-12-31&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"+
            f"&api_key={api_key}"
        )
        eia_data1 = response.json()

        # Get final batch of data {'2025-01-01' through provided value}
        response = requests.get(
            f"https://api.eia.gov/v2/petroleum/pri/spt/data/?frequency=daily&data[0]=value&facets[series][]={series_name}"+
            f"&start=2025-01-01&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"+
            f"&api_key={api_key}"
        )
        eia_data2 = response.json()

        # Create Pandas df and union datasets.
        df = (
            pd.concat([
                pd.DataFrame(eia_data1['response']['data'])[['period', 'value']],
                pd.DataFrame(eia_data2['response']['data'])[['period', 'value']]
                ], axis=0, ignore_index=True)
        )

        # Convert to PySpark and update variable names and dtypes
        df = (
            spark.createDataFrame(df)
            .withColumns({
                'date': F.date_format(F.to_date(F.col('period'), 'yyyy-MM-dd'), 'yyyyMMdd').cast('int'),
                f'{col_name}_dollars_per_barrel': F.col('value').cast('double')
            })
            .select('date', f'{col_name}_dollars_per_barrel')
        )
        print(f"Import for [{series_name}] Successful!\n************************************")
        return df
    except Exception as e:
        print(f"✗ Error fetching EIA data for {series_name}: {str(e)[:100]}")

## Query
### Import Crude Oil Price Data 
oil_dict = {
    'brent': 'RBRTE',
    'wti': 'RWTC'
}

globals().update({
    f"{oil_name}_df": get_oil_data(oil_name, series_name, end_date=end_date, api_key=api_key)
    for oil_name, series_name in oil_dict.items()
})

print("Joining and caching Spark DFs")
oil_df = brent_df.join(wti_df, on=['date'], how='inner')
oil_df.repartition().cache().count()

oil_df.write.mode('overwrite').parquet(f'{out_path}/oil.parquet')