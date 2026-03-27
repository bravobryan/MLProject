# # Import International Monetary Fund (IMF) data
# - Author: Bryan Bravo
# - Created: 2026-03-19
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

import sdmx

# AWS Glue spec. libraries
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()

#######################################################################################

## Variables
end_date = (dt.now().replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
out_path = 's3a://ml-project-s3-bronze/input_folder/'

## Query


country_mapping = {
    'australia': 'AUS',
    'brazil': 'BRA',
    'canada': 'CAN',
    'china': 'CHN',
    # 'euro': 'EU',  # No IMF Data, must source from elsewhere.
    'france': 'FRA',
    'germany': 'DEU',
    'india': 'IND',
    'italy': 'ITA',
    'japan': 'JPN',
    'mexico': 'MEX',
    'south_korea': 'KOR',
    'russia': 'RUS',
    'south_africa': 'ZAF',
    'turkiye': 'TUR',
    'united_kingdom': 'GBR',
    'united_states': 'USA'
}
countries = [country for country in country_mapping.values()]

key = f"{'+'.join(countries)}.CPI._T.IX.M"

print(f"Requesting data for key: {key} starting {2006}...")
IMF_DATA = sdmx.Client('IMF_DATA')
try:
    data_msg = IMF_DATA.data('CPI', key=key, params={'startPeriod': 2006, 'endPeriod': end_date[0:4]})
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

cpi_df = sdmx.to_pandas(data_msg).reset_index()

cpi_df.columns = [col.lower() for col in cpi_df.columns]
# Subset columns for use
cpi_df = cpi_df[['time_period', 'country', 'value']]

# Extract year and Month
cpi_df[['year', 'month']] = cpi_df['time_period'].str.split('-', expand=True)
cpi_df['year'] = cpi_df['year'].astype(int)
cpi_df['month'] = cpi_df['month'].str[1:].astype(int)


# Remap countries
cpi_df['country'] = cpi_df['country'].map({
    code: cntry for cntry, code in country_mapping.items()
})

cpi_df.drop('time_period', axis=1, inplace=True)

# Convert to Spark DF
cpi_df = spark.createDataFrame(cpi_df)
cpi_df.repartition(10).cache().count()


cpi_df.write.mode('overwrite').parquet(f'{out_path}/cpi.parquet')
spark.stop()