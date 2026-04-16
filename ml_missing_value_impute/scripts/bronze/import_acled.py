# # Import ACLED data
# - Author: Bryan Bravo
# - Created: 2026-03-23
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

# ### Import ACLED: Number of political violence events by country-month-year
# This includes all battle, explosions/remote violence, and violence against civilians events.
# I'm currently importing from local environment, since dataset requires consent to import.

# Check acleddata.com for updated excel file.

raw_data = pd.read_excel('s3a://ml-project-s3-bronze/excel_files/number_of_political_violence_events_by_country-month-year_as-of-13Mar2026.xlsx',
                         header=0)
acled_df = raw_data.copy()
acled_df.columns = [col.lower() for col in acled_df.columns]

# subset for countries in the analysis
acled_df['country'] = acled_df['country'].str.lower().str.strip()
acled_df = acled_df[acled_df['country'].isin(country_name.replace('_', ' ') for country_name in country_mapping.keys())]
acled_df['country'] = acled_df['country'].str.replace(' ', '_')

# subset for year greater than or equal to 2006
acled_df = acled_df[acled_df['year'] >= 2006]

# remap month to integer
acled_df['month'] = acled_df['month'].map({
    mon: str(i+1) for i, mon in enumerate(['January', 'February', 'March', 'April', 'May', 'June',
                                    'July', 'August', 'September', 'October', 'November', 'December'])
    }).astype(int)

# Convert to Spark DF
acled_df = spark.createDataFrame(acled_df)
acled_df.repartition(10).cache().count()

acled_df.write.mode('overwrite').parquet(f"{out_path}/acled.parquet")