# # Import UN Trade and Development (UNCTAD) API data

# - Author: Bryan Bravo
# - Created: 2026-03-24
# - Modified by Bryan Bravo on 2026-04-03: Adjusted time period for imports data to start from 2005 to capture more historical data, as some countries have missing values for 2006. Also added filtering to include only USD values for FX reserves, as there are additional rows with unit measure 'XDR' that are not relevant for our analysis.
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

### Import FX monthly reserves
def import_wb_fx_reserves(country_code, end_date):
    response = requests.get(
        f"https://data360api.worldbank.org/data360/data?DATABASE_ID=IMF_IFS&INDICATOR=IMF_IFS_RAXG&REF_AREA={country_code}" +
        f"&FREQ=M&timePeriodFrom=2006-01&timePeriodTo={end_date[:7]}&skip=0"
    )
    wb_data = response.json()

    if 'error_code' in wb_data:  # check if response is None
        print(f"✗ WB API error: {wb_data.get('error_code', 'error_message')}")

    wb_df = pd.DataFrame(wb_data['value'])
    wb_df.columns = [col.lower() for col in wb_df.columns]
    ## Corrected on 04-03-2026: 
        # There are additional rows where the unit measure has ['USD', 'XDR'], filtering to include only USD values.
    wb_df = wb_df[wb_df['unit_measure'] == 'USD'] 
    wb_df = wb_df[['obs_value', 'ref_area', 'time_period']]
    return wb_df


fx_df = pd.DataFrame(columns=['obs_value', 'ref_area', 'time_period'])

for country_code in country_mapping.values():
    print(f"Importing {country_code} from World Bank Data")
    fx_df = pd.concat([fx_df, import_wb_fx_reserves(country_code, end_date)], ignore_index=True)

fx_df[['year', 'month']] = fx_df['time_period'].str.split('-', expand=True).astype(int)
fx_df['country'] = fx_df['ref_area'].map({code: name for name, code in country_mapping.items()})
fx_df['fx_reserves'] = fx_df['obs_value'].astype(float)

fx_df = fx_df[['country', 'year', 'month', 'fx_reserves']]

### Import `Imports of goods and services (current US$)`

# https://data360.worldbank.org/en/indicator/WB_WDI_NE_IMP_GNFS_CD
def import_wb_imports_dol(country_code, end_date):
    response = requests.get(
        f"https://data360api.worldbank.org/data360/data?DATABASE_ID=WB_WDI&INDICATOR=WB_WDI_NE_IMP_GNFS_CD&REF_AREA={country_code}" +
        f"&timePeriodFrom=2005-01&timePeriodTo={end_date[:4]}&skip=0"  # Corrected on 04-03-2026: Adjusted time period to start from 2005 to capture more historical data, as some countries have missing values for 2006.
    )
    wb_data = response.json()

    if 'error_code' in wb_data:  # check if response is None
        print(f"✗ WB API error: {wb_data.get('error_code', 'error_message')}")

    wb_df = pd.DataFrame(wb_data['value'])
    wb_df.columns = [col.lower() for col in wb_df.columns]
    wb_df = wb_df[['obs_value', 'ref_area', 'time_period']]
    return wb_df

import_df = pd.DataFrame(columns=['obs_value', 'ref_area', 'time_period'])

for country_code in country_mapping.values():
    print(f"Importing {country_code} from World Bank Data")
    import_df = pd.concat([import_df, import_wb_imports_dol(country_code, end_date)], ignore_index=True)
    print(f"Finished importing {country_code}")

import_df['year'] = import_df['time_period'].astype(int)
import_df['country'] = import_df['ref_area'].map({code: name for name, code in country_mapping.items()})
import_df['imports_good_service'] = import_df['obs_value'].astype(float) / 12  # Computes approximate month from annual

import_df = import_df[['country', 'year', 'imports_good_service']]

## Join DFs
wb_df = spark.createDataFrame(
    fx_df.merge(import_df, how='outer', on=['country', 'year'])
    )

wb_df = (
    wb_df
    .withColumn('imports_good_service',  # Making correction on how nulls are recorded.
                F.when(F.col('imports_good_service').contains("NaN"), F.lit(None)).otherwise(F.col('imports_good_service'))
    )
)

wb_df.repartition(10).cache().count()

wb_df.write.mode('overwrite').parquet(f"{out_path}/wb.parquet")