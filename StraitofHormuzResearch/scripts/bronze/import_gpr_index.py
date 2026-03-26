# # Import Geopolitical Risk Index (GPR)
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
out_path = 's3a://ml-project-s3-bronze/input_folder/'

## Query
# Map country with country code
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

# Import `xls` from website
raw_df = pd.read_excel("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
                       header=0,
                       engine='xlrd')
raw_df.columns = [col.lower() for col in raw_df.columns]

# subset df to include dates after `2006-01-01`
raw_df = raw_df.loc[:, 'month':'gprhc_zaf']
raw_df['month'] = raw_df['month'].dt.strftime('%Y%m%d').astype(int)
raw_df = raw_df[raw_df['month']>=20060101]

# Convert pandas df to spark df
gpr_df = spark.createDataFrame(raw_df)

# Melt country columns into a single `gpr_index` variable
gpr_df = (
    gpr_df.melt(
        ids=['month'], 
        values=[f"gprc_{country_code.lower()}" for country_code in country_mapping.values()],
        variableColumnName='country', valueColumnName='gpr_index'
    )
)

# remap values in `country` variable.
gpr_df = (
    gpr_df
    .withColumn('country', F.split(F.col('country'), '_')[1])
    .replace({cntry_code.lower(): cntry_name for cntry_name, cntry_code in country_mapping.items()}, subset=['country'])
)

# separate joining variables
gpr_df = (
    gpr_df
    .withColumns({
        'year': F.substring(F.col('month').cast('string'), 1, 4).cast('int'),
        'month': F.substring(F.col('month').cast('string'), 5, 2).cast('int')
    })
    .select('country', 'year', 'month', 'gpr_index')
)

gpr_df.write.mode('overwrite').parquet(f"{out_path}/gpr.parquet")