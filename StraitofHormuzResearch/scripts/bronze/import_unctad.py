# # Import UN Trade and Development (UNCTAD) API data

# - Author: Bryan Bravo
# - Created: 2026-03-24
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
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'UNCTAD_CLIENT_ID', 'UNCTAD_API_KEY'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()

import gzip
import io
#######################################################################################

## Variables
end_date = (dt.now().replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
out_path = 's3a://ml-project-s3-bronze/input_folder/'

CLIENT_ID = args['UNCTAD_CLIENT_ID']
CLIENT_SECRET = args['UNCTAD_API_KEY']
TEMP_FILE_PATH = "s3a://ml-project-s3-bronze/excel_files/"  # to produce csv for import

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

### Import LSCI 



# Configuration
URL = "https://unctadstat-user-api.unctad.org/US.LSCI_M/cur/Facts?culture=en"

# The same filter string from your R script (kept as a single string)
FILTER = (
    "Economy/Code in ('008','012','016','024','660','028','032','533','036','044','048','050',"
    "'052','056','084','204','060','535','076','092','096','100','132','116','120','124','136',"
    "'152','156','344','158','162','166','170','174','178','184','188','384','191','192','531',"
    "'196','408','180','208','262','212','214','218','818','222','226','232','233','238','234',"
    "'242','246','250','254','258','266','270','268','276','288','292','300','304','308','312',"
    "'316','320','831','324','624','328','332','340','352','356','360','364','368','372','376',"
    "'380','388','392','832','400','404','296','414','428','422','430','434','440','450','458',"
    "'462','470','584','474','478','480','175','484','583','499','500','504','508','104','516',"
    "'520','528','530','540','554','558','566','570','574','580','578','512','586','585','591',"
    "'598','600','604','608','616','620','630','634','410','498','638','642','643','654','659',"
    "'662','666','670','882','678','682','686','891','690','694','702','534','705','090','706',"
    "'710','724','144','729','736','740','752','760','764','626','768','776','780','788','792',"
    "'796','798','804','784','826','834','840','850','858','548','862','704','876','887') and "
    "Month/Code in ('2006M02','2006M05','2006M08','2006M11','2007M02','2007M05','2007M08','2007M11',"
    "'2008M02','2008M05','2008M08','2008M11','2009M02','2009M05','2009M08','2009M11','2010M02',"
    "'2010M05','2010M08','2010M11','2011M02','2011M05','2011M08','2011M11','2012M02','2012M05',"
    "'2012M08','2012M11','2013M02','2013M05','2013M08','2013M11','2014M02','2014M05','2014M08',"
    "'2014M11','2015M02','2015M05','2015M08','2015M11','2016M02','2016M05','2016M08','2016M11',"
    "'2017M02','2017M05','2017M08','2017M11','2018M02','2018M05','2018M08','2018M11','2019M02',"
    "'2019M05','2019M08','2019M11','2020M02','2020M05','2020M08','2020M11','2021M02','2021M05',"
    "'2021M08','2021M11','2022M02','2022M05','2022M08','2022M11','2023M01','2023M02','2023M03',"
    "'2023M04','2023M05','2023M06','2023M07','2023M08','2023M09','2023M10','2023M11','2023M12',"
    "'2024M01','2024M02','2024M03','2024M04','2024M05','2024M06','2024M07','2024M08','2024M09',"
    "'2024M10','2024M11','2024M12','2025M01','2025M02','2025M03','2025M04','2025M05','2025M06',"
    "'2025M07','2025M08','2025M09','2025M10','2025M11','2025M12','2026M01','2026M02','2026M03')"
)

FORM = {
    "$select": "Economy/Label ,Month/Label, Index_Average_M2_2023__100_Value, Index_Average_M2_2023__100_Footnote, Index_Average_M2_2023__100_MissingValue",
    "$filter": FILTER,
    "$orderby": "Economy/Order asc ,Month/Order asc",
    "$compute": "round(M6048/Value div 1, 2) as Index_Average_M2_2023__100_Value, M6048/Footnote/Text as Index_Average_M2_2023__100_Footnote, M6048/MissingValue/Label as Index_Average_M2_2023__100_MissingValue",
    "$format": "csv",
    "compress": "gz",
}

HEADERS = {
    "ClientId": CLIENT_ID,
    "ClientSecret": CLIENT_SECRET,
    # Optional: set a user agent
    "User-Agent": "python-unctad-client/1.0",
}


def fetch_and_save(url: str, form: dict, headers: dict, out_path: str) -> None:
    """
    POST the form and stream the gzipped CSV to out_path.
    """
    with requests.post(url, data=form, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        # Write streamed content to file
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def load_gz_csv_to_dataframe(gz_path: str) -> pd.DataFrame:
    """
    Read a gzipped CSV into a pandas DataFrame with appropriate dtypes.
    """
    # pandas can read gzip directly
    df = pd.read_csv(gz_path, compression="gzip", header=0, na_values="", encoding="utf-8")
    # Ensure column types similar to R colClasses
    # If columns exist, coerce types
    expected_cols = [
        "Economy/Label",
        "Month/Label",
        "Index_Average_M2_2023__100_Value",
        "Index_Average_M2_2023__100_Footnote",
        "Index_Average_M2_2023__100_MissingValue",
    ]
    # Rename columns to simpler names if desired
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def import_lsci():
    try:
        print("Requesting UNCTAD LSCI data...")
        fetch_and_save(URL, FORM, HEADERS, TEMP_FILE_PATH)
        print(f"Saved compressed CSV to {TEMP_FILE_PATH}")
        df = load_gz_csv_to_dataframe(TEMP_FILE_PATH)
        print("Loaded DataFrame with shape:", df.shape)
        # Show first rows
        print(df.head().to_string(index=False))
        return df
    except requests.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
    except Exception as e:
        print("Error:", e, file=sys.stderr)

lsci_df = import_lsci()[['economy_label', 'month_label', 'index_average_m2_2023__100_value']]
# rename columns
lsci_df.columns = ['country', 'month_label', 'lsci']


# lowercase country and w/trim country and month_label
lsci_df['country'] = lsci_df['country'].str.lower().str.strip()
lsci_df['month_label'] = lsci_df['month_label'].str.lower().str.strip()

# Extract year and month from month_label
lsci_df[['month', 'year']] = lsci_df['month_label'].str.split('. ', expand=True)
lsci_df['year'] = lsci_df['year'].astype(int)
lsci_df['month'] = lsci_df['month'].map({
    mon: i+1 
    for i, mon in enumerate(['jan', 'feb', 'mar', 'apr', 'ma', 'jun',
                             'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
}).astype(int)

# Filter for countries in the analysis
lsci_df = lsci_df[lsci_df['country'].isin(country_name for country_name in country_mapping.keys())]

# Convert to Spark Df
lsci_df = spark.createDataFrame(lsci_df[['country', 'year', 'month', 'lsci']])
lsci_df.repartition(10).cache().count()

lsci_df.write.mode('overwrite').parquet(f'{out_path}/lsci.parquet')