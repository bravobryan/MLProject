# # Import Federal Reserve Bank of St. Louis (FRED) data
# - Author: Bryan Bravo
# - Created: 2026-03-02
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
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'FRED_API_KEY'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()

#######################################################################################


### Variables

# fred_api = hardcoded_keys.FRED_API_KEY
fred_api = args['FRED_API_KEY']
end_date = (dt.now() - relativedelta(days=1)).strftime("%Y-%m-%d")

out_path = 's3a://ml-project-s3-bronze/input_folder/'

### Custom Functions

def fetch_fred_data(
    api_key: str,
    country_name: str, # country name of series id.
    series_id: str,  # FRED series ID
    rate_name: str = None,
    start_date: str = "2006-04-01",  # Chosen because the broad fx rate for the US begins in 2006
    end_date: str = (dt.now().replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
    ) -> pd.DataFrame:
    """
    Fetches daily foreign exchange rate data from the FRED API, cleans it, and returns
    a standardized DataFrame containing the latest revision for each date.

    This function:
    - Adds a `country` column for identification
    - Computes a unified USD exchange rate (`us_fx_rate`) so that:
        • If the series is FX-per-USD (e.g., DEXJPUS), it inverts the value
        • Otherwise, it uses the value as-is

    Parameters
    ----------
    api_key : str
        FRED API key used for authentication.
    country_name : str
        Human-readable country name to attach to the output (e.g., "euro", "yen").
    series_id : str
        FRED series ID (e.g., "DEXUSEU", "DEXJPUS").
    rate_name : str
        If API value is based on interest rates, this will create a column name matching the string value.
    start_date : str
        Start date for the query in YYYY-MM-DD format. (Defaults to "2006-01-01")
    end_date : str
        End date for the query in YYYY-MM-DD format. (Defaults to the last day of the previous month)

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - date : datetime64
        - country : str
        - `rate_name` <- value derived from variable given : float
        representing the latest available FX rate for each date.

    Raises
    ------
    Exception
        If the API request fails or the response cannot be parsed.

    """

    try:
        response = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&realtime_start={start_date}&realtime_end={end_date}&api_key={api_key}&file_type=json",
            timeout=10)
        fred_data = response.json()

        if 'error_code' in fred_data:  # check if response is None
            print(f"✗ FRED API error: {fred_data.get('error_code', 'error_message')}")

        # Create pandas DataFrame from observations
        df = pd.DataFrame(fred_data['observations'])

        # Select relevant columns
        df = df[['date', 'realtime_start', 'value']]
        df['country'] = country_name


        # Correct data types and add joining variables
        df['date'] = pd.to_datetime(df['date'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert to numeric, coerce errors to NaN

        df['join_dt'] = df['date'].dt.strftime("%Y%m").astype(int)
        # Keep only the most recent revision for each date
        df = df.loc[df.groupby('date')['realtime_start'].idxmax()]
        df = df[df['date'] >= start_date]

        # Add USD exchange rate (USD/FX)
        if rate_name == 'fx_rate':
            if series_id[5:] == "US":
                df['fx_rate'] = 1 / df['value']
            else:
                df['fx_rate'] = df['value']

            # subset columns
            df = df[['date', 'country', 'fx_rate', 'join_dt']].reset_index(drop=True)

        # Add column with identified rate name
        else:
            df[rate_name] = df['value']
            # subset columns
            df = df[['date', 'country', rate_name, 'join_dt']].reset_index(drop=True)
        

        print(f"""✓ Fetched {len(df)} records for '{country_name}': {rate_name}
              from FRED for years {dt.strftime(df['date'].min(), '%Y-%m-%d')} through {dt.strftime(df['date'].max(), '%Y-%m-%d')}
              ******************************************************************************""")
        return df
    except Exception as e:
        print(f"✗ Error fetching FRED data on {country_name}: {str(e)[:100]}")

## Query

fred_fx_series = {
    'australia': {
        'interest_rate': 'IRSTCI01AUM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Australia [IRSTCI01AUM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01AUM156N, March 14, 2026.
        'fx_rate': 'DEXUSAL'  # Board of Governors of the Federal Reserve System (US), U.S. Dollars to Australian Dollar Spot Exchange Rate [DEXUSAL], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXUSAL, March 14, 2026.
    },
    'brazil': {
        'interest_rate': 'IRSTCI01BRM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Brazil [IRSTCI01BRM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01BRM156N, March 14, 2026.
        'fx_rate': 'DEXBZUS'  # Board of Governors of the Federal Reserve System (US), Brazilian Reals to U.S. Dollar Spot Exchange Rate [DEXBZUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXBZUS, March 14, 2026.
    },
    'canada': {
        'interest_rate': 'IRSTCI01CAM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Canada [IRSTCI01CAM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01CAM156N, March 14, 2026.
        'fx_rate': 'DEXCAUS'  # Board of Governors of the Federal Reserve System (US), Canadian Dollars to U.S. Dollar Spot Exchange Rate [DEXCAUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXCAUS, March 14, 2026.
    },
    'china': {
        'interest_rate': 'IRSTCI01CNM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for China [IRSTCI01CNM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01CNM156N, March 14, 2026.
        'fx_rate': 'DEXCHUS'  # Board of Governors of the Federal Reserve System (US), Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate [DEXCHUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXCHUS, March 14, 2026.
    },
    # 'euro': {
    #     'interest_rate': 'IRSTCI01EZM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Euro Area (19 Countries) [IRSTCI01EZM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01EZM156N, March 14, 2026.
    #     'fx_rate': 'DEXUSEU'  # Board of Governors of the Federal Reserve System (US), U.S. Dollars to Euro Spot Exchange Rate [DEXUSEU], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXUSEU, March 14, 2026.
    # },
    'france': {
        'interest_rate': 'IRSTCI01FRM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for France [IRSTCI01FRM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01FRM156N, March 14, 2026.
        'fx_rate': 'DEXSZUS'  # Board of Governors of the Federal Reserve System (US), Swiss Francs to U.S. Dollar Spot Exchange Rate [DEXSZUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXSZUS, March 14, 2026.
    },
    'germany': {
        'interest_rate': 'IRSTCI01DEM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Germany [IRSTCI01DEM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01DEM156N, March 14, 2026.
        'fx_rate': 'CCUSSP01DEM650N'  # Organization for Economic Co-operation and Development, Currency Conversions: US Dollar Exchange Rate: Spot, End of Period: USD: National Currency for Germany [CCUSSP01DEM650N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CCUSSP01DEM650N, March 14, 2026.
    },
    'india': {
        'interest_rate': 'IRSTCI01INM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for India [IRSTCI01INM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01INM156N, March 14, 2026.
        'fx_rate': 'DEXINUS'  # Board of Governors of the Federal Reserve System (US), Indian Rupees to U.S. Dollar Spot Exchange Rate [DEXINUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXINUS, March 14, 2026.
    },
    'italy': {
        'interest_rate': 'IRSTCI01ITM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Italy [IRSTCI01ITM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01ITM156N, March 14, 2026.
        'fx_rate': 'CCUSSP01ITM650N'  # Organization for Economic Co-operation and Development, Currency Conversions: US Dollar Exchange Rate: Spot, End of Period: USD: National Currency for Italy [CCUSSP01ITM650N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CCUSSP01ITM650N, March 15, 2026.
    },
    'japan': {
        'interest_rate': 'IRSTCI01JPM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Japan [IRSTCI01JPM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01JPM156N, March 14, 2026.
        'fx_rate': 'DEXJPUS'  # Board of Governors of the Federal Reserve System (US), Japanese Yen to U.S. Dollar Spot Exchange Rate [DEXJPUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXJPUS, March 14, 2026.
    },
    'mexico': {
        'interest_rate': 'IRSTCI01MXM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Mexico [IRSTCI01MXM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01MXM156N, March 15, 2026.
        'fx_rate': 'DEXMXUS'  # Board of Governors of the Federal Reserve System (US), Mexican Pesos to U.S. Dollar Spot Exchange Rate [DEXMXUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXMXUS, March 15, 2026.
    },
    'south_korea': {
        'interest_rate': 'IRSTCI01KRM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Korea [IRSTCI01KRM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01KRM156N, March 15, 2026.
        'fx_rate': 'DEXKOUS'  # Board of Governors of the Federal Reserve System (US), South Korean Won to U.S. Dollar Spot Exchange Rate [DEXKOUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXKOUS, March 15, 2026.
    },
    'russia': {
        'interest_rate': 'IRSTCI01RUM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Russia [IRSTCI01RUM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01RUM156N, March 15, 2026.
        'fx_rate': 'CCUSSP02RUM650N'  # Organization for Economic Co-operation and Development, Currency Conversions: US Dollar Exchange Rate: Spot, End of Period: National Currency: USD for Russia [CCUSSP02RUM650N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CCUSSP02RUM650N, March 15, 2026.
    },
    'south_africa': {
        'interest_rate': 'IRSTCI01ZAM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for South Africa [IRSTCI01ZAM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01ZAM156N, March 15, 2026.
        'fx_rate': 'DEXSFUS'  # Board of Governors of the Federal Reserve System (US), South African Rand to U.S. Dollar Spot Exchange Rate [DEXSFUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXSFUS, March 15, 2026.
    },
    'turkiye': {
        'interest_rate': 'IRSTCI01TRM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for Turkey [IRSTCI01TRM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01TRM156N, March 15, 2026.
        'fx_rate': 'CCUSSP01TRM650N'  # Organization for Economic Co-operation and Development, Currency Conversions: US Dollar Exchange Rate: Spot, End of Period: USD: National Currency for Turkey [CCUSSP01TRM650N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CCUSSP01TRM650N, March 15, 2026.
    },
    'united_kingdom': {
        'interest_rate': 'IRSTCI01GBM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for United Kingdom [IRSTCI01GBM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01GBM156N, March 15, 2026.
        'fx_rate': 'DEXUSUK'  # Board of Governors of the Federal Reserve System (US), U.S. Dollars to U.K. Pound Sterling Spot Exchange Rate [DEXUSUK], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXUSUK, March 15, 2026.
    },
    'united_states': {
        'interest_rate': 'IRSTCI01USM156N',  # Organization for Economic Co-operation and Development, Interest Rates: Immediate Rates (< 24 Hours): Call Money/Interbank Rate: Total for United States [IRSTCI01USM156N], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IRSTCI01USM156N, March 15, 2026.
        'fx_rate': 'DTWEXBGS'   # Board of Governors of the Federal Reserve System (US), Nominal Broad U.S. Dollar Index [DTWEXBGS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DTWEXBGS, March 15, 2026.
    }
}

fred_df = pd.DataFrame([[]])
for country, kval in fred_fx_series.items():
    

    for i, (rate_name, series) in enumerate(kval.items()):
        if i == 0:
            df1 = fetch_fred_data(fred_api, country, series, rate_name, end_date=end_date)
        else:
            df2 = fetch_fred_data(fred_api, country, series, rate_name, end_date=end_date)
            df1 = (df1.drop('date', axis=1)
                   .merge(df2, how='outer', on=['join_dt', 'country']))

        fred_df = pd.concat([fred_df, df1], ignore_index=True)

# Convert to PySpark df
fred_df = (
    spark.createDataFrame(fred_df.drop('join_dt', axis=1))
    .withColumn('date', F.date_format(F.col('date'), "yyyyMMdd").cast('int'))
    .dropna())  # Drop all null values.
fred_df.cache().count()

## Write to S3 and Spark Stop
fred_df.write.mode('overwrite').parquet(f"{out_path}/fred.parquet")
spark.stop()