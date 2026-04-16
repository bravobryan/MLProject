import os
import sys

# Set JAVA_HOME before importing PySpark and use findspark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-22'  # May need to remove or update in cloud environment.
import findspark
findspark.init()

import requests
import pandas as pd
import numpy as np
import json
import pyspark
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

# api keys and other hardcoded values for the Strait of Hormuz Research project.
# Note: In a production environment, these should be stored securely and not hardcoded.
import hardcoded_keys # Note: This file is added to .gitignore to prevent accidental commits of sensitive information.