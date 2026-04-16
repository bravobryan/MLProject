# MLProject — Random Forest Imputation
_(In Progress)_

Tech Stack:
- Python (pandas, NumPy, scikit-learn)
- PySpark (for demonstrating distributed data processing)
- Jupyter Notebooks
- Git + GitHub
- Kaggle (for dataset distribution)

## Repository Structure
```
ml_missing_value_impute/
├─ hardcoded_keys.py         # not stored in repo (private API data)
├─ proj_vars.py
├─ import_datasets/          # local raw files (not stored in repo)
│  ├─ australiancpi.csv
│  └─ US_LSCI_M.csv
├─ notebooks/
│  ├─ import_data/           # run these first (API import notebooks)
│  │  ├─ import_acled.ipynb
│  │  ├─ import_eia.ipynb
│  │  ├─ import_fred.ipynb
│  │  ├─ import_gpr_index.ipynb
│  │  ├─ import_imf.ipynb
│  │  ├─ import_unctad.ipynb
│  │  └─ import_wb.ipynb
│  └─ transform/             # run after import_data; impute_missing_data.ipynb is last
│     ├─ impute_missing_data.ipynb
│     └─ joined_input.ipynb
└─ processed_datasets/       # produced outputs (to be uploaded to Kaggle)
   ├─ acled.csv
   ├─ cpi.csv
   ├─ final_df.csv           # Final dataset
   ├─ fred.csv
   ├─ gpr.csv
   ├─ joined_input.csv
   ├─ lsci.csv
   ├─ oil.csv
   └─ wb.csv
```

# Executive Summary

### Business Question: 
#####  Can these indicators (oil prices, LSCI, GPR, events, CPI, rates, FX) forecast near-term changes in a country's FX reserves for treasury/planning decisions?

### Problem and Context: 
For businesses using industry analysis and financial forecasting to inform strategy, lagging or incomplete indicator data can degrade decision quality. In today’s highly competitive economic landscape, businesses need to make quick, data-driven decisions to remain relevant. 

Using imputation and predictive models helps fill gaps and enables more data-focused, timely strategy and positioning decisions. The complexity of operating in a global economic environment can make it challenging to determine how to position a strategy across major economies.

## Summary of the Machine Learning process:
### Sourced Data
Data is gathered from multiple APIs and sources (some monthly, some daily) using the notebooks in `import_data`. 

Sources:
- ACLED: This includes all global battles, explosions/remote violence, and violence against civilians events.

Inspected missing brent_dollars_per_barrel and wti_dollars_per_barrel (occurred on market/non-trading days) and applied forward-fill per country using a window (W.partitionBy('country').orderBy('date') + F.last(..., ignorenulls=True)).
ACLED (events):

Treated missing events as 0 (F.when(...).otherwise(0)).
Since acled_df is monthly vs daily base df, kept event counts only on the last day of each month per country (row_number over partition → keep last_events==1, set others to 0).
CPI:

For missing Australian CPI, imported import_datasets/australiancpi.csv, parsed quarter months, joined it in and coalesced into cpi.
Applied a special correction for early 2006 (cpi = 84.5 for a specific Australia window), then forward-filled cpi by country ordered by year, month using F.last(..., ignorenulls=True).

### Data Preparation
Data cleaning and transformations are done in the `transform` notebooks (primary work in `joined_input.ipynb`; some one-hot encoding in `impute_missing_data.ipynb`).

Most missing-value handling is applied during dataset joins. Condensed, GitHub-friendly summary:

- **FRED + Oil**
    - Left-joined on `date`; duplicates checked.
    - Missing `brent_dollars_per_barrel` and `wti_dollars_per_barrel` (market/non-trading days) are forward-filled per country with a window: `W.partitionBy('country').orderBy('date')` + `F.last(..., ignorenulls=True)`.
    - Outlier detection: `detect_outliers_by_partition` on `interest_rate`, `fx_rate`, `brent_dollars_per_barrel`, `wti_dollars_per_barrel` (partitioned by `country`, `year`).

- **ACLED (`events`)**
    - Joined on `['year','month','country']`.
    - Missing `events` interpreted as `0` (assumed no recorded events).
    - Because `acled_df` is monthly vs a daily base, `events` are kept only on the last day of each month per country (row_number over partition → keep last), other days set to `0`.
    - Outlier detection applied to `events` to highlight notable political shocks.

- **CPI**
    - Renamed source `value` → `cpi` and joined on `['year','month','country']`.
    - For missing Australian CPI, imported `import_datasets/australiancpi.csv` (quarterly) and expanded/coalesced values into monthly `cpi`.
    - Applied a small early-2006 correction for Australia and forward-filled `cpi` per country (ordered by `year, month`).

- **GPR**
    - Joined on `['year','month','country']`; no additional missingness required after join.

- **LSCI**
    - Joined on `['year','month','country']`.
    - Reporting frequency changed (quarterly → monthly); forward-fill used per country (`F.last(..., ignorenulls=True)` over `W.partitionBy('country').orderBy('year','month')`).
    - Applied targeted 2006 fixes and seeded missing March/April 2006 values for Australia using a lookup (`lsci_dict`).
    - Removed inconsistent early data (filtered out dates on or before `2006-04-30`).

Notes:
- Forward-fill is used where values are expected to hold until the next report (e.g., index benchmarks, quarterly-to-monthly conversions).
- Larger temporal gaps (notably `fx_reserves`) are handled with ML imputation in `impute_missing_data.ipynb` rather than simple propagation.



- **Limitations of the Analysis:** 
    - Feature relevance varies by country so results are heterogeneous; model performance depends on historical data quality and coverage. Temporal mismatches (monthly vs daily) require aggregation/aligning choices that can introduce bias. Imputation may not capture structural breaks or regime changes.

- **Summary of the Analysis results:** 
    - Country-level Random Forest models successfully reduced missingness and produced plausible imputations; tuning via GridSearchCV improved predictive performance compared to default hyperparameters. Results vary by country and feature set.

- **Recommended course of action:** 
    - Use imputed series for exploratory forecasting and treasury planning with caveats; retrain and retune models periodically and validate imputed values against new observed data when available. For production use, adopt a monitoring process to detect drift and re-evaluate feature sets per country.

- **Benefit of the Analysis:** 
    - Provides a practical way to fill gaps in indicator series, enabling more complete datasets for forecasting, planning, and scenario analysis—leading to faster, more informed decisions.

## Machine Learning Skills Demonstrated:
- Data ingestion from multiple APIs and CSV sources; handling authentication-sensitive code outside the repo (`import_datasets/` and `hardcoded_keys.py` not stored)
- Time-series alignment and aggregation (handling monthly vs daily frequencies)
- Feature engineering per-country to improve model relevance
- Model building with Random Forest Regressor for imputation tasks
- Hyperparameter tuning with `GridSearchCV` for each RF model
- Use of both pandas and PySpark pipelines to demonstrate local and distributed processing
- Notebook-driven reproducibility and modular notebook ordering (import → transform → impute)
- Preparing and exporting processed datasets for publishing (Kaggle)

Notes
- The `import_datasets` folder and `hardcoded_keys.py` are intentionally not stored in the public repo to protect credentials and private data access.
- Run order: execute all notebooks under `notebooks/import_data/` first (these collect and save raw inputs), then run the notebooks under `notebooks/transform/`, leaving `notebooks/transform/impute_missing_data.ipynb` as the final notebook to run.
- All produced datasets in `processed_datasets/` are intended to be uploaded to Kaggle for public reuse.

