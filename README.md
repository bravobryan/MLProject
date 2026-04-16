# MLProject — Random Forest Imputation

Tech Stack:
- Python (pandas, scikit-learn)
- PySpark (for demonstrating distributed data processing)
- Jupyter Notebooks
- NumPy, SciPy
- scikit-learn's GridSearchCV
- Git + GitHub
- Kaggle (for dataset distribution)

## Repository Structure:
```
ml_missing_value_impute/
    hardcoded_keys.py (not stored in repo)
    proj_vars.py
    import_datasets/        (not stored in repo; private API data)
        australiancpi.csv
        US_LSCI_M.csv
    notebooks/
        import_data/         # run these first (API import notebooks)
            import_acled.ipynb
            import_eia.ipynb
            import_fred.ipynb
            import_gpr_index.ipynb
            import_imf.ipynb
            import_unctad.ipynb
            import_wb.ipynb
        transform/           # run after import_data; impute_missing_data.ipynb last
            impute_missing_data.ipynb
            joined_input.ipynb
    processed_datasets/     # produced outputs (to be uploaded to Kaggle)
        acled.csv
        cpi.csv
        final_df.csv
        fred.csv
        gpr.csv
        joined_input.csv
        lsci.csv
        oil.csv
        wb.csv
```

## Executive Summary

- **Business Question:** 
    - Can these indicators (oil prices, LSCI, GPR, events, CPI, rates, FX) forecast near-term changes in a country's FX reserves for treasury/planning decisions?

- **Problem and Context:** 
    - For businesses using industry analysis and financial forecasting to inform strategy, lagging or incomplete indicator data can degrade decision quality. Using imputation and predictive models helps fill gaps and enables more data-focused, timely strategy and positioning decisions.

- **Summary of the Machine Learning process:** 
    - Data is gathered from multiple APIs and sources (some monthly, some daily) using the notebooks in `import_data`. Data cleaning and transformations are performed in the `transform` notebooks. Missing values are handled via targeted Random Forest Regressor models: each country's missing-value imputation model is trained with the most relevant features for that country. Models were tuned using `GridSearchCV` and implemented with both pandas (local) and PySpark (distributed) workflows as a demonstration of versatility.

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

