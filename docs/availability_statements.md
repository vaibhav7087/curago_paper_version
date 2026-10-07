# Data and Code Availability Statements

This document provides formal availability statements, licensing terms, provenance records, and exact step-by-step reproduction instructions for the research paper *"Real-Data Machine Learning for Rural Telemedicine Triage and Outbreak Surveillance"*.

---

## 1. Formal IEEE Statements

### Data Availability Statement
> The datasets analyzed in this study are publicly accessible through official research repositories and open-access scientific archives. Patient triage evaluations utilize the 2019 Emergency Department dataset from the National Hospital Ambulatory Medical Care Survey (NHAMCS), published by the CDC National Center for Health Statistics (US Public Domain; available at https://www.cdc.gov/nchs/nhamcs/). District-level outbreak surveillance evaluates the open-access COVID-19 India district time-series static archive (available at https://data.incovid19.org/). Supplementary pharmaceutical supply forecasts evaluate the Kaggle Pharma Sales point-of-sale dataset under the Creative Commons Attribution-NonCommercial 4.0 International license (available at https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data). No private, proprietary, or synthetic patient data were used in this work.

### Code Availability Statement
> All source code, data ingestion scripts, model training routines, evaluation protocols, and artifact generation pipelines are publicly available under the open-source MIT License at the project repository: https://github.com/vaibhav7087/curago_paper_version. The entire data science pipeline executes deterministically from raw data to generate publication figures and IEEE LaTeX tables without manual intervention.

---

## 2. Dataset Provenance and Licensing Matrix

| Dataset | Module / Function | Primary File | Dimensions / Records | Source & Access URL | License / Terms |
|---|---|---|---|---|---|
| **CDC NHAMCS 2019 ED** | Module 1: Patient Triage & Severity | `ed2019_sas.sas7bdat` | 19,481 raw encounters (13,595 analysis cohort, 911 features) | [CDC NCHS Archive](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/ed2019_sas.zip) | US Government Work (Public Domain) |
| **COVID-19 India District Time-Series** | Module 2: Outbreak Surveillance | `district_weekly_cases.csv` | 126,712 district-weeks, 840 state-qualified districts | [incovid19 Static Archive](https://data.incovid19.org/) | Open Research Data (Volunteer Archive) |
| **Pharma Sales Point-of-Sale Data** | Supplementary: Medicine Demand Forecast | `salesmonthly.csv` | 70 monthly rows (69 analyzed, Oct 2019 partial excluded), 8 ATC groups | [Kaggle Dataset](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data) | CC BY-NC 4.0 (Attribution-NonCommercial) |

### Strict Fail-Closed Policy
To prevent accidental data leakage or the use of mock data, the repository enforces a strict **fail-closed** architecture:
- If any raw data file is missing from `ds/data/raw/`, all training and generation scripts immediately abort with exit code `1` and raise a descriptive `FileNotFoundError` providing direct download URLs.
- Synthetic, simulated, or mocked data generation is strictly prohibited across all pipelines.

---

## 3. Environment Specification and Dependencies

The pipeline has been developed and validated on Python 3.11 and 3.12 across Windows and Linux environments.

### Core Libraries:
- `pandas` >= 2.0.0 (tabular manipulation)
- `numpy` >= 1.24.0 (numerical operations)
- `scikit-learn` >= 1.3.0 (modeling, cross-validation, metrics)
- `xgboost` >= 2.0.0 (gradient-boosted classification)
- `lightgbm` >= 4.0.0 (gradient-boosted time series)
- `statsmodels` >= 0.14.0 (Holt-Winters exponential smoothing)
- `shap` >= 0.43.0 (TreeExplainer feature attribution)
- `pyreadstat` >= 1.2.0 (SAS .sas7bdat parser)
- `matplotlib` >= 3.8.0 (300 DPI publication plotting)

---

## 4. End-to-End Exact Reproduction Commands

All commands are executed from the repository root directory. The pipeline is fully deterministic (fixed random seed = 42).

### Step 1: Raw Data Acquisition
Place the three downloaded raw files into `ds/data/raw/`:
- `ds/data/raw/ed2019_sas.sas7bdat` (unzipped from CDC archive)
- `ds/data/raw/district_weekly_cases.csv` (downloaded or aggregated from incovid19)
- `ds/data/raw/salesmonthly.csv` (downloaded from Kaggle)

### Step 2: Model Training & Threshold Optimization
```bash
# 1. Train triage models (Logistic Regression + XGBoost, 5-fold CV)
python ds/train_triage.py
# Outputs: ds/models/triage.pkl, ds/models/metric_table.csv, ds/data/processed/nhamcs_triage_features.csv

# 2. Tune operating point for safety (sweep thresholds for recall >= 0.80)
python ds/tune_threshold.py
# Outputs: ds/models/operating_point.json, ds/models/threshold_tradeoff.csv
```

### Step 3: Conformal Triage & Uncertainty Quantification
```bash
# 3. Run split-conformal calibration across 200 random splits
python ds/conformal_triage.py
# Outputs: ds/models/conformal_repeats.csv, ds/models/conformal_alpha_sweep.csv, ds/models/conformal_recall_hist.png

# 4. Compute bootstrap confidence intervals, Brier score, and NEWS2 baselines
python ds/severity_eval_extras.py
# Outputs: ds/models/severity_extras.json, ds/models/calibration_curve.png
```

### Step 4: Population Outbreak Surveillance & Demand Forecasting
```bash
# 5. Evaluate retrospective IsolationForest on Delta wave window
python ds/evaluate_outbreak.py
# Outputs: ds/models/outbreak_delta_eval.json, ds/models/outbreak_flagged_weeks.csv

# 6. Multi-method outbreak evaluation (Single vs Confirmed vs Z-score)
python ds/outbreak_eval_v2.py
# Outputs: ds/models/outbreak_eval_v2.json, ds/models/outbreak_methods_compare.png

# 7. Train supplementary pharmaceutical demand forecasts (12-step holdout)
python ds/train_forecast.py
# Outputs: ds/models/demand_forecast_metrics.csv
```

### Step 5: Publication Artifact Generation
```bash
# 8. Generate 300-DPI IEEE figures (Fig 1 - Fig 6, Fig S1, Fig S1b)
python ds/paper_figures.py
# Outputs: ds/models/paper/paper_fig0..fig5.png, ds/models/paper/paper_figS1.png, ds/models/paper/paper_figS1b.png

# 9. Generate IEEE LaTeX booktabs tables, CSVs, and verified numbers
python ds/paper_tables.py
# Outputs: ds/models/paper/paper_T1..T7.csv/tex, ds/models/paper/paper_TS1_forecast.csv/tex,
#          ds/models/paper/paper_numbers.json, docs/captions.md
```

---

## 5. Automated Integrity Checks
Every artifact script (`paper_figures.py`, `paper_tables.py`) contains built-in assertions checking:
1. Exact row counts and cohort filters (13,595 analysis encounters, 126,712 district-weeks, 69 months).
2. Bit-exact numerical equality against source model outputs.
3. Matching column headers and formatting between CSV and LaTeX outputs.
4. Total absence of synthetic mock fallbacks.
