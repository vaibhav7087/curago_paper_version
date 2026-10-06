# Data manifest — download before running any trainer

This repo is **real-data only**. Every trainer fails closed if its raw file is
missing. Raw files are gitignored; derived artifacts in `ds/models/` are committed.

## 1. F1 triage — CDC NHAMCS 2019 ED (US public domain)

| Item | Value |
|---|---|
| Primary URL | `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/ed2019_sas.zip` (~3.5 MB, verified 2026-10-05) |
| Fallback URL | `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/stata/ED2019-stata.zip` (inner file: `ED2019.dta`, verified) |
| Place here | `ds/data/raw/` (unzip; any depth — probed by glob, filename not assumed) |
| Access | HTTPS, no auth |
| Citation | NCHS. National Hospital Ambulatory Medical Care Survey – Emergency Department, 2019. See `docs/citations.md` |

## 2. F2 demand forecast — Kaggle pharma sales

| Item | Value |
|---|---|
| URL | `https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data` |
| File | `salesmonthly.csv` (2014–2019 monthly, ATC columns M01AB…R06) |
| Place here | `ds/data/raw/salesmonthly.csv` |
| Access | **Manual download — Kaggle login required** |
| License | **CC BY-NC 4.0** (Milan Zdravković) — attribution in `docs/citations.md`; non-commercial use only |

## 3. F3 outbreak — machine-readable district timeseries

| Item | Value |
|---|---|
| Preferred URL | `https://data.incovid19.org/csv/latest/districts.csv` (open, no auth) |
| Alternative | any CSV → resample to `district,week,case_count,disease` → `ds/data/raw/district_weekly_cases.csv` |
| Window | Observed in file: **2020-04-26 to 2023-08-22** (retrieved 2026-10-07). Static archive, not live surveillance - cite window + retrieval date |
| Rejected | ProMED-mail / WHO outbreak PDFs — no tabular machine-readable download; transcription risk (see `docs/citations.md`) |

## Provenance

Each `ds/models/*.csv` artifact carries a `Data Source` column matching this
manifest. If the column and this file disagree, the artifact is stale —
regenerate with the commands in the repo README.
