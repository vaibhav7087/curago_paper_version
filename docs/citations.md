# Citations and data licenses

## Datasets

1. **CDC NHAMCS 2019 Emergency Department** (F1 triage model)
   - National Center for Health Statistics, National Hospital Ambulatory Medical
     Care Survey – Emergency Department, 2019.
   - Documentation: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/doc19-ed-508.pdf`
   - Data page: `https://www.cdc.gov/nchs/nhamcs/documentation/about-the-data-2019.html`
   - License: US Government work — public domain. No attribution restriction;
     citation above is scholarly practice.

2. **Pharma sales dataset** (F2 demand forecast)
   - Milan Zdravković, "Pharma sales data", Kaggle,
     `https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data`
   - Six years (2014–2019) of point-of-sale transactional data resampled to
     monthly aggregates, 8 ATC categories.
   - License: **Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** —
     `https://creativecommons.org/licenses/by-nc/4.0/`
   - Obligations met by this citation; non-commercial research/hackathon use is
     within the NC clause. Do not redistribute commercially.

3. **COVID-19 India district timeseries** (F3 outbreak detection)
   - covid19india.com volunteer project, `https://data.incovid19.org/`
   - License: data freely usable (project licence on covid19india.com);
     static archive, observed series **2020-04-26 to 2023-08-22** (retrieved 2026-10-07).
   - Must cite window + retrieval date; results are retrospective detection
     on a static archive, not live surveillance.

## Rejected sources (with rationale)

- **ProMED-mail**: narrative email/HTML alert network — no tabular
  machine-readable download; community scrapers depend on fragile AJAX/HTML
  parsing (ToS-gray, venue-Wi-Fi-hostile). Peer-reviewed corroboration:
  *OpenDengue* (Scientific Data, 2024) notes ProMED "does not have tabular
  machine readable download options and requires substantial manual processing."
- **WHO outbreak PDFs**: require table scraping + hand transcription —
  transcription error is a fabrication risk under competition integrity rules.

## Models and tools (acknowledged, not authored)

- scikit-learn, XGBoost, LightGBM, statsmodels (Holt-Winters), SHAP, pyreadstat.
- Our contribution: dataset preparation, feature engineering, training/eval
  protocol, under-triage metric framing, and integration into the Curago
  telemedicine workflow. All metrics in `ds/models/` are reproducible from the
  commands in the README.
