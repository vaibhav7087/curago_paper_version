# Curago — Paper Version (Data Science evidence)

Real-data-only reproduction of the Curago DS layer for the Nexathon II
**Paper Presentation** track. Every trainer here **fails closed**: no synthetic
data, no simulated outbreaks, no fallback series. If a raw file is missing you
get a hard error with the exact download URL.

The full-stack demo (voice intake → triage → doctor dashboard → delivery) lives
in the separate **demo repository** submitted for the Project Competition track.
This repo contains only the evidence: data pipeline, training scripts,
notebooks, committed artifacts, and citations.

## Modules

| Module | Question | Raw data | Trainer → artifacts |
|---|---|---|---|
| F1 triage severity | Can a model trained on real ED visits reduce under-triage? | CDC NHAMCS 2019 ED (public domain) | `ds/train_triage.py` → `triage.pkl`, `triage_features.json`, `metric_table.csv`, `shap_summary.png` |
| F2 demand forecast | Which forecaster wins on a 6-year pharmacy series? | Kaggle pharma-sales (CC BY-NC 4.0) | `ds/train_forecast.py` → `demand_forecast_comparison.png`, `demand_forecast_metrics.csv` |
| F3 outbreak detection | Can anomaly detection flag district-level case spikes? | incovid19.org district timeseries (frozen 2020–2021) | `ds/train_outbreak.py` → `outbreak_iso.pkl`, `outbreak_features.json` |

## Reproduce everything

```bash
# 1. env
python -m venv .venv && .venv\Scripts\activate      # or: source .venv/bin/activate
pip install -r ds/requirements.txt

# 2. raw data — follow ds/data/README.md (URLs + placement)

# 3. train (each fails closed with instructions if raw data missing)
python ds/train_triage.py
python ds/train_forecast.py
python ds/train_outbreak.py

# 4. notebooks (same pipeline, rendered narrative for the paper)
jupyter notebook ds/notebooks/
```

## Integrity rules

- `ds/data/raw/` is gitignored; `ds/models/` artifacts are committed.
- Every metric CSV carries a `Data Source` column; it must match
  `ds/data/README.md`. Stale artifacts → regenerate.
- No ProMED / WHO-PDF ingestion anywhere (see `docs/citations.md` for why).
- Inference code (`ds/predict.py`) is pure local sklearn — no network calls.

## Citations

See `docs/citations.md` (dataset licenses, rejected sources, acknowledgments).
