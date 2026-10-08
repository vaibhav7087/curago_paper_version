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
| F1 triage severity | What under-triage rate does a model trained on real ED visits achieve? | CDC NHAMCS 2019 ED (public domain) | `ds/train_triage.py` → `triage.pkl`, `triage_features.json`, `metric_table.csv`, `shap_summary.png` (+ `ds/tune_threshold.py` → tuned operating point) |
| F2 demand forecast | Which forecaster wins on a 6-year pharmacy series? | Kaggle pharma-sales (CC BY-NC 4.0) | `ds/train_forecast.py` → `demand_forecast_comparison.png`, `demand_forecast_metrics.csv` |
| F3 outbreak detection | Can anomaly detection flag district-level case spikes? | incovid19.org district timeseries (2020-04-26..2023-08-22) | `ds/train_outbreak.py` → `outbreak_iso.pkl`, `outbreak_features.json` |

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

## Submission deliverables

| Deliverable | File(s) | Generator |
|---|---|---|
| IEEE paper (6 pp., 13 refs) | `paper/main.tex` · `main.pdf` · `main.md` · `main.docx` | `paper\build.bat`, `paper/export_md.py`, `paper/export_docx.py` |
| Presentation deck (11 slides) | `ppt/paper_presentation.pptx` · `paper_presentation.pdf` | `node ppt/make_deck.js` (+ COM PDF export) |
| IEEE-template deck (23 slides, ICFACT structure) | `ppt/ieee_presentation.pptx` · `ieee_presentation.pdf` | `node ppt/make_ieee_deck.js` (+ COM PDF export) |
| Rubric self-scoring loop | `paper/selfscore_v1..v4.md` (88 → 92 → 98 → 98, review T1–T3 closed, gate passed) | manual, evidence-backed |
| Automated audits | `python paper/audit.py` → ALL PASS | numbers / cites / originality / honesty / format |

Every number in every format resolves from `ds/models/paper/paper_numbers.json`
(seeded, fail-closed pipeline) — nothing is hand-typed.

## Citations

See `docs/citations.md` (dataset licenses, rejected sources, acknowledgments).
