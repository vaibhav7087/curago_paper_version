# Verified results (regenerated 2026-10-07, deterministic — see QA note)

All numbers below are read straight from the committed artifacts in `ds/models/`.
Rerunning any trainer on the raw data from `ds/data/README.md` reproduces them
exactly (verified: F2/F3 rerun → zero `git diff`; F1 rerun → identical CSVs).

## F1 — triage severity (`metric_table.csv`, CDC NHAMCS 2019 ED, n=13,595)

| Model | AUC-ROC | Recall | Precision | F1 | Under-triage |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.719 | 0.609 | 0.269 | 0.373 | 0.391 |
| XGBoost (class-weighted) | 0.732 | 0.513 | 0.334 | 0.405 | 0.487 |

Reading: XGBoost wins on AUC/F1, but **under-triages more** (0.487 vs 0.391) —
reported as-is. A recall-tuned operating threshold is the natural next step and
is explicitly future work, not a claimed result.

## F2 — demand forecast (`demand_forecast_metrics.csv`, 12-step holdout MAE)

| Best model per drug | Drugs |
|---|---|
| Seasonal naive (3/8) | M01AB, M01AE, N02BE |
| Holt-Winters (3/8) | N05C, R03, R06 |
| LightGBM (2/8) | N02BA, N05B |

Correction: an earlier commit message said "naive wins 4/8" — the artifact shows
**3/8** (N05B goes to LightGBM, 50.54 < 50.78). The CSV was always correct; this
file is the corrected record. Finding: a naive baseline is competitive — worth
stating in the paper, not hiding.

## F3 — outbreak detection (IsolationForest, contamination=0.05)

- Input: 126,712 district-weeks, 840 state-qualified districts, 2020-04-26–2023-08-22.
- Flagged: 6,336/126,712 (5.0% — by construction of the contamination parameter).
- Interpretation for the paper: retrospective spike flagging on a static archive,
  not live surveillance. Precision/recall against labeled outbreak events is not
  claimed — no labeled ground truth was available.

## QA note (2026-10-07)

- Fail-closed verified: hiding `ds/data/raw/` makes all three trainers exit 1
  with download instructions.
- Determinism verified: rerunning F2/F3 produces zero `git diff` vs committed artifacts.
- License chain verified: Kaggle page (CC BY-NC 4.0) → mirror README restates it →
  `docs/citations.md` attributes → metric CSVs carry the license tag.
