# Verified results (regenerated 2026-10-08, deterministic — see QA note)

All numbers below are read straight from the committed artifacts in `ds/models/`.
Rerunning any trainer on the raw data from `ds/data/README.md` reproduces them
exactly (verified: F1/F2/F3 reruns → identical CSVs/pkls; SHAP PNG has minor
render jitter across runs, data identical — original rendering kept).

## F1 — triage severity (CDC NHAMCS 2019 ED, n=13,595)

Baseline at default 0.5 threshold (`metric_table.csv`, 5-fold CV — no held-out
test set; CV means reported). Primary model is fixed by a pre-declared rule:
lowest OOF Brier among the calibrated candidates (ties → isotonic):

| Model | AUC-ROC | Recall | Precision | F1 | Under-triage | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.719 | 0.609 | 0.269 | 0.373 | 0.391 | 0.209 |
| XGBoost (class-weighted) | 0.732 | 0.513 | 0.334 | 0.405 | 0.487 | 0.165 |
| XGBoost (unweighted) | 0.743 | 0.188 | 0.571 | 0.283 | 0.812 | 0.113 |
| **XGBoost (unweighted + isotonic) — primary** | 0.742 | 0.141 | 0.606 | 0.227 | 0.859 | **0.113** |
| XGBoost (unweighted + Platt) | 0.743 | 0.158 | 0.593 | 0.249 | 0.842 | 0.113 |
| Constant (train prevalence) | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.129 |

AUC 95% CI (paired bootstrap, 1000 resamples): primary [0.728, 0.751],
LogReg [0.708, 0.731], difference [0.012, 0.031] (excludes zero).
Calibrated Brier (0.113) beats the constant reference (0.129); class weighting
distorts probabilities (0.165).

Tuned operating point (`operating_point.json` + `threshold_tradeoff.csv`,
rule: maximum precision among thresholds with OOF recall ≥ 0.80, ties →
higher threshold; AUC unchanged — tuning moves along the ROC curve):

| Model | Threshold | Recall | Precision | FPR | Under-triage | Abstention |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.40 | 0.811 | 0.214 | — | 0.189 | 0.394 |
| XGBoost primary (pooled) | 0.11 | 0.800 | 0.226 | 0.493 | 0.200 | 0.042 |
| XGBoost primary (**nested cross-fitted**) | 0.09–0.10 | **0.829** | 0.214 | 0.545 | **0.172** | 0.042 |

The nested row is the conservative headline: each outer fold re-selects its
threshold on the other four folds only, then evaluates on its own test fold —
no test-fold peeking. Reading: tuning cuts under-triage 0.859 → 0.172 at the
cost of FPR (0.016 → 0.545) and precision (0.606 → 0.214). The abstain rule
(`max(proba) < 0.60`, i.e. calibrated probability in (0.4, 0.6)) is unchanged;
4.2% of cases abstain at the tuned point (39.4% for LogReg). This is the
standard sensitivity-first tradeoff of triage, not metric inflation.

Top RFV drivers decoded in `docs/rfv_codebook.md` (official NCHS labels).

## F2 — demand forecast (12-step holdout MAE, single holdout — no rolling CV)

69 usable months (Jan 2014 – Sep 2019; Oct 2019 excluded as a partial collection
month, documented in the trainer). Holdout = Oct 2018 – Sep 2019.
(`demand_forecast_metrics.csv`):

| Best model per drug | Drugs |
|---|---|
| Seasonal naive (2/8) | M01AB (16.31), N02BE (151.64) |
| Holt-Winters (4/8) | M01AE (21.89), N05C (4.72), R03 (66.54), R06 (19.75) |
| LightGBM (2/8) | N02BA (9.87), N05B (35.63) |

History note: an earlier commit message said "naive wins 4/8" — that run
included the partial Oct-2019 month in the holdout, contaminating every MAE.
After the documented exclusion the tally above (from the artifact) is correct.
Finding stands: naive baselines are competitive on 2/8 drugs — stated, not hidden.

## F3 — outbreak detection (IsolationForest, contamination=0.05)

- Input: 126,712 district-weeks, 840 state-qualified districts, 2020-04-26–2023-08-22.
- Flagged: 6,336/126,712 (5.0% — by construction of the contamination parameter;
  never presented as a detection rate).
- Delta-window evaluation (`outbreak_delta_eval.json`, window 2021-04-01–2021-06-30,
  descriptive enrichment — no labeled ground truth exists, so no precision/recall
  is claimed):
  - flag rate inside window: **0.2496** vs **0.0356** outside (~7x enrichment);
  - district coverage: **334/840 (39.8%)** districts flagged at least once in-window.
- Interpretation: retrospective spike flagging on a static archive, not live
  surveillance. Flagged weeks listed in `outbreak_flagged_weeks.csv`.

## QA note (2026-10-07)

- Fail-closed verified: hiding `ds/data/raw/` makes all trainers (and the tuner)
  exit 1 with download instructions.
- Determinism verified: reruns reproduce committed CSVs/pkls/JSON byte-identically.
- License chain verified: Kaggle page (CC BY-NC 4.0) → mirror README restates it →
  `docs/citations.md` attributes → metric CSVs carry the license tag. Project code
  is MIT (`LICENSE`); data terms are unaffected.

## Conformal triage (added 2026-10-08, `conformal_triage.py`)

Split-conformal recall control, alpha=0.15: 60/20/20 stratified splits,
primary model (unweighted XGB + isotonic calibration), calibration quantile on
positives. 200 seeds (0–199), two rules:

- **Marginal (textbook rank k=⌊α(n+1)⌋):** mean test recall **0.884**
  (SD 0.033) vs target 0.85; **12.5% of individual splits fall below 0.85** —
  the guarantee is average over calibration draws, never per-split. Mean
  threshold 0.094, mean precision 0.196, mean abstention 0.043.
- **PAC (Beta order statistic, delta=0.10):** mean recall **0.900**
  (SD 0.030); below-target share **3.5%** (within the delta budget); mean
  threshold 0.089, mean precision 0.191. A lower rank → lower threshold →
  higher recall, at a small precision cost.
- Alpha sweep means (marginal | PAC): 0.05→0.966/0.169 | 0.976/0.164;
  0.10→0.924/0.183 | 0.938/0.179; 0.15→0.884/0.196 | 0.900/0.191;
  0.20→0.832/0.212 | 0.858/0.204; 0.30→0.737/0.243 | 0.768/0.233
  (`conformal_alpha_sweep.csv`).
- Seed-42 reference split (`severity_extras.json`): t=0.0985, recall 0.886
  (95% CI 0.856–0.914), precision 0.199 (0.179–0.215), FPR 0.643, AUC 0.755
  (0.728–0.781), Brier 0.112 (constant 0.129), abstention 0.036.
- NEWS2 vitals-only baseline: **no cutoff ≥1 reaches 0.85 recall on
  calibration** (matched comparison infeasible — documented). Clinical cutoff
  ≥5: recall 0.138/precision 0.207; max-recall cutoff 1: 0.669/0.168.
  The learned model (0.886/0.199) wins on recall at comparable precision.
- Reliability diagram (`calibration_curve.png`): isotonic calibration beats
  the constant baseline (Brier 0.112 vs 0.129) — probabilities drive the
  threshold rule and are not quoted as absolute risks.

Limitations: both conformal rules assume exchangeability between calibration
and test (holds within this US ED sample by construction); neither transfers
automatically to deployment data. F1 evaluation is 5-fold CV plus nested
cross-fitted threshold tuning plus this split study — no external test set.

## Outbreak methods comparison (added 2026-10-08, `outbreak_eval_v2.py`)

| Method | Delta rate / coverage / lead | Omicron rate / coverage / lead | Background |
|---|---|---|---|
| iso_single | 0.250 / 0.398 / 2.0 wk | 0.167 / 0.525 / 2.0 wk | 0.024 |
| iso_confirmed | 0.199 / 0.324 / 1.0 wk | 0.116 / 0.399 / 1.0 wk | 0.013 |
| zscore (prior-4wk) | 0.276 / 0.751 / 4.0 wk | 0.135 / 0.583 / 2.0 wk | 0.093 |

Reading: confirmation costs ~1 week of lead for fewer/weaker alerts; the
z-baseline leads earliest with widest coverage but ~4x the background rate.
All descriptive (`outbreak_eval_v2.json`, `outbreak_methods_compare.png`).
