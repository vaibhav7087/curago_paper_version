# Data Science Evidence Pack & Paper Handover Document

**Paper Title:** Real-Data Machine Learning for Telemedicine Triage and Outbreak Surveillance  
**Target Venue:** IEEE Transactions / Conference Publication (Nexathon II Paper Track)  
**Repository:** `curago_paper_version`  
**Date:** October 2026  
**Status:** Complete, Verified, Deterministic (Seed 42), Real-Data Only (Fail-Closed)

---

## 1. Executive Summary & Core Philosophy

This document serves as the master data science handover package for the academic writing team. The accompanying repository contains a complete, verified, and 100% reproducible experimental pipeline for the Curago telemedicine platform.

### Core Scientific Principles Enforced:
1. **Zero Synthetic Data:** All findings, figures, and tables are generated strictly from authenticated, publicly available real-world data sources (CDC NHAMCS 2019 ED, COVID-19 India district time-series, and Kaggle Pharma Sales POS data). If raw data files are absent, scripts abort immediately (*fail-closed* policy).
2. **Deterministic Reproducibility:** Every pipeline script is pinned to seed 42 (and multi-seed sweeps across seeds 0–199). Output numbers, metrics, and LaTeX tables match bit-for-bit across runs.
3. **Safety-First Clinical Framing:** In healthcare triage, false negatives (under-triage) can be fatal. We explicitly prioritize sensitivity/recall over raw precision, demonstrating how threshold tuning and split-conformal calibration reduce under-triage from **85.9%** (default $t=0.50$) down to **17.2%** (nested cross-fitted thresholds $t=0.09$–$0.10$).
4. **Honest Scientific Disclosure:** We preempt reviewer skepticism by explicitly highlighting negative or nuanced findings:
   - The marginal conformal guarantee is average-case: **12.5%** of individual splits fall below the nominal 85% target; the **PAC rule** (Beta order statistic, $\delta=10\%$) bounds that share to **3.5%** at a small precision cost.
   - Probabilities are calibrated (OOF Brier **0.113** beats the constant **0.129**) but are not validated as absolute risks — they drive a threshold rule, not quoted risks.
   - Naive time-series baselines outperform or match complex ML on 2 of 8 pharmaceutical groups.
   - Outbreak detection operates on an alert budget without gold-standard district infection labels, evaluated via empirical wave enrichment rather than synthetic ROC curves.

---

## 2. Complete Figure Catalog (300 DPI IEEE Figures)

All figures reside in `ds/models/paper/` and are rendered at **300 DPI** in publication-ready IEEE style (vector fonts, publication palettes, explicit data attribution tags).

| Figure ID | Filename | LaTeX Label | Source Script | Description & Key Clinical / Methodological Takeaway |
|---|---|---|---|---|
| **Fig. 1** | `paper_fig0_pipeline.png` | `\label{fig:pipeline}` | `ds/paper_figures.py` | **Curago Architectural Integration:** Flow diagram depicting patient intake (voice transcription, vital extraction), Module 1 patient-level triage (XGBoost + abstention band routing uncertain cases to physicians), Module 2 population outbreak surveillance (IsolationForest alert feed), and pharmaceutical inventory forecasting. |
| **Fig. 2** | `paper_fig1_roc.png` | `\label{fig:roc}` | `ds/paper_figures.py` | **Triage ROC Curves (5-Fold CV):** Comparison of Logistic Regression (AUC 0.719, CI 0.708–0.731) and the calibrated primary XGBoost (AUC 0.742, CI 0.728–0.751) across $n=13{,}595$ encounters. Highlighted operating points: default $t=0.50$ (Recall 0.141), tuned $t=0.11$ (Recall 0.800), and the seed-42 conformal split ($t=0.099$). Demonstrates gradient boosting dominance across the entire sensitivity curve. |
| **Fig. 3** | `paper_fig2_shap.png` | `\label{fig:shap}` | `ds/paper_figures.py` | **Feature Attribution (SHAP Summary):** Global feature importances for triage. Age, pulse, respiratory rate, and pain score are primary drivers; codified Reason for Visit (RFV) categories contribute clinically meaningful risk adjustments. |
| **Fig. 4** | `paper_fig3_tradeoff.png` | `\label{fig:tradeoff}` | `ds/paper_figures.py` | **Threshold Operating Trade-Offs:** Out-of-fold sweep of threshold $t$. Shows the safety-first trade-off of the nested cross-fitted operating point: under-triage $0.859 \to 0.172$ as thresholds move to $t=0.09$–$0.10$, at FPR $0.545$ and abstention rate $4.2\%$. |
| **Fig. 5** | `paper_fig4_conformal_hist.png` | `\label{fig:conformal}` | `ds/paper_figures.py` | **Split-Conformal Recall Distribution (200 Seeds):** Empirical recall distributions at $\alpha=0.15$ (target $0.85$). Marginal rule: mean $0.884$, $SD=0.033$, **12.5%** of splits below $0.85$. **PAC rule** ($\delta=10\%$): mean $0.900$, below-target share **3.5%** — a PAC upper bound on the below-target share, not a per-split guarantee. |
| **Fig. 6** | `paper_fig5_outbreak.png` | `\label{fig:outbreak}` | `ds/paper_figures.py` | **Outbreak Surveillance Evaluation:** Multi-method comparison across Delta and Omicron windows. Shows single-week IsolationForest flags yield $7.0\times$ enrichment ($25.0\%$ in-window vs $3.6\%$ baseline) and $39.8\%$ district coverage, while confirmed alerts cut background noise in half. |
| **Suppl. Fig. S1** | `paper_figS1_forecast.png` | `\label{fig:forecast}` | `ds/paper_figures.py` | **12-Month Pharmaceutical Demand Forecasts:** Out-of-sample holdout trajectories across 8 ATC drug categories comparing Actual vs Holt-Winters vs LightGBM. Shows seasonal baselines competitive on low-variance series. |
| **Suppl. Fig. S1b** | `paper_figS1b_calibration.png` | `\label{fig:calibration}` | `ds/paper_figures.py` | **Triage Reliability Diagram:** Calibration curve with Brier **0.112** (seed-42 test, isotonic) vs **0.129** for a constant at train prevalence. Isotonic calibration beats the constant; probabilities are still used to drive a threshold rule, not quoted as absolute risks. |

---

## 3. Complete Table Catalog (LaTeX & CSV)

All tables are saved in both CSV (for spreadsheet verification) and LaTeX `booktabs` format in `ds/models/paper/`. Numeric cells are rounded to 3 decimal places to prevent drift.

| Table ID | CSV / TeX Filename | LaTeX Label | Primary Metric / Topic | Headline Values |
|---|---|---|---|---|
| **Table I** | `paper_T1_triage_models` | `\label{tab:triage_models}` | Triage Model Comparison (5-Fold CV, $t=0.50$, OOF Brier) | LogReg: AUC 0.719, Rec 0.609, Prec 0.269, F1 0.373, Brier 0.209<br>Class-weighted XGB: AUC 0.732, Rec 0.513, Prec 0.334, F1 0.405, Brier 0.165<br>**Primary (unweighted + isotonic): AUC 0.742, Rec 0.141, Brier 0.113**<br>Constant reference: AUC 0.500, Brier 0.129 |
| **Table II** | `paper_T2_operating_points` | `\label{tab:operating_points}` | Operating Points & Abstention Band (FPR included) | Primary $t=0.50$: Rec 0.141, FPR 0.016, Under-triage 0.859, Abstain 0.042<br>Primary $t=0.11$ (pooled): Rec 0.800, Prec 0.226, FPR 0.493, Under 0.200, Abstain 0.042<br>**Nested cross-fitted $t=0.09$–$0.10$: Rec 0.829, FPR 0.545, Under 0.172**<br>LogReg $t=0.40$: Rec 0.811, FPR 0.536, Under 0.189, Abstain 0.394<br>Seed-42 conformal: Rec 0.886, Prec 0.199, FPR 0.643, Abstain 0.036 |
| **Table III** | `paper_T3_threshold_sweep` | `\label{tab:threshold_sweep}` | Full OOF Threshold Sweep (FPR included) | Comprehensive grid of sensitivity, precision, over-triage, under-triage, FPR, and abstention for both models. |
| **Table IV** | `paper_T4_conformal` | `\label{tab:conformal}` | Conformal Alpha Sweep, Marginal vs PAC (200 Splits) | $\alpha=0.15$ marginal: Rec 0.884, Prec 0.196, thr 0.094, below 12.5%<br>$\alpha=0.15$ **PAC ($\delta=0.10$): Rec 0.900, Prec 0.191, thr 0.089, below 3.5%**<br>Full $\alpha \in \{0.05,0.10,0.15,0.20,0.30\}$ × Rule grid |
| **Table V** | `paper_T5_severity_baseline` | `\label{tab:severity_baseline}` | Conformal vs NEWS2 Clinical Baseline | Conformal seed-42: Rec 0.886 [95% CI 0.856–0.914], Prec 0.199, AUC 0.755, Brier 0.112 (constant 0.129)<br>NEWS2 Cutoff $\ge 5$: Rec 0.138, Prec 0.207<br>NEWS2 Cutoff $\ge 1$: Rec 0.669, Prec 0.168 (no cutoff matches 0.85) |
| **Table VI** | `paper_T6_outbreak` | `\label{tab:outbreak}` | Outbreak Detection Methods vs Surveillance Windows | Single Isolation: Delta rate 0.250, Coverage 39.8%, Lead 2.0 wk, Bkgd 0.024<br>Confirmed Isolation: Delta rate 0.199, Coverage 32.4%, Lead 1.0 wk, Bkgd 0.013<br>Z-Score ($4$-wk): Delta rate 0.276, Coverage 75.1%, Lead 4.0 wk, Bkgd 0.093 |
| **Table VII** | `paper_T7_cohort` | `\label{tab:cohort}` | Cohort Demographics & Dataset Characteristics | Raw NHAMCS: $n=19,481$; Analysis: $n=13,595$; Acute prevalence: $15.2\%$<br>Vital medians: Age 45, Pulse 84, RR 18, SBP 132, O2 98%, Pain 4<br>India: 126,712 district-weeks, 840 districts; Pharma: 69 months |
| **Suppl. Tab. S1** | `paper_TS1_forecast` | `\label{tab:forecast}` | 12-Month Holdout Forecast MAE | Holt-Winters wins 4/8 (M01AE, N05C, R03, R06)<br>LightGBM wins 2/8 (N02BA, N05B)<br>Seasonal Naive wins 2/8 (M01AB, N02BE) |

---

## 4. Claim $\to$ Artifact Traceability Matrix

Every single metric reported in the paper is programmatically asserted against its generating artifact in `ds/models/paper/paper_numbers.json` and its source file:

| Scientific Claim in Paper | Value | Source Artifact | Exact Key / Column | Corroborating Table / Figure |
|---|---|---|---|---|
| **Primary Model CV AUC** | 0.742 | `ds/models/metric_table.csv` | `AUC-ROC` (row `XGBoost (unweighted + isotonic)`) | Table I, Fig. 2 |
| **Primary AUC 95% CI / Diff CI** | [0.728, 0.751] / [0.012, 0.031] | `ds/models/paper/paper_numbers.json` | `module1_triage.auc_xgb_ci95` / `auc_diff_ci95` | §IV-A |
| **Class-Weighted XGB CV AUC** | 0.732 | `ds/models/metric_table.csv` | `AUC-ROC` (row `XGBoost (class-weighted)`) | Table I, Fig. 2 |
| **Logistic Regression CV AUC** | 0.719 | `ds/models/metric_table.csv` | `AUC-ROC` (row `Logistic Regression`) | Table I, Fig. 2 |
| **Primary Default Under-Triage ($t=0.5$)** | 0.859 | `ds/models/metric_table.csv` | `Under-Triage Rate` (primary row) | Table I |
| **Primary Default FPR / Abstention** | 0.016 / 0.042 | `ds/models/threshold_tradeoff.csv` | `fpr`, `abstention_rate` at `threshold=0.50` | Table II |
| **Tuned Pooled Threshold / Recall / FPR** | 0.11 / 0.800 / 0.493 | `ds/models/operating_point.json` | `xgboost.threshold`, `expected_recall`, `expected_fpr` | Table II, Fig. 4 |
| **Nested Cross-Fitted Recall / Under-Triage** | 0.829 / 0.172 | `ds/models/operating_point.json` | `nested_xgboost.pooled.recall`, `.under_triage` | Table II, Fig. 4 |
| **Nested Thresholds (5 outer folds)** | 0.09–0.10 | `ds/models/operating_point.json` | `nested_xgboost.thresholds` | Table II |
| **Tuned Abstention Rate** | 0.042 | `ds/models/operating_point.json` | `xgboost.abstention_rate` | Table II, Fig. 4 |
| **Marginal Conformal Mean Recall ($\alpha=0.15$)** | 0.884 | `ds/models/conformal_repeats.csv` | `test_recall.mean()` | Table IV, Fig. 5 |
| **Marginal Recall SD / Fraction Below 0.85** | 0.033 / 0.125 (12.5%) | `ds/models/conformal_repeats.csv` | `test_recall.std()`, `(test_recall < 0.85).mean()` | Table IV, Fig. 5 |
| **Marginal Mean Threshold / Precision / Abstention** | 0.094 / 0.196 / 0.043 | `ds/models/conformal_repeats.csv` | `threshold`, `test_precision`, `abstention` means | Table IV |
| **PAC Mean Recall / Fraction Below 0.85** | 0.900 / 0.035 (3.5%) | `ds/models/conformal_repeats.csv` | rows `rule=pac` | Table IV, Fig. 5 |
| **PAC Mean Threshold / Precision / SD** | 0.089 / 0.191 / 0.030 | `ds/models/conformal_repeats.csv` | rows `rule=pac` | Table IV |
| **PAC delta ($\delta$)** | 0.10 | `ds/conformal_triage.py` | `PAC_DELTA` | §III-C, Table IV |
| **Seed-42 Test Split Recall** | 0.886 | `ds/models/severity_extras.json` | `xgboost_test.recall` | Table II, Table V |
| **Seed-42 Recall 95% Bootstrap CI** | [0.856, 0.914] | `ds/models/severity_extras.json` | `bootstrap_95ci.recall` | Table V |
| **Seed-42 Test Split AUC** | 0.755 | `ds/models/severity_extras.json` | `xgboost_test.auc` | Table V, Fig. 2 |
| **Seed-42 Test Split Brier (vs Constant)** | 0.112 (0.129) | `ds/models/severity_extras.json` | `xgboost_test.brier`, `.brier_constant_trainprev` | Table V, Suppl. Fig. S1b |
| **Seed-42 Test Split FPR / Threshold** | 0.643 / 0.099 | `ds/models/severity_extras.json` | `xgboost_test.fpr`, `split.threshold` | Table II, Table V |
| **NEWS2 Clinical Cutoff $\ge 5$ Recall** | 0.138 | `ds/models/severity_extras.json` | `news2_eval.cutoff_5.recall` | Table V |
| **NEWS2 Best Possible Recall (Cutoff $\ge 1$)** | 0.669 | `ds/models/severity_extras.json` | `news2_eval.cutoff_1.recall` | Table V |
| **Delta Outbreak In-Window Flag Rate** | 0.250 | `ds/models/outbreak_delta_eval.json`| `inside_window_rate` | Table VI, Fig. 6 |
| **Outbreak Background Flag Rate** | 0.036 (outside) / 0.024 (pre-wave) | `ds/models/outbreak_delta_eval.json`| `outside_window_rate` | Table VI, Fig. 6 |
| **Delta Wave Flag Rate Enrichment** | 7.011 ($7.0\times$) | `ds/models/outbreak_delta_eval.json`| `enrichment` | Table VI, Fig. 6 |
| **Delta Window District Coverage** | 0.398 (39.8%) | `ds/models/outbreak_delta_eval.json`| `flagged_district_ratio` | Table VI, Fig. 6 |
| **Isolation Confirmed Background Rate** | 0.013 | `ds/models/outbreak_eval_v2.json` | `iso_confirmed.background_rate`| Table VI |
| **Z-Score Baseline Background Rate** | 0.093 | `ds/models/outbreak_eval_v2.json` | `zscore.background_rate` | Table VI |
| **Pharma Forecast Usable Months** | 69 | `ds/models/demand_forecast_metrics.csv` | Excludes Oct 2019 partial month | Table VII |
| **Forecast Model Win Count (MAE)** | HW: 4, LGBM: 2, Naive: 2 | `ds/models/demand_forecast_metrics.csv` | Lowest MAE per group | Suppl. Table S1 |

---

## 5. Formal Methods Specification

### 5.1 Module 1: Patient-Level Severity Triage
* **Target Construct:** Identification of acute encounters requiring urgent physician escalation (defined as emergent/urgent triage level, hospital admission, ICU transfer, or emergency resuscitation).
* **Feature Representation:** Demographic variables (age, sex), physiological vital signs (systolic blood pressure, pulse, respiratory rate, oxygen saturation, pain scale), and categorical Reason for Visit (RFV) clusters.
* **Class Imbalance Handling:** Real-world acute prevalence is $15.2\%$ ($2,067 / 13,595$). The class-weighted candidate computes `scale_pos_weight` strictly on training folds to prevent leakage; the **primary** model is unweighted (class weighting distorts probabilities — Brier $0.165$ vs $0.113$).
* **Model Selection:** Six pipelines are compared (LogReg, weighted XGB, unweighted XGB, + isotonic, + Platt, constant). The primary is fixed by a pre-declared rule: **lowest 5-fold OOF Brier among the calibrated candidates** (ties → isotonic). Primary AUC $0.742$ (CI $0.728$–$0.751$) vs LogReg $0.719$ (CI $0.708$–$0.731$); paired difference CI $[0.012, 0.031]$.
* **Decision Optimization:**
  * Standard threshold $t=0.50$ produces high under-triage ($85.9\%$, FPR $0.016$), violating clinical safety.
  * We formulate an explicit clinical threshold search rule:
    $$t^{\*} = \arg\max_{t} \ \text{Precision}_{\text{OOF}}(t) \quad \text{s.t.} \quad \text{Recall}_{\text{OOF}}(t) \ge 0.80$$
    (grid of $0.01$ steps; ties → higher threshold). This yields $t=0.11$ pooled for the primary and $t=0.40$ for Logistic Regression.
  * **Nested cross-fitting:** each of 5 outer folds re-selects its threshold on the other four folds only, then evaluates on its own test fold — the conservative headline: thresholds $0.09$–$0.10$, recall $0.829$, FPR $0.545$, under-triage $0.172$.
* **Abstention Mechanism:**
  * Telemedicine requires human-in-the-loop oversight for borderline predictions.
  * Instances where $\max(P(Y=1|x), P(Y=0|x)) < 0.60$ — i.e. calibrated probability in $(0.4, 0.6)$ — trigger an automatic "Abstain" routing to senior physician queues. This band marks indecision near the boundary, not confident errors; abstention is $4.2\%$ for the primary ($39.4\%$ for LogReg) and threshold-independent.
* **Split-Conformal Inference:**
  * Stratified 60% train / 20% calibration / 20% test splits across 200 random seeds.
  * For nominal miscoverage level $\alpha=0.15$ (target recall $1-\alpha=0.85$), the marginal rule uses the textbook rank $k=\lfloor \alpha(n+1) \rfloor$ on calibration positives.
  * The **PAC rule** instead picks the largest $k$ with $\mathrm{BetaCDF}(1-\alpha;\, n+1-k,\, k) \le \delta$ ($\delta=0.10$): a smaller $k$ → lower threshold → higher recall.
  * Results: marginal mean recall $0.884$ (SD $0.033$), **12.5%** of splits below target; PAC mean $0.900$ (SD $0.030$), below-target share **3.5%** ($\le \delta$). The marginal rule guarantees only the expectation; the PAC rule bounds the *share of below-target splits* at $\delta$ — still not a per-split guarantee.

### 5.2 Module 2: Retrospective Population Outbreak Surveillance
* **Objective:** Detect localized syndromic case spikes across 840 Indian administrative districts without supervised disease labels.
* **Surveillance Engine:** IsolationForest trained on rolling district case dynamics (current week counts, 2-week momentum, 4-week moving average).
* **Operational Alert Budget:** Contamination is fixed at $c=0.05$ ($5.0\%$ of total 126,712 district-weeks). This represents a strict administrative investigation budget rather than a classification boundary.
* **Evaluation Framework:** Evaluated during the historical Indian Delta wave (April 1 – June 30, 2021) and Omicron wave (Jan 1 – Feb 28, 2022).
* **Multi-Method Comparison:**
  1. *Single-Week Isolation:* Flags individual anomalies ($25.0\%$ in-window rate, $2.0$ weeks lead time, $0.024$ pre-wave noise).
  2. *Confirmed Rule (2 consecutive weeks):* Requires consecutive alerts, cutting false alarm background rate by $46\%$ ($0.013$) at the expense of $1$ week lead time.
  3. *Statistical Z-Score (4-week rolling baseline):* Traditional epidemiological alerting ($Z > 2.0$), offering early 4-week lead times but suffering from high background alarm rates ($0.093$, nearly $4\times$ higher noise).

### 5.3 Supplementary: Supply Chain Demand Forecasting
* **Data Granularity:** Monthly point-of-sale volume across 8 Anatomical Therapeutic Chemical (ATC) classifications.
* **Validation Scheme:** Single 12-month holdout (Oct 2018 – Sep 2019) with 57 months of historical training.
* **Comparative Baselines:** Holt-Winters Multiplicative Exponential Smoothing vs LightGBM autoregressive boosting vs Seasonal Naive baseline ($y_t = y_{t-12}$).

---

## 6. Reviewer Q&A: Preempting Methodological Objections

### Q1: "Why evaluate rural Indian telemedicine triage using US ED data (NHAMCS)?"
> **Author Response:** Granular, open-access, labeled emergency triage records containing synchronized vital signs, reason-for-visit codes, and clinical dispositions do not exist in public Indian electronic health record (EHR) repositories due to data privacy frameworks and low electronic health record penetration in primary health centers. NHAMCS represents the gold standard for clinical triage benchmark research. However, we explicitly state that while conformal coverage holds within exchangeable samples of this dataset, it does not transfer unconditionally under geographic covariate shift. In Curago, this model serves as an algorithmic decision-support prior, guarded by the abstention band and physician verification.

### Q2: "Why report outbreak performance as 'enrichment' rather than standard Precision, Recall, and ROC-AUC?"
> **Author Response:** District-level epidemiological time-series in open volunteer repositories lack binary, gold-standard 'outbreak' truth labels at weekly resolution. Creating synthetic labels or using arbitrary case thresholds introduces researcher bias. Instead, we adhere to established epidemiological surveillance benchmarks (e.g., CDC EARS methodology), reporting empirical alert concentration during verified wave periods versus background rates. Under an invariant $5\%$ alert budget, observing a $7.0\times$ concentration ($25.0\%$ vs $3.6\%$) confirms high syndromic signal capture without metric fabrication.

### Q3: "Why did seasonal naive forecasting outperform machine learning on two pharmaceutical groups?"
> **Author Response:** Reporting where naive baselines succeed is critical for scientific integrity. For drug classes with rigid annual seasonality and low high-frequency variance (e.g., anti-inflammatory M01AB and analgesics N02BE), gradient boosting overfits residual fluctuations, whereas seasonal persistence ($y_t = y_{t-12}$) provides a robust, zero-parameter forecast. We report this transparently to demonstrate that ML models should only be deployed where they empirically outperform parsimonious baselines.

### Q4: "If conformal prediction targets 85% recall, why do 12.5% of splits have recall under 0.85?"
> **Author Response:** Standard split-conformal inference provides a **marginal guarantee** over the random drawing of the calibration and test sets: $\mathbb{E}_{S_{\text{cal}}, S_{\text{test}}}[\text{Recall}] \ge 1 - \alpha$. It does *not* guarantee that every realization satisfies the bound. Over 200 random splits the empirical mean recall is $0.884$ (SD $0.033$), with $12.5\%$ of splits below $0.85$. We therefore also report a **PAC rule** (Beta order statistic, $\delta=0.10$), which chooses a lower rank so that the *share of below-target splits* is bounded at $\delta$: observed $3.5\%$, mean recall $0.900$, at a small precision cost ($0.191$ vs $0.196$). Neither rule is a per-split guarantee — this distinction is stated explicitly in the paper, with the full dual histogram (`Fig. 5`) and dispersion table (`Table IV`).

### Q5: "Does the reliability diagram invalidate the model's probabilities?"
> **Author Response:** The primary model is isotonic-calibrated precisely to address this: OOF Brier $0.113$ beats both the class-weighted candidate ($0.165$) and the constant-at-prevalence baseline ($0.129$). However, we do not claim the probabilities are validated patient risks — there is no external test set, and calibration holds only within this cohort. The deployment architecture therefore uses the probabilities to drive a threshold rule and an abstention band rather than quoting them as absolute risks, with the conformal rules providing the recall control.

---

## 7. Glossary of Clinical and Statistical Terms

* **Under-Triage Rate:** The fraction of true high-acuity/emergent patients incorrectly classified as non-urgent ($1 - \text{Recall}$). In emergency medicine, this represents dangerous false discharge.
* **Over-Triage Rate:** The fraction of low-acuity patients classified as urgent ($1 - \text{Specificity}$ or False Positive Rate). Represents resource utilization burden.
* **Abstention Band:** A prediction region where the classifier abstains from automated recommendation due to insufficient confidence ($\max(P) < 0.60$), triggering manual escalation.
* **NEWS2 (National Early Warning Score 2):** The standard UK/NHS physiological bedside scoring system based strictly on vital signs (respiration, oxygen saturation, blood pressure, pulse, temperature, consciousness).
* **Conformal Miscoverage ($\alpha$):** The allowable nominal error rate for the target class; $\alpha=0.15$ enforces a minimum marginal sensitivity of $1 - \alpha = 0.85$ ($85\%$).
* **Contamination Factor ($c$):** In unsupervised IsolationForest modeling, the pre-set proportion of anomalous points in the dataset (fixed at $0.05$ to reflect public health investigation capacity).
* **ATC Classification:** The WHO Anatomical Therapeutic Chemical classification system for pharmaceutical substances (e.g., M01A anti-inflammatory, N02B analgesics, R03 antiasthmatics).

---

## 8. IEEE Manuscript Formatting & Assembly Checklist

- [x] **Figures:** All 8 figures exported at 300 DPI (`paper_fig0` through `paper_fig5`, `paper_figS1`, `paper_figS1b`).
- [x] **Captions:** Self-contained, publication-ready captions compiled in `docs/captions.md`.
- [x] **Tables:** All 8 tables available in LaTeX `booktabs` format with standard alignment tags (`ds/models/paper/paper_T*.tex`).
- [x] **Data Source Attributions:** Every table and figure explicitly prints the originating data source.
- [x] **Bibliography:** Fully formatted BibTeX entries in `docs/references.bib`.
- [x] **Data & Code Availability:** Formal declaration ready for inclusion in `docs/availability_statements.md`.
- [x] **Zero Drift:** All values in `paper_numbers.json` identical to numbers cited in tables, captions, and text.
