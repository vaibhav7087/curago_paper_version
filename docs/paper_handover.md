# Data Science Evidence Pack & Paper Handover Document

**Paper Title:** Real-Data Machine Learning for Rural Telemedicine Triage and Outbreak Surveillance  
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
3. **Safety-First Clinical Framing:** In healthcare triage, false negatives (under-triage) can be fatal. We explicitly prioritize sensitivity/recall over raw precision, demonstrating how threshold tuning and split-conformal calibration reduce under-triage from **48.7%** down to **15.5%** and lower.
4. **Honest Scientific Disclosure:** We preempt reviewer skepticism by explicitly highlighting negative or nuanced findings:
   - Conformal guarantees are marginal, not conditional (46% of splits fall below the nominal 85% target).
   - Uncalibrated tree probabilities display overconfidence (probabilities serve as rank orderings, not absolute risk estimates).
   - Naive time-series baselines outperform or match complex ML on 2 of 8 pharmaceutical groups.
   - Outbreak detection operates on an alert budget without gold-standard district infection labels, evaluated via empirical wave enrichment rather than synthetic ROC curves.

---

## 2. Complete Figure Catalog (300 DPI IEEE Figures)

All figures reside in `ds/models/paper/` and are rendered at **300 DPI** in publication-ready IEEE style (vector fonts, publication palettes, explicit data attribution tags).

| Figure ID | Filename | LaTeX Label | Source Script | Description & Key Clinical / Methodological Takeaway |
|---|---|---|---|---|
| **Fig. 1** | `paper_fig0_pipeline.png` | `\label{fig:pipeline}` | `ds/paper_figures.py` | **Curago Architectural Integration:** Flow diagram depicting patient intake (voice transcription, vital extraction), Module 1 patient-level triage (XGBoost + abstention band routing uncertain cases to physicians), Module 2 population outbreak surveillance (IsolationForest alert feed), and pharmaceutical inventory forecasting. |
| **Fig. 2** | `paper_fig1_roc.png` | `\label{fig:roc}` | `ds/paper_figures.py` | **Triage ROC Curves (5-Fold CV):** Comparison of Logistic Regression (AUC 0.719) and XGBoost (AUC 0.732) across $n=13,595$ encounters. Highlighted operating points: default $t=0.50$, safety-tuned $t=0.25$, and conformal reference split ($t=0.173$). Demonstrates gradient boosting dominance across the entire sensitivity curve. |
| **Fig. 3** | `paper_fig2_shap.png` | `\label{fig:shap}` | `ds/paper_figures.py` | **Feature Attribution (SHAP Summary):** Global feature importances for triage. Age, pulse, respiratory rate, and pain score are primary drivers; codified Reason for Visit (RFV) categories contribute clinically meaningful risk adjustments. |
| **Fig. 4** | `paper_fig3_tradeoff.png` | `\label{fig:tradeoff}` | `ds/paper_figures.py` | **Threshold Operating Trade-Offs:** Out-of-fold sweep of threshold $t \in [0.05, 0.95]$. Illustrates the sharp drop in under-triage from $0.487$ to $0.155$ as $t$ shifts to $0.25$, while tracking precision, over-triage, and abstention rate ($23.5\%$). |
| **Fig. 5** | `paper_fig4_conformal_hist.png` | `\label{fig:conformal}` | `ds/paper_figures.py` | **Split-Conformal Recall Distribution (200 Seeds):** Empirical recall distribution at $\alpha=0.15$ target ($1-\alpha=0.85$). Demonstrates marginal coverage ($mean=0.850$, $SD=0.027$) and documents that $46.0\%$ of individual test splits fall below $0.85$, proving the coverage guarantee holds marginally over calibration sets. |
| **Fig. 6** | `paper_fig5_outbreak.png` | `\label{fig:outbreak}` | `ds/paper_figures.py` | **Outbreak Surveillance Evaluation:** Multi-method comparison across Delta and Omicron windows. Shows single-week IsolationForest flags yield $7.0\times$ enrichment ($25.0\%$ in-window vs $3.6\%$ baseline) and $39.8\%$ district coverage, while confirmed alerts cut background noise in half. |
| **Suppl. Fig. S1** | `paper_figS1_forecast.png` | `\label{fig:forecast}` | `ds/paper_figures.py` | **12-Month Pharmaceutical Demand Forecasts:** Out-of-sample holdout trajectories across 8 ATC drug categories comparing Actual vs Holt-Winters vs LightGBM. Shows seasonal baselines competitive on low-variance series. |
| **Suppl. Fig. S1b** | `paper_figS1b_calibration.png` | `\label{fig:calibration}` | `ds/paper_figures.py` | **Triage Reliability Diagram:** Test split calibration curve showing XGBoost probability overconfidence (Brier score $0.158$). Justifies treating predicted probabilities as ordinal risk scores rather than exact frequentist risks. |

---

## 3. Complete Table Catalog (LaTeX & CSV)

All tables are saved in both CSV (for spreadsheet verification) and LaTeX `booktabs` format in `ds/models/paper/`. Numeric cells are rounded to 3 decimal places to prevent drift.

| Table ID | CSV / TeX Filename | LaTeX Label | Primary Metric / Topic | Headline Values |
|---|---|---|---|---|
| **Table I** | `paper_T1_triage_models` | `\label{tab:triage_models}` | Triage Model Comparison (5-Fold CV, $t=0.50$) | LogReg: AUC 0.719, Rec 0.609, Prec 0.269, F1 0.373, Under-triage 0.391<br>XGBoost: AUC 0.732, Rec 0.513, Prec 0.334, F1 0.405, Under-triage 0.487 |
| **Table II** | `paper_T2_operating_points` | `\label{tab:operating_points}` | Operating Points & Abstention Band | XGB $t=0.50$: Rec 0.513, Under-triage 0.487, Abstain 0.080<br>XGB $t=0.25$: Rec 0.845, Under-triage 0.155, Abstain 0.235<br>LogReg $t=0.40$: Rec 0.811, Under-triage 0.189, Abstain 0.394<br>Conformal Seed 42: Rec 0.913, Prec 0.187, Abstain 0.231 |
| **Table III** | `paper_T3_threshold_sweep` | `\label{tab:threshold_sweep}` | Full OOF Threshold Sweep ($t=0.05 \dots 0.95$) | Comprehensive grid of sensitivity, precision, over-triage, under-triage, and abstention for both models. |
| **Table IV** | `paper_T4_conformal` | `\label{tab:conformal}` | Split-Conformal Alpha Sweep (200 Splits) | $\alpha=0.05 \to$ Rec 0.953, Prec 0.171<br>$\alpha=0.10 \to$ Rec 0.902, Prec 0.185<br>$\alpha=0.15 \to$ Rec 0.850, Prec 0.198, Mean Thresh 0.212, Split $<0.85$: 46.0%<br>$\alpha=0.20 \to$ Rec 0.800, Prec 0.212<br>$\alpha=0.30 \to$ Rec 0.700, Prec 0.241 |
| **Table V** | `paper_T5_severity_baseline` | `\label{tab:severity_baseline}` | Conformal vs NEWS2 Clinical Baseline | Conformal XGB: Rec 0.913 [95% CI 0.886–0.940], Prec 0.187, AUC 0.739<br>NEWS2 Cutoff $\ge 5$: Rec 0.138, Prec 0.207<br>NEWS2 Cutoff $\ge 1$: Rec 0.669, Prec 0.168 (no cutoff matches 0.85) |
| **Table VI** | `paper_T6_outbreak` | `\label{tab:outbreak}` | Outbreak Detection Methods vs Surveillance Windows | Single Isolation: Delta rate 0.250, Coverage 39.8%, Lead 2.0 wk, Bkgd 0.024<br>Confirmed Isolation: Delta rate 0.199, Coverage 32.4%, Lead 1.0 wk, Bkgd 0.013<br>Z-Score ($4$-wk): Delta rate 0.276, Coverage 75.1%, Lead 4.0 wk, Bkgd 0.093 |
| **Table VII** | `paper_T7_cohort` | `\label{tab:cohort}` | Cohort Demographics & Dataset Characteristics | Raw NHAMCS: $n=19,481$; Analysis: $n=13,595$; Acute prevalence: $15.2\%$<br>Vital medians: Age 45, Pulse 84, RR 18, SBP 132, O2 98%, Pain 4<br>India: 126,712 district-weeks, 840 districts; Pharma: 69 months |
| **Suppl. Tab. S1** | `paper_TS1_forecast` | `\label{tab:forecast}` | 12-Month Holdout Forecast MAE | Holt-Winters wins 4/8 (M01AE, N05C, R03, R06)<br>LightGBM wins 2/8 (N02BA, N05B)<br>Seasonal Naive wins 2/8 (M01AB, N02BE) |

---

## 4. Claim $\to$ Artifact Traceability Matrix

Every single metric reported in the paper is programmatically asserted against its generating artifact in `ds/models/paper/paper_numbers.json` and its source file:

| Scientific Claim in Paper | Value | Source Artifact | Exact Key / Column | Corroborating Table / Figure |
|---|---|---|---|---|
| **XGBoost Cross-Validation AUC** | 0.732 | `ds/models/metric_table.csv` | `AUC-ROC` (row `XGBoost`) | Table I, Fig. 2 |
| **Logistic Regression CV AUC** | 0.719 | `ds/models/metric_table.csv` | `AUC-ROC` (row `Logistic Regression`) | Table I, Fig. 2 |
| **XGBoost Default Under-Triage** | 0.487 | `ds/models/metric_table.csv` | `Under-Triage Rate` | Table I |
| **XGBoost Tuned Operating Recall** | 0.845 | `ds/models/operating_point.json` | `xgboost.expected_recall` | Table II, Fig. 4 |
| **XGBoost Tuned Under-Triage** | 0.155 | `ds/models/operating_point.json` | `xgboost.expected_under_triage` | Table II, Fig. 4 |
| **XGBoost Tuned Threshold** | 0.250 | `ds/models/operating_point.json` | `xgboost.threshold` | Table II, Fig. 4 |
| **XGBoost Tuned Abstention Rate** | 0.235 | `ds/models/operating_point.json` | `xgboost.abstention_rate` | Table II, Fig. 4 |
| **Conformal Mean Test Recall ($\alpha=0.15$)** | 0.850 | `ds/models/conformal_repeats.csv` | `test_recall.mean()` | Table IV, Fig. 5 |
| **Conformal Recall Standard Deviation** | 0.027 | `ds/models/conformal_repeats.csv` | `test_recall.std()` | Table IV, Fig. 5 |
| **Conformal Mean Threshold** | 0.212 | `ds/models/conformal_repeats.csv` | `threshold.mean()` | Table IV |
| **Conformal Mean Abstention** | 0.218 | `ds/models/conformal_repeats.csv` | `abstention.mean()` | Table IV |
| **Fraction of Splits Below Target ($<0.85$)** | 0.460 (46%) | `ds/models/conformal_repeats.csv` | `(test_recall < 0.85).mean()` | Table IV, Fig. 5 |
| **Seed-42 Test Split Recall** | 0.913 | `ds/models/severity_extras.json` | `xgboost_test.recall` | Table II, Table V |
| **Seed-42 Recall 95% Bootstrap CI** | [0.886, 0.940] | `ds/models/severity_extras.json` | `xgboost_test.recall_ci95` | Table V |
| **Seed-42 Test Split AUC** | 0.739 | `ds/models/severity_extras.json` | `xgboost_test.auc` | Table V, Fig. 2 |
| **Seed-42 Test Split Brier Score** | 0.158 | `ds/models/severity_extras.json` | `xgboost_test.brier` | Table V, Suppl. Fig. S1b |
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
* **Class Imbalance Handling:** Real-world acute prevalence is $15.2\%$ ($2,067 / 13,595$). Class weighting (`scale_pos_weight = 5.58`) is computed strictly on training folds to prevent target leakage.
* **Decision Optimization:**
  * Standard threshold $t=0.50$ produces high under-triage ($48.7\%$), violating clinical safety.
  * We formulate an explicit clinical threshold search rule:
    $$\min t \quad \text{s.t.} \quad \text{Recall}_{\text{OOF}}(t) \ge 0.80$$
    Tie-breaking selects the threshold maximizing precision. This yields $t=0.25$ for XGBoost and $t=0.40$ for Logistic Regression.
* **Abstention Mechanism:**
  * Telemedicine requires human-in-the-loop oversight for borderline predictions.
  * Instances where $\max(P(Y=1|x), P(Y=0|x)) < 0.60$ trigger an automatic "Abstain" routing to senior physician queues. At $t=0.25$, $23.5\%$ of encounters are escalated.
* **Split-Conformal Inference:**
  * Stratified 60% train / 20% calibration / 20% test splits across 200 random seeds.
  * For nominal miscoverage level $\alpha=0.15$ (target recall $1-\alpha=0.85$), the calibration threshold is selected via the empirical conformal quantile:
    $$\hat{q} = \text{Quantile}\left(\{ \hat{s}_i \}_{i \in \text{Cal}^+}, \frac{\lfloor (n_{\text{pos}} + 1)(1 - \alpha) \rfloor}{n_{\text{pos}}} \right)$$
  * Guarantees marginal coverage over the drawing of calibration sets ($E[\text{Recall}] = 0.8504$).

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

### Q4: "If conformal prediction guarantees 85% coverage, why do 46% of splits have recall under 0.85?"
> **Author Response:** Standard split-conformal inference provides a **marginal guarantee** over the random drawing of the calibration and test sets: $\mathbb{E}_{S_{\text{cal}}, S_{\text{test}}}[\text{Recall}] \ge 1 - \alpha$. It does *not* provide a PAC or conditional guarantee that every realization satisfies the bound. Over 200 random splits, the empirical mean recall is $0.8504$ (exactly matching theory), with split recalls distributed approximately symmetrically around the mean ($\text{SD} = 0.027$). We include the full histogram (`Fig. 5`) and dispersion table (`Table IV`) specifically to educate clinical readers on the distinction between marginal and conditional conformal coverage.

### Q5: "The calibration diagram shows probability overconfidence. Does this invalidate the triage model?"
> **Author Response:** Tree-based ensembles with class reweighting are well-known to distort posterior probability calibration towards extreme values while preserving rank ordering. As shown in Suppl. Fig. S1b (Brier score $0.158$), probabilities should not be interpreted as frequentist patient mortality risks. Instead, our deployment architecture treats model outputs as monotonic risk scores, utilizing conformal quantile mapping and ROC threshold tuning rather than raw probability cutoffs.

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
