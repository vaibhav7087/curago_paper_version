# Paper captions (figures + tables)

All numbers below are asserted against committed artifacts by
`ds/paper_figures.py` / `ds/paper_tables.py`. Labels are suggestions for LaTeX.

## Figures (300 dpi PNGs in `ds/models/paper/`)

**Fig. 1 (fig0, `paper_fig0_pipeline.png`, label `fig:pipeline`)** — The Curago
data-science layer: voice intake and vitals + reason-for-visit features feed
Module 1 (patient-level triage, XGBoost with an abstain band), while district
counts feed Module 2 (population-level outbreak surveillance, IsolationForest);
low-confidence cases route to a doctor, and forecasts support medicine
delivery. Data: CDC NHAMCS 2019 (public domain); covid19india static archive;
Kaggle pharma sales (CC BY-NC 4.0, supplementary).

**Fig. 2 (fig1, `paper_fig1_roc.png`, label `fig:roc`)** — Triage ROC under
5-fold cross-validation (13,595 analysis
encounters): LogReg AUC 0.719,
XGBoost AUC 0.732. Marked operating
points: XGB t=0.50 (FPR 0.18, TPR 0.51), tuned t=0.25 (0.58, 0.84), and the
seed-42 conformal split (0.71, 0.91). Data: CDC NHAMCS 2019 ED.

**Fig. 3 (fig2, `paper_fig2_shap.png`, label `fig:shap`)** — SHAP summary for
the XGBoost triage model: age, pulse, respiratory rate and pain scale dominate;
reason-for-visit codes contribute through clinical-pattern splits. Data: CDC
NHAMCS 2019 ED.

**Fig. 4 (fig3, `paper_fig3_tradeoff.png`, label `fig:tradeoff`)** — OOF
threshold trade-offs (recall, precision, under-/over-triage, abstention) with
the pre-declared selections t=0.40 (LogReg) and t=0.25 (XGBoost). Data: CDC
NHAMCS 2019 ED.

**Fig. 5 (fig4, `paper_fig4_conformal_hist.png`, label `fig:conformal`)** —
Split-conformal test recall over 200 seeds at alpha=0.15: mean
0.850 vs the 0.85 target;
46% of individual
splits fall below target — the guarantee is marginal, not per-split. Data: CDC
NHAMCS 2019 ED.

**Fig. 6 (fig5, `paper_fig5_outbreak.png`, label `fig:outbreak`)** —
Outbreak methods versus the Delta and Omicron windows (in-window flag rate and
district coverage): descriptive enrichment only; isolation single-week flags
enrich Delta 7.0x. Data: covid19india static archive, 840 districts.

**Suppl. Fig. S1 (figS1, `paper_figS1_forecast.png`, label `fig:forecast`)** —
Held-out 12-month forecasts for 8 ATC drug groups (actual vs Holt-Winters vs
LightGBM, MAE in legend). Data: Kaggle pharma sales (CC BY-NC 4.0).

**Suppl. Fig. S1b (figS1b, `paper_figS1b_calibration.png`,
label `fig:calibration`)** — Reliability diagram, seed-42 test (Brier
0.1580): the model is overconfident;
probabilities are rank-useful for the threshold rule, not quoted as risks.
Data: CDC NHAMCS 2019 ED.

## Tables (`ds/models/paper/paper_T*.csv` + `.tex`)

**Table I (T1, label `tab:triage_models`)** — Triage model comparison, 5-fold
CV (t=0.5): XGBoost AUC 0.732 vs LogReg
0.719; XGB under-triage
0.487 at default
threshold. Data: CDC NHAMCS 2019 ED.

**Table II (T2, label `tab:operating_points`)** — Operating points: default
t=0.50, tuned t=0.25 (recall 0.845,
under-triage 0.155),
LogReg t=0.40, and conformal seed-42 (recall
0.913). Data: CDC NHAMCS 2019 ED.

**Table III (T3, label `tab:threshold_sweep`)** — OOF threshold sweep for both
models (supplementary). Data: CDC NHAMCS 2019 ED.

**Table IV (T4, label `tab:conformal`)** — Conformal recall across 200 splits
per alpha; mean recall
0.850 at alpha=0.15 with
SD 0.027; fraction of splits
below 0.85 = 0.460.
Data: CDC NHAMCS 2019 ED.

**Table V (T5, label `tab:severity_baseline`)** — Seed-42 conformal test
performance with bootstrap 95% CIs versus the NEWS2 vitals-only baseline; no
NEWS2 cutoff >= 1 reaches 0.85 recall on calibration (matched comparison
infeasible). Data: CDC NHAMCS 2019 ED.

**Table VI (T6, label `tab:outbreak`)** — Outbreak methods x windows with
background rates; Delta enrichment 7.0x for single-week isolation flags.
Descriptive only — no labeled ground truth. Data: covid19india static archive.

**Table VII (T7, label `tab:cohort`)** — Cohort and dataset summary (n,
prevalence, vitals median [Q1, Q3]; district-weeks; months). Data: as listed
per row.

**Suppl. Table S1 (TS1, label `tab:forecast`)** — Forecast MAE per drug;
Holt-Winters 4/8, naive 2/8, LightGBM 2/8. Data: Kaggle pharma sales
(CC BY-NC 4.0).
