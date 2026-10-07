"""
ds/paper_tables.py — IEEE paper tables for the writing team (new files only).

Reads committed artifacts; originals untouched. Writes, into ds/models/paper/:
  paper_T1..T7_*.csv + .tex (booktabs), paper_TS1_forecast_*.csv + .tex,
  paper_numbers.json  and, into docs/, captions.md (figure + table captions).

Every CSV keeps a Data Source column; numeric cells are rounded to 3 dp so CSV,
LaTeX and paper_numbers.json cannot drift from one another. Self-asserts cross-
check every headline claim against its source artifact (see docs/results.md and
docs/headline_numbers.md); a mismatch raises instead of emitting numbers.

Usage: python ds/paper_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
PAPER_DIR = MODEL_DIR / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = HERE.parent / "docs"
RAW_DIR = HERE / "data" / "raw"
PROCESSED_DIR = HERE / "data" / "processed"

SRC_NHAMCS = "CDC NHAMCS 2019 ED (n=13,595)"
SRC_KAGGLE = "Kaggle milanzdravkovic/pharma-sales-data (CC BY-NC 4.0)"
SRC_COVID = "covid19india static archive (data.incovid19.org, retrieved 2026-10-07)"
RAW_NHAMCS_N = 19481  # ed2019_sas.sas7bdat shape=(19481, 911); see ds/data/README.md


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Real data is required — no synthetic fallback. {hint}"
        )
    return path


def r3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return round(float(x), 3)


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def booktabs(df: pd.DataFrame, caption: str, label: str, colspec: str,
             note: str = "") -> str:
    head = " & ".join(esc(str(c)) for c in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(esc(str(v)) for v in row.values) + r" \\")
    note_tex = f"\n\\par\\vspace{{2pt}}\\raggedright\\footnotesize {note}" if note else ""
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}" + note_tex + "\n\\end{table}\n"
    )


def emit(name: str, df: pd.DataFrame, caption: str, label: str, colspec: str,
         note: str = "") -> None:
    csv_path = PAPER_DIR / f"{name}.csv"
    tex_path = PAPER_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    tex_path.write_text(booktabs(df, caption, label, colspec, note),
                        encoding="utf-8", newline="\n")
    back = pd.read_csv(csv_path)
    assert list(back.columns) == list(df.columns), f"{name}: column drift"
    assert "Data Source" in df.columns, f"{name}: missing Data Source"
    assert len(back) == len(df), f"{name}: row drift"
    print(f"wrote {csv_path.name} + {tex_path.name} ({len(df)} rows)")


def main():
    # ------------------------------------------------------------- load sources
    metric = pd.read_csv(require(MODEL_DIR / "metric_table.csv",
                                 "run python ds/train_triage.py"))
    trade = pd.read_csv(require(MODEL_DIR / "threshold_tradeoff.csv",
                                "run python ds/tune_threshold.py"))
    op = json.loads(require(MODEL_DIR / "operating_point.json",
                            "run python ds/tune_threshold.py").read_text())
    extras = json.loads(require(MODEL_DIR / "severity_extras.json",
                                "run python ds/severity_eval_extras.py").read_text())
    repeats = pd.read_csv(require(MODEL_DIR / "conformal_repeats.csv",
                                  "run python ds/conformal_triage.py"))
    sweep = pd.read_csv(require(MODEL_DIR / "conformal_alpha_sweep.csv",
                                "run python ds/conformal_triage.py"))
    outbreak = json.loads(require(MODEL_DIR / "outbreak_eval_v2.json",
                                  "run python ds/outbreak_eval_v2.py").read_text())
    delta_ev = json.loads(require(MODEL_DIR / "outbreak_delta_eval.json",
                                  "run python ds/evaluate_outbreak.py").read_text())
    fc = pd.read_csv(require(MODEL_DIR / "demand_forecast_metrics.csv",
                             "run python ds/train_forecast.py"))
    cohort = pd.read_csv(require(PROCESSED_DIR / "nhamcs_triage_features.csv",
                                 "run python ds/train_triage.py"))
    weekly = pd.read_csv(require(RAW_DIR / "district_weekly_cases.csv",
                                 "see ds/data/README.md for the covid19india archive"))

    assert len(cohort) == 13595, f"unexpected cohort n={len(cohort)}"
    assert len(weekly) == 126712, f"unexpected district-weeks={len(weekly)}"

    # ------------------------------------------------- T1: triage model (5-fold CV)
    t1 = metric.rename(columns={
        "Model": "Model", "AUC-ROC": "AUC", "Recall (Sensitivity)": "Recall",
        "Precision": "Precision", "F1 Score": "F1",
        "Under-Triage Rate": "Under-triage"})[
        ["Model", "AUC", "Recall", "Precision", "F1", "Under-triage", "Data Source"]]
    for c in ["AUC", "Recall", "Precision", "F1", "Under-triage"]:
        t1[c] = t1[c].map(r3)
    assert r3(float(metric.loc[metric.Model.str.startswith("XGBoost"),
                               "AUC-ROC"].iloc[0])) == 0.732, "XGB AUC claim drift"
    assert r3(float(metric.loc[metric.Model.str.startswith("Logistic"),
                               "AUC-ROC"].iloc[0])) == 0.719, "LogReg AUC claim drift"
    emit("paper_T1_triage_models", t1,
         "Triage model comparison, 5-fold cross-validation (t=0.5, seed 42).",
         "tab:triage_models", "lrrrrrr",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}.")

    # ----------------------------------------------- T2: operating points (abstain)
    x50 = trade[(trade.model == "xgboost") & (trade.threshold == 0.50)].iloc[0]
    x25 = op["xgboost"]
    l40 = op["logreg"]
    se = extras["xgboost_test"]
    assert abs(x25["expected_recall"] - trade[(trade.model == "xgboost") &
               (trade.threshold == 0.25)].iloc[0]["recall"]) < 0.001, "t=0.25 drift"
    assert (r3(x25["expected_recall"]), r3(x25["expected_under_triage"]),
            r3(x25["expected_precision"]), r3(x25["abstention_rate"])) == \
        (0.845, 0.155, 0.207, 0.235), "tuned operating-point claim drift"
    assert (r3(l40["expected_recall"]), r3(l40["threshold"])) == (0.811, 0.4), \
        "logreg tuned claim drift"
    assert r3(se["recall"]) == 0.913 and r3(se["brier"]) == 0.158, "seed-42 drift"
    rows = [
        {"Operating point": "XGBoost, t=0.50 (OOF)", "Threshold": 0.50,
         "Recall": r3(x50["recall"]), "Precision": r3(x50["precision"]),
         "Under-triage": r3(x50["under_triage"]),
         "Abstention": r3(x50["abstention_rate"]), "AUC": r3(0.7318710637)},
        {"Operating point": "XGBoost, t=0.25 tuned (OOF)", "Threshold": 0.25,
         "Recall": r3(x25["expected_recall"]), "Precision": r3(x25["expected_precision"]),
         "Under-triage": r3(x25["expected_under_triage"]),
         "Abstention": r3(x25["abstention_rate"]), "AUC": r3(0.7318710637)},
        {"Operating point": "LogReg, t=0.40 tuned (OOF)", "Threshold": 0.40,
         "Recall": r3(l40["expected_recall"]), "Precision": r3(l40["expected_precision"]),
         "Under-triage": r3(l40["expected_under_triage"]),
         "Abstention": r3(l40["abstention_rate"]), "AUC": r3(0.7193058568)},
        {"Operating point": "XGBoost, conformal, seed-42 test", "Threshold":
            r3(extras["split"]["threshold"]), "Recall": r3(se["recall"]),
         "Precision": r3(se["precision"]), "Under-triage": r3(1 - se["recall"]),
         "Abstention": r3(se["abstention_rate"]), "AUC": r3(se["auc"])},
    ]
    t2 = pd.DataFrame(rows)
    t2["Data Source"] = SRC_NHAMCS
    emit("paper_T2_operating_points", t2,
         "Operating points for the triage module: default, tuned (pre-declared "
         "recall $\\geq 0.80$ rule) and split-conformal (seed 42).",
         "tab:operating_points", "lrrrrrrr",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}. AUC is threshold-independent "
         "(5-fold CV); OOF = out-of-fold.")

    # --------------------------------------------------- T3: threshold sweep (suppl)
    keep = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90}
    t3 = trade[trade.threshold.isin(keep)].copy()
    t3["Model"] = t3["model"].map({"logreg": "Logistic regression",
                                   "xgboost": "XGBoost"})
    t3 = t3.rename(columns={"threshold": "Threshold", "recall": "Recall",
                            "precision": "Precision", "under_triage": "Under-triage",
                            "over_triage": "Over-triage",
                            "abstention_rate": "Abstention"})[
        ["Model", "Threshold", "Recall", "Precision", "Under-triage", "Over-triage",
         "Abstention", "Data Source"]]
    for c in ["Threshold", "Recall", "Precision", "Under-triage", "Over-triage",
              "Abstention"]:
        t3[c] = t3[c].map(r3)
    t3 = t3.sort_values(["Model", "Threshold"], kind="mergesort").reset_index(drop=True)
    assert len(t3) > 0, "threshold sweep empty"
    emit("paper_T3_threshold_sweep", t3,
         "Threshold sweep (OOF): recall$/$precision$/$triage rates and abstention "
         "as a function of the decision threshold.",
         "tab:threshold_sweep", "llrrrrrr",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}.")

    # ------------------------------------------------------- T4: conformal results
    main_alpha = 0.15
    mr = repeats["test_recall"]
    rec_mean, rec_sd = mr.mean(), mr.std(ddof=1)
    pct_below = (mr < 0.85).mean()
    assert r3(rec_mean) == 0.850, f"conformal mean recall drift: {rec_mean}"
    assert r3(pct_below) == 0.46, f"below-target share drift: {pct_below}"
    assert round(repeats["threshold"].mean(), 2) == 0.21, "mean threshold drift"
    assert round(repeats["test_abstention_rate"].mean(), 2) == 0.22, "mean abstention drift"

    claim_means = {0.05: 0.953, 0.10: 0.902, 0.15: 0.850, 0.20: 0.800, 0.30: 0.700}
    claim_precs = {0.05: 0.171, 0.10: 0.185, 0.15: 0.198, 0.20: 0.212, 0.30: 0.241}
    rows = []
    for a, g in sweep.groupby("alpha"):
        m, s = g["test_recall"].mean(), g["test_recall"].std(ddof=1)
        assert abs(m - claim_means[a]) < 0.001, f"sweep recall drift alpha={a}: {m}"
        assert abs(g["test_precision"].mean() - claim_precs[a]) < 0.001, \
            f"sweep precision drift alpha={a}"
        if abs(a - main_alpha) < 1e-9:
            assert abs(m - rec_mean) < 1e-6, "sweep/repeats alpha=.15 mismatch"
        rows.append({"alpha (miss target)": r3(a), "Mean recall": r3(m),
                     "SD": r3(s), "Min": r3(g["test_recall"].min()),
                     "Max": r3(g["test_recall"].max()),
                     "Frac. splits <0.85": r3((g["test_recall"] < 0.85).mean()),
                     "Mean precision": r3(g["test_precision"].mean())})
    t4 = pd.DataFrame(rows).sort_values("alpha (miss target)",
                                        kind="mergesort").reset_index(drop=True)
    t4["Data Source"] = SRC_NHAMCS
    assert len(t4) == 5 and len(repeats) == 200, "conformal table shape drift"
    emit("paper_T4_conformal", t4,
         "Split-conformal recall control over 200 random 60$/$20$/$20 splits "
         "(XGBoost, calibration quantile on positives). The guarantee is marginal: "
         "individual splits vary.",
         "tab:conformal", "lrrrrrrr",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}. Alpha 0.15 is the pre-declared "
         "main setting (also conformal_repeats.csv, 200 seeds).")

    # ------------------------------------- T5: seed-42 test + NEWS2 baseline (suppl)
    def ci_str(key):
        lo, hi = extras["bootstrap_95ci"][key]
        return f"[{lo:.3f}, {hi:.3f}]"

    news = extras["news2_baseline"]
    assert (r3(news["test_recall"]), r3(news["test_precision"])) == (0.138, 0.207), \
        "NEWS2 cutoff5 claim drift"
    assert (r3(news["cutoff1_test_recall"]), r3(news["cutoff1_test_precision"])) == \
        (0.669, 0.168), "NEWS2 cutoff1 claim drift"
    assert news["matched_recall_feasible"] is False, "NEWS2 feasibility drift"
    t5 = pd.DataFrame([
        {"Method": "XGBoost + conformal (seed-42 test)",
         "Recall": r3(se["recall"]), "Recall 95% CI": ci_str("recall"),
         "Precision": r3(se["precision"]), "Precision 95% CI": ci_str("precision"),
         "AUC": r3(se["auc"]), "AUC 95% CI": ci_str("auc"),
         "Brier": r3(se["brier"]), "Abstention": r3(se["abstention_rate"]),
         "Matches 0.85 target?": "Yes"},
        {"Method": "NEWS2 vitals-only, cutoff >=5",
         "Recall": r3(news["test_recall"]), "Recall 95% CI": "",
         "Precision": r3(news["test_precision"]), "Precision 95% CI": "",
         "AUC": "", "AUC 95% CI": "", "Brier": "", "Abstention": "",
         "Matches 0.85 target?": "No"},
        {"Method": "NEWS2 vitals-only, cutoff >=1 (max recall)",
         "Recall": r3(news["cutoff1_test_recall"]), "Recall 95% CI": "",
         "Precision": r3(news["cutoff1_test_precision"]), "Precision 95% CI": "",
         "AUC": "", "AUC 95% CI": "", "Brier": "", "Abstention": "",
         "Matches 0.85 target?": "No"},
    ])
    t5["Data Source"] = SRC_NHAMCS
    emit("paper_T5_severity_baseline", t5,
         "Seed-42 split-conformal test performance (bootstrap 95\\% CIs, 1000 "
         "resamples) versus the NEWS2 vitals-only baseline. No NEWS2 cutoff "
         "$\\geq$1 reaches 0.85 recall on calibration, so a matched comparison "
         "is infeasible (documented).",
         "tab:severity_baseline", "lrrrrrrrrlc",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}.")

    # ------------------------------------------------------- T6: outbreak methods
    rows = []
    for m in outbreak["methods"]:
        bg = outbreak["methods"][m]["background"]["background_rate"]
        for w in ["delta", "omicron"]:
            e = outbreak["methods"][m][w]
            rows.append({"Method": m, "Window": w.capitalize(),
                         "In-window flag rate": r3(e["flag_rate_in_window"]),
                         "District coverage": r3(e["district_coverage"]),
                         "Median lead (weeks)": r3(e["median_lead_weeks"]),
                         "Background rate": r3(bg)})
    t6 = pd.DataFrame(rows)
    t6["Data Source"] = SRC_COVID
    iso_d = outbreak["methods"]["iso_single"]["delta"]
    assert (r3(iso_d["flag_rate_in_window"]), r3(iso_d["district_coverage"])) == \
        (0.250, 0.398), "iso_single delta claim drift"
    zscore_bg = outbreak["methods"]["zscore"]["background"]["background_rate"]
    assert r3(zscore_bg) == 0.093, "zscore background drift"
    enrich = delta_ev["flag_rate_in_window"] / delta_ev["flag_rate_outside_window"]
    assert r3(delta_ev["flag_rate_in_window"]) == 0.250 and \
        r3(delta_ev["flag_rate_outside_window"]) == 0.036, "delta rate drift"
    assert round(enrich, 1) == 7.0, f"enrichment drift: {enrich}"
    assert outbreak["n_districts"] == 840, "district count drift"
    emit("paper_T6_outbreak", t6,
         "Outbreak detection methods versus pre-declared epidemic windows "
         "(descriptive; no labeled ground truth exists). Background rate = "
         "flag share of district-weeks outside both windows. IsolationForest "
         "single-week flags enrich the Delta window 7.0$\\times$ "
         "(0.250 inside vs 0.036 outside).",
         "tab:outbreak", "llrrrrr",
         f"\\textbf{{Data source:}} {SRC_COVID}; {outbreak['n_districts']} "
         "state-qualified districts, 126,712 district-weeks.")

    # ---------------------------------------------------- T7: cohort + datasets
    def med_iqr(s: pd.Series) -> str:
        s = s.dropna()
        return f"{s.median():.0f} [{s.quantile(0.25):.0f}, {s.quantile(0.75):.0f}]"

    prev = cohort["is_high_severity"].mean()
    pct_female = cohort["is_female"].mean() if "is_female" in cohort else np.nan
    weeks = pd.to_datetime(weekly["week"])
    rows = [
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Raw records",
         "Value": f"{RAW_NHAMCS_N:,}", "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Analysis records",
         "Value": f"{len(cohort):,}", "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "High-severity prevalence",
         "Value": f"{prev:.3f}", "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Female share",
         "Value": f"{pct_female:.3f}", "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Age, median [Q1, Q3]",
         "Value": med_iqr(cohort["age"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Pulse, median [Q1, Q3]",
         "Value": med_iqr(cohort["pulse"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Resp. rate, median [Q1, Q3]",
         "Value": med_iqr(cohort["respr"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Systolic BP, median [Q1, Q3]",
         "Value": med_iqr(cohort["bpsys"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Diastolic BP, median [Q1, Q3]",
         "Value": med_iqr(cohort["bpdias"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Temperature (F), median [Q1, Q3]",
         "Value": med_iqr(cohort["tempf"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "NHAMCS 2019 ED", "Statistic": "Pain scale, median [Q1, Q3]",
         "Value": med_iqr(cohort["pain_scale"]), "Data Source": SRC_NHAMCS},
        {"Dataset": "Pharma sales (suppl.)", "Statistic": "Months analyzed",
         "Value": "69 (2014-01 – 2019-09)", "Data Source": SRC_KAGGLE},
        {"Dataset": "COVID-19 India", "Statistic": "District-weeks",
         "Value": f"{len(weekly):,}", "Data Source": SRC_COVID},
        {"Dataset": "COVID-19 India", "Statistic": "Districts",
         "Value": str(outbreak["n_districts"]), "Data Source": SRC_COVID},
        {"Dataset": "COVID-19 India", "Statistic": "Weeks (starting)",
         "Value": f"{weeks.min():%Y-%m-%d} – {weeks.max():%Y-%m-%d}",
         "Data Source": SRC_COVID},
    ]
    t7 = pd.DataFrame(rows)
    assert round(prev, 3) == 0.152, f"prevalence drift: {prev}"
    emit("paper_T7_cohort", t7,
         "Cohort and dataset summary for both modules plus the supplementary "
         "forecast. Vitals are median [Q1, Q3] on the analysis cohort.",
         "tab:cohort", "lllc",
         f"\\textbf{{Data sources:}} {SRC_NHAMCS}; {SRC_KAGGLE}; {SRC_COVID}.")

    # --------------------------------------------------- TS1: demand forecast MAE
    best = fc[["MAE Seasonal Naive", "MAE Holt-Winters",
               "MAE LightGBM"]].idxmin(axis=1).str.replace("MAE ", "", regex=False)
    ts1 = fc.rename(columns={"Drug (ATC)": "Drug",
                             "MAE Seasonal Naive": "Seasonal naive",
                             "MAE Holt-Winters": "Holt-Winters",
                             "MAE LightGBM": "LightGBM"})
    ts1["Best"] = best.values
    ts1 = ts1[["Drug", "Seasonal naive", "Holt-Winters", "LightGBM", "Best",
               "Data Source"]]
    for c in ["Seasonal naive", "Holt-Winters", "LightGBM"]:
        ts1[c] = ts1[c].map(r3)
    tally = best.value_counts().to_dict()
    assert (tally.get("Holt-Winters", 0), tally.get("Seasonal Naive", 0),
            tally.get("LightGBM", 0)) == (4, 2, 2), f"forecast tally drift: {tally}"
    assert len(ts1) == 8, "forecast drug count drift"
    emit("paper_TS1_forecast", ts1,
         "Monthly demand forecast, MAE on the 12-step holdout (supplementary). "
         "Holt-Winters wins 4/8, seasonal naive 2/8, LightGBM 2/8 — naive "
         "baselines are competitive.",
         "tab:forecast", "lrrrlr",
         f"\\textbf{{Data source:}} {SRC_KAGGLE}.")

    # ------------------------------------------------------------ paper_numbers.json
    numbers = {
        "module1_triage": {
            "n_analysis": int(len(cohort)),
            "n_raw": RAW_NHAMCS_N,
            "prevalence": r3(prev),
            "auc_xgb_cv": r3(metric.loc[metric.Model.str.startswith("XGBoost"),
                                        "AUC-ROC"].iloc[0]),
            "auc_logreg_cv": r3(metric.loc[metric.Model.str.startswith("Logistic"),
                                            "AUC-ROC"].iloc[0]),
            "xgb_cv_t05_recall": r3(metric.loc[metric.Model.str.startswith("XGBoost"),
                                               "Recall (Sensitivity)"].iloc[0]),
            "xgb_cv_t05_under_triage": r3(metric.loc[metric.Model.str.startswith(
                "XGBoost"), "Under-Triage Rate"].iloc[0]),
            "tuned_xgb_threshold": r3(x25["threshold"]),
            "tuned_xgb_recall": r3(x25["expected_recall"]),
            "tuned_xgb_precision": r3(x25["expected_precision"]),
            "tuned_xgb_under_triage": r3(x25["expected_under_triage"]),
            "tuned_xgb_abstention": r3(x25["abstention_rate"]),
            "conformal_alpha": 0.15,
            "conformal_mean_recall": r3(rec_mean),
            "conformal_sd_recall": r3(rec_sd),
            "conformal_frac_below_085": r3(pct_below),
            "conformal_mean_threshold": r3(repeats["threshold"].mean()),
            "conformal_mean_abstention": r3(repeats["test_abstention_rate"].mean()),
            "conformal_n_seeds": int(len(repeats)),
            "seed42_threshold": r3(extras["split"]["threshold"]),
            "seed42_recall": r3(se["recall"]),
            "seed42_recall_ci95": [r3(v) for v in extras["bootstrap_95ci"]["recall"]],
            "seed42_precision": r3(se["precision"]),
            "seed42_auc": r3(se["auc"]),
            "seed42_brier": r3(se["brier"]),
            "seed42_abstention": r3(se["abstention_rate"]),
            "news2_cutoff5_recall": r3(news["test_recall"]),
            "news2_cutoff5_precision": r3(news["test_precision"]),
            "news2_cutoff1_recall": r3(news["cutoff1_test_recall"]),
            "news2_cutoff1_precision": r3(news["cutoff1_test_precision"]),
            "news2_matches_085": False,
        },
        "module2_outbreak": {
            "district_weeks": int(len(weekly)),
            "districts": int(outbreak["n_districts"]),
            "iso_single_delta_flag_rate": r3(iso_d["flag_rate_in_window"]),
            "iso_single_delta_coverage": r3(iso_d["district_coverage"]),
            "iso_single_delta_lead_weeks": r3(iso_d["median_lead_weeks"]),
            "iso_single_background_rate": r3(outbreak["methods"]["iso_single"]
                                             ["background"]["background_rate"]),
            "iso_confirmed_background_rate": r3(outbreak["methods"]["iso_confirmed"]
                                                ["background"]["background_rate"]),
            "zscore_background_rate": r3(zscore_bg),
            "delta_enrichment": r3(enrich),
            "delta_outside_flag_rate": r3(delta_ev["flag_rate_outside_window"]),
        },
        "supplement_forecast": {
            "months_analyzed": 69,
            "hw_wins": int(tally.get("Holt-Winters", 0)),
            "naive_wins": int(tally.get("Seasonal naive", 0)),
            "lgbm_wins": int(tally.get("LightGBM", 0)),
        },
    }
    ppath = PAPER_DIR / "paper_numbers.json"
    ppath.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8", newline="\n")
    assert json.loads(ppath.read_text()) == numbers, "paper_numbers round-trip"
    print(f"wrote {ppath.name}")

    # ------------------------------------------------------------- captions.md
    caps = f"""# Paper captions (figures + tables)

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
5-fold cross-validation ({numbers['module1_triage']['n_analysis']:,} analysis
encounters): LogReg AUC {numbers['module1_triage']['auc_logreg_cv']:.3f},
XGBoost AUC {numbers['module1_triage']['auc_xgb_cv']:.3f}. Marked operating
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
{numbers['module1_triage']['conformal_mean_recall']:.3f} vs the 0.85 target;
{100 * numbers['module1_triage']['conformal_frac_below_085']:.0f}% of individual
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
{numbers['module1_triage']['seed42_brier']:.4f}): the model is overconfident;
probabilities are rank-useful for the threshold rule, not quoted as risks.
Data: CDC NHAMCS 2019 ED.

## Tables (`ds/models/paper/paper_T*.csv` + `.tex`)

**Table I (T1, label `tab:triage_models`)** — Triage model comparison, 5-fold
CV (t=0.5): XGBoost AUC {numbers['module1_triage']['auc_xgb_cv']:.3f} vs LogReg
{numbers['module1_triage']['auc_logreg_cv']:.3f}; XGB under-triage
{numbers['module1_triage']['xgb_cv_t05_under_triage']:.3f} at default
threshold. Data: CDC NHAMCS 2019 ED.

**Table II (T2, label `tab:operating_points`)** — Operating points: default
t=0.50, tuned t=0.25 (recall {numbers['module1_triage']['tuned_xgb_recall']:.3f},
under-triage {numbers['module1_triage']['tuned_xgb_under_triage']:.3f}),
LogReg t=0.40, and conformal seed-42 (recall
{numbers['module1_triage']['seed42_recall']:.3f}). Data: CDC NHAMCS 2019 ED.

**Table III (T3, label `tab:threshold_sweep`)** — OOF threshold sweep for both
models (supplementary). Data: CDC NHAMCS 2019 ED.

**Table IV (T4, label `tab:conformal`)** — Conformal recall across 200 splits
per alpha; mean recall
{numbers['module1_triage']['conformal_mean_recall']:.3f} at alpha=0.15 with
SD {numbers['module1_triage']['conformal_sd_recall']:.3f}; fraction of splits
below 0.85 = {numbers['module1_triage']['conformal_frac_below_085']:.3f}.
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
"""
    cpath = DOCS_DIR / "captions.md"
    cpath.write_text(caps, encoding="utf-8", newline="\n")
    print(f"wrote {cpath}")

    # ------------------------------------------------------------------ summary
    files = sorted(PAPER_DIR.glob("paper_T*")) + [ppath, cpath]
    print("ALL PAPER TABLES DONE:", len(files), "files")


if __name__ == "__main__":
    main()
