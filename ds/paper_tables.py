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
SRC_NHAMCS_SHORT = "CDC NHAMCS 2019 ED"
SRC_KAGGLE = "Kaggle milanzdravkovic/pharma-sales-data (CC BY-NC 4.0)"
SRC_COVID = "covid19india static archive (data.incovid19.org, retrieved 2026-10-07)"
SRC_COVID_SHORT = "covid19india static archive"
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
    s = str(s)
    # cells are authored LaTeX: pass math/LaTeX through untouched; escape
    # only characters that break text-mode table cells
    if "$" in s or "\\" in s:
        return s
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def booktabs(df: pd.DataFrame, caption: str, label: str, colspec: str,
             note: str = "", span: bool = False) -> str:
    head = " & ".join(esc(str(c)) for c in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(esc(str(v)) for v in row.values) + r" \\")
    note_tex = f"\n\\par\\vspace{{2pt}}\\raggedright\\footnotesize {note}" if note else ""
    env, close = ("table*", "table*") if span else ("table", "table")
    body = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
    body += "\n".join(rows)
    body += "\n\\bottomrule\n\\end{tabular}"
    if span:
        # two-column span; shrink only if wider than \textwidth (never stretch)
        body = ("\\begin{adjustbox}{max width=0.98\\textwidth}\n"
                + body + "\n\\end{adjustbox}")
    return (
        f"\\begin{{{env}}}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        + body + note_tex + f"\n\\end{{{env}}}\n"
    )


def emit(name: str, df: pd.DataFrame, caption: str, label: str, colspec: str,
         note: str = "", span: bool = False) -> None:
    csv_path = PAPER_DIR / f"{name}.csv"
    tex_path = PAPER_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    tex_path.write_text(booktabs(df, caption, label, colspec, note, span),
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
    oof_stats = json.loads(require(MODEL_DIR / "triage_oof_stats.json",
                                   "run python ds/train_triage.py").read_text())
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
    def mt_row(label: str):
        m = metric[metric.Model == label]
        assert len(m) == 1, f"metric_table row missing/duplicated: {label}"
        return m.iloc[0]

    t1 = metric.rename(columns={
        "Model": "Model", "AUC-ROC": "AUC", "Recall (Sensitivity)": "Recall",
        "Precision": "Precision", "F1 Score": "F1",
        "Under-Triage Rate": "Under-triage", "Brier (OOF)": "Brier"})[
        ["Model", "AUC", "Recall", "Precision", "F1", "Under-triage", "Brier",
         "Data Source"]]
    for c in ["AUC", "Recall", "Precision", "F1", "Under-triage", "Brier"]:
        t1[c] = t1[c].map(r3)
    t1["Data Source"] = SRC_NHAMCS_SHORT
    # claim locks (5-fold CV, seed 42)
    assert r3(mt_row("XGBoost (class-weighted)")["AUC-ROC"]) == 0.732, "XGB weighted AUC drift"
    assert r3(mt_row("Logistic Regression (baseline)")["AUC-ROC"]) == 0.719, "LogReg AUC drift"
    assert r3(mt_row("XGBoost (unweighted)")["AUC-ROC"]) == 0.743, "XGB unweighted AUC drift"
    assert r3(mt_row("XGBoost (unweighted + isotonic)")["AUC-ROC"]) == 0.742, "XGB primary AUC drift"
    assert r3(mt_row("XGBoost (unweighted + isotonic)")["Brier (OOF)"]) == 0.113, "primary Brier drift"
    assert r3(mt_row("Constant (train prevalence)")["Brier (OOF)"]) == 0.129, "constant Brier drift"
    assert r3(mt_row("XGBoost (class-weighted)")["Brier (OOF)"]) == 0.165, "weighted Brier drift"
    emit("paper_T1_triage_models", t1,
         "Triage model comparison, 5-fold cross-validation ($t=0.5$, seed 42), "
         "with out-of-fold Brier scores. The constant row predicts each "
         "training fold's prevalence; it is the Brier reference the calibrated "
         "primary model must beat.",
         "tab:triage_models", "lrrrrrrc",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}.", span=True)

    # ----------------------------------------------- T2: operating points (abstain)
    x50 = trade[(trade.model == "xgboost") & (trade.threshold == 0.50)].iloc[0]
    x25 = op["xgboost"]
    l40 = op["logreg"]
    nx = op["nested_xgboost"]
    nxp = nx["pooled"]
    se = extras["xgboost_test"]
    auc_pooled = r3(oof_stats["oof_auc"]["xgb_unw_isotonic"])
    auc_logreg_pooled = r3(oof_stats["oof_auc"]["logreg"])
    # claim locks: pooled tuned point, nested cross-fitted headline, logreg
    assert (r3(x25["threshold"]), r3(x25["expected_recall"]),
            r3(x25["expected_precision"]), r3(x25["expected_fpr"]),
            r3(x25["expected_under_triage"])) == (0.11, 0.8, 0.226, 0.493, 0.2), \
        "tuned operating-point claim drift"
    assert (r3(nxp["recall"]), r3(nxp["precision"]), r3(nxp["fpr"]),
            r3(nxp["under_triage"])) == (0.829, 0.214, 0.545, 0.172), \
        "nested claim drift"
    assert nx["thresholds"] == [0.09, 0.1, 0.1, 0.1, 0.09], "nested t drift"
    assert (r3(l40["expected_recall"]), r3(l40["threshold"])) == (0.811, 0.4), \
        "logreg tuned claim drift"
    assert (r3(se["recall"]), r3(se["precision"]), r3(se["fpr"])) == \
        (0.886, 0.199, 0.643), "seed-42 claim drift"
    nx_t = nx["thresholds"]
    nx_t_str = f"{np.mean(nx_t):.3f} [{min(nx_t):.2f}, {max(nx_t):.2f}]"
    rows = [
        {"Operating point": "Primary, t=0.50 (pooled OOF)", "Threshold": 0.50,
         "Recall": r3(x50["recall"]), "Precision": r3(x50["precision"]),
         "FPR": r3(x50["fpr"]),
         "Under-triage": r3(x50["under_triage"]),
         "Abstention": r3(x50["abstention_rate"]), "AUC": auc_pooled},
        {"Operating point": "Primary, tuned (pooled OOF)",
         "Threshold": r3(x25["threshold"]),
         "Recall": r3(x25["expected_recall"]), "Precision": r3(x25["expected_precision"]),
         "FPR": r3(x25["expected_fpr"]),
         "Under-triage": r3(x25["expected_under_triage"]),
         "Abstention": r3(x25["abstention_rate"]), "AUC": auc_pooled},
        {"Operating point": "Primary, cross-fitted (headline)",
         "Threshold": nx_t_str,
         "Recall": r3(nxp["recall"]), "Precision": r3(nxp["precision"]),
         "FPR": r3(nxp["fpr"]), "Under-triage": r3(nxp["under_triage"]),
         "Abstention": r3(nxp["abstention_rate"]), "AUC": r3(nxp["auc"])},
        {"Operating point": "LogReg, tuned (pooled OOF)",
         "Threshold": r3(l40["threshold"]),
         "Recall": r3(l40["expected_recall"]), "Precision": r3(l40["expected_precision"]),
         "FPR": r3(l40["expected_fpr"]),
         "Under-triage": r3(l40["expected_under_triage"]),
         "Abstention": r3(l40["abstention_rate"]), "AUC": auc_logreg_pooled},
        {"Operating point": "Primary, conformal, seed-42 test",
         "Threshold": r3(extras["split"]["threshold"]), "Recall": r3(se["recall"]),
         "Precision": r3(se["precision"]), "FPR": r3(se["fpr"]),
         "Under-triage": r3(1 - se["recall"]),
         "Abstention": r3(se["abstention_rate"]), "AUC": r3(se["auc"])},
    ]
    t2 = pd.DataFrame(rows)
    t2["Data Source"] = SRC_NHAMCS_SHORT
    emit("paper_T2_operating_points", t2,
         "Operating points for the triage module: default, tuned "
         "(pre-declared rule: max precision among thresholds with OOF recall "
         "$\\geq 0.80$), the nested cross-fitted evaluation, and "
         "split-conformal (seed 42).",
         "tab:operating_points", "llrrrrrrc",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}. AUC is "
         "threshold-independent; OOF = out-of-fold; cross-fitted rows tune on "
         "inner folds and evaluate on held-out outer folds.", span=True)

    # --------------------------------------------------- T3: threshold sweep (suppl)
    keep = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90}
    t3 = trade[trade.threshold.isin(keep)].copy()
    t3["Model"] = t3["model"].map({"logreg": "Logistic regression",
                                   "xgboost": "Primary (XGB+isotonic)"})
    t3 = t3.rename(columns={"threshold": "Threshold", "recall": "Recall",
                            "precision": "Precision", "fpr": "FPR",
                            "under_triage": "Under-triage",
                            "over_triage": "Over-triage",
                            "abstention_rate": "Abstention"})[
        ["Model", "Threshold", "Recall", "Precision", "FPR", "Under-triage",
         "Over-triage", "Abstention", "Data Source"]]
    for c in ["Threshold", "Recall", "Precision", "FPR", "Under-triage",
              "Over-triage", "Abstention"]:
        t3[c] = t3[c].map(r3)
    t3 = t3.sort_values(["Model", "Threshold"], kind="mergesort").reset_index(drop=True)
    assert len(t3) > 0, "threshold sweep empty"
    emit("paper_T3_threshold_sweep", t3,
         "Threshold sweep (OOF): recall$/$precision$/$FPR$/$triage rates and "
         "abstention as a function of the decision threshold.",
         "tab:threshold_sweep", "lrrrrrrrc",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}.")

    # ------------------------------------------------------- T4: conformal results
    main_alpha = 0.15
    from conformal_triage import PAC_DELTA
    mr = repeats["test_recall"]
    pr_ = repeats["pac_test_recall"]
    rec_mean, rec_sd = mr.mean(), mr.std(ddof=1)
    pac_mean, pac_sd = pr_.mean(), pr_.std(ddof=1)
    pct_below = (mr < 0.85).mean()
    pac_pct_below = (pr_ < 0.85).mean()
    # claim locks (alpha=0.15, 200 seeds, primary model)
    assert (r3(rec_mean), r3(rec_sd), r3(pct_below)) == \
        (0.884, 0.033, 0.125), f"marginal claim drift: {r3(rec_mean)}"
    assert (r3(pac_mean), r3(pac_sd), r3(pac_pct_below)) == \
        (0.9, 0.03, 0.035), f"PAC claim drift: {r3(pac_mean)}"
    assert pac_pct_below <= PAC_DELTA, "PAC delta violated empirically"
    assert r3(repeats["threshold"].mean()) == 0.094, "mean threshold drift"
    assert r3(repeats["test_abstention_rate"].mean()) == 0.043, "mean abstention drift"
    assert r3(repeats["pac_threshold"].mean()) == 0.089, "PAC mean threshold drift"
    assert r3(repeats["test_precision"].mean()) == 0.196, "marginal precision drift"
    assert r3(repeats["pac_test_precision"].mean()) == 0.191, \
        "PAC precision drift"

    claim_marg_means = {0.05: 0.966, 0.10: 0.924, 0.15: 0.884, 0.20: 0.832, 0.30: 0.737}
    claim_marg_precs = {0.05: 0.169, 0.10: 0.183, 0.15: 0.196, 0.20: 0.212, 0.30: 0.243}
    claim_pac_means = {0.05: 0.976, 0.10: 0.938, 0.15: 0.9, 0.20: 0.858, 0.30: 0.768}
    claim_pac_precs = {0.05: 0.164, 0.10: 0.179, 0.15: 0.191, 0.20: 0.204, 0.30: 0.233}
    rows = []
    for a, g in sweep.groupby("alpha"):
        target = 1.0 - a
        m, s = g["test_recall"].mean(), g["test_recall"].std(ddof=1)
        pm, ps = g["pac_test_recall"].mean(), g["pac_test_recall"].std(ddof=1)
        frac_m = (g["test_recall"] < target).mean()
        frac_p = (g["pac_test_recall"] < target).mean()
        assert r3(m) == claim_marg_means[a], f"sweep marginal recall drift alpha={a}: {r3(m)}"
        assert r3(g["test_precision"].mean()) == claim_marg_precs[a], \
            f"sweep marginal precision drift alpha={a}"
        assert r3(pm) == claim_pac_means[a], f"sweep PAC recall drift alpha={a}: {r3(pm)}"
        assert r3(g["pac_test_precision"].mean()) == claim_pac_precs[a], \
            f"sweep PAC precision drift alpha={a}"
        assert frac_p <= PAC_DELTA + 1e-9, f"PAC below-target share > delta at alpha={a}"
        if abs(a - main_alpha) < 1e-9:
            assert abs(m - rec_mean) < 1e-6, "sweep/repeats alpha=.15 mismatch"
            assert abs(pm - pac_mean) < 1e-6, "sweep/repeats PAC alpha=.15 mismatch"
        rows.append({"alpha (miss target)": r3(a), "Rule": "marginal",
                     "Mean recall": r3(m), "SD": r3(s),
                     "Frac. below $1-\\alpha$": r3(frac_m),
                     "Mean precision": r3(g["test_precision"].mean())})
        rows.append({"alpha (miss target)": r3(a),
                     "Rule": f"PAC ($\\delta$={PAC_DELTA})",
                     "Mean recall": r3(pm), "SD": r3(ps),
                     "Frac. below $1-\\alpha$": r3(frac_p),
                     "Mean precision": r3(g["pac_test_precision"].mean())})
    t4 = pd.DataFrame(rows).sort_values(
        ["alpha (miss target)", "Rule"],
        key=lambda c: c.map({"marginal": 0}) if c.name == "Rule" else c,
        kind="mergesort").reset_index(drop=True)
    t4["Data Source"] = SRC_NHAMCS
    assert len(t4) == 10 and len(repeats) == 200, "conformal table shape drift"
    emit("paper_T4_conformal", t4,
         "Split-conformal recall control over 200 random 60$/$20$/$20 splits "
         "(primary XGBoost$+$isotonic; calibration quantile on positives). "
         "Marginal: textbook $k=\\lfloor\\alpha(n+1)\\rfloor$ rule, average "
         "guarantee. PAC: Beta order-statistic rank at "
         f"$\\delta$={PAC_DELTA}, bounding the below-target share per split.",
         "tab:conformal", "llrrrrc",
         f"\\textbf{{Data source:}} {SRC_NHAMCS}. Alpha 0.15 is the "
         "pre-declared main setting (also conformal\\_repeats.csv, 200 seeds).",
         span=True)

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
    # seed-42 primary claim locks (marginal conformal at alpha=0.15)
    assert (r3(se["recall"]), r3(se["precision"]), r3(se["fpr"]),
            r3(se["auc"]), r3(se["brier"]),
            r3(se["brier_constant_trainprev"])) == \
        (0.886, 0.199, 0.643, 0.755, 0.112, 0.129), "seed-42 extras drift"
    t5 = pd.DataFrame([
        {"Method": "Primary + conformal (seed-42 test)",
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
         "is infeasible (documented). Brier of the calibrated primary model: "
         f"{r3(se['brier'])} vs {r3(se['brier_constant_trainprev'])} for a "
         "constant predictor at the train-split prevalence.",
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
    t6["Data Source"] = SRC_COVID_SHORT
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
         "state-qualified districts, 126,712 district-weeks.", span=True)

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
    from triage_models import LABELS as M_LABELS, PRIMARY
    pri = mt_row(M_LABELS[PRIMARY])
    wtd = mt_row(M_LABELS["xgb_weighted"])
    con = mt_row(M_LABELS["constant"])
    x50f = trade[(trade.model == "xgboost") & (trade.threshold == 0.50)].iloc[0]
    numbers = {
        "module1_triage": {
            "n_analysis": int(len(cohort)),
            "n_raw": RAW_NHAMCS_N,
            "prevalence": r3(prev),
            "auc_xgb_cv": r3(pri["AUC-ROC"]),
            "auc_xgb_weighted_cv": r3(wtd["AUC-ROC"]),
            "auc_logreg_cv": r3(mt_row(M_LABELS["logreg"])["AUC-ROC"]),
            "auc_xgb_ci95": [r3(v) for v in oof_stats["auc_ci95"]["xgb_unw_isotonic"]],
            "auc_logreg_ci95": [r3(v) for v in oof_stats["auc_ci95"]["logreg"]],
            "auc_diff_ci95": [r3(v) for v in
                              oof_stats["auc_diff_primary_minus_logreg_ci95"]],
            "xgb_cv_t05_recall": r3(pri["Recall (Sensitivity)"]),
            "xgb_cv_t05_under_triage": r3(pri["Under-Triage Rate"]),
            "xgb_cv_t05_fpr": r3(x50f["fpr"]),
            "brier_primary_oof": r3(pri["Brier (OOF)"]),
            "brier_weighted_oof": r3(wtd["Brier (OOF)"]),
            "brier_constant_oof": r3(con["Brier (OOF)"]),
            "brier_isotonic_oof": round(float(pri["Brier (OOF)"]), 4),
            "brier_platt_oof": round(float(mt_row(M_LABELS["xgb_unw_platt"])
                                           ["Brier (OOF)"]), 4),
            "tuned_xgb_threshold": r3(x25["threshold"]),
            "tuned_xgb_recall": r3(x25["expected_recall"]),
            "tuned_xgb_precision": r3(x25["expected_precision"]),
            "tuned_xgb_fpr": r3(x25["expected_fpr"]),
            "tuned_xgb_under_triage": r3(x25["expected_under_triage"]),
            "tuned_xgb_abstention": r3(x25["abstention_rate"]),
            "nested_threshold_mean": r3(float(np.mean(nx_t))),
            "nested_threshold_min": r3(min(nx_t)),
            "nested_threshold_max": r3(max(nx_t)),
            "nested_recall": r3(nxp["recall"]),
            "nested_precision": r3(nxp["precision"]),
            "nested_fpr": r3(nxp["fpr"]),
            "nested_under_triage": r3(nxp["under_triage"]),
            "nested_abstention": r3(nxp["abstention_rate"]),
            "conformal_alpha": 0.15,
            "conformal_mean_recall": r3(rec_mean),
            "conformal_sd_recall": r3(rec_sd),
            "conformal_frac_below_085": r3(pct_below),
            "conformal_mean_threshold": r3(repeats["threshold"].mean()),
            "conformal_mean_abstention": r3(repeats["test_abstention_rate"].mean()),
            "conformal_mean_precision": r3(repeats["test_precision"].mean()),
            "conformal_n_seeds": int(len(repeats)),
            "pac_delta": PAC_DELTA,
            "pac_mean_recall": r3(pac_mean),
            "pac_sd_recall": r3(pac_sd),
            "pac_frac_below_085": r3(pac_pct_below),
            "pac_mean_threshold": r3(repeats["pac_threshold"].mean()),
            "pac_mean_precision": r3(repeats["pac_test_precision"].mean()),
            "seed42_threshold": r3(extras["split"]["threshold"]),
            "seed42_recall": r3(se["recall"]),
            "seed42_recall_ci95": [r3(v) for v in extras["bootstrap_95ci"]["recall"]],
            "seed42_precision": r3(se["precision"]),
            "seed42_fpr": r3(se["fpr"]),
            "seed42_auc": r3(se["auc"]),
            "seed42_brier": r3(se["brier"]),
            "seed42_brier_constant": r3(se["brier_constant_trainprev"]),
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
            "naive_wins": int(tally.get("Seasonal Naive", 0)),
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
primary XGBoost+isotonic AUC
{numbers['module1_triage']['auc_xgb_cv']:.3f} (95% CI
{numbers['module1_triage']['auc_xgb_ci95'][0]:.3f}--{numbers['module1_triage']['auc_xgb_ci95'][1]:.3f}).
Marked operating points: t=0.50 (FPR {numbers['module1_triage']['xgb_cv_t05_fpr']:.3f},
TPR {numbers['module1_triage']['xgb_cv_t05_recall']:.3f}), tuned t={numbers['module1_triage']['tuned_xgb_threshold']:.2f}
({numbers['module1_triage']['tuned_xgb_fpr']:.3f},
{numbers['module1_triage']['tuned_xgb_recall']:.3f}), and the seed-42
conformal split ({numbers['module1_triage']['seed42_fpr']:.3f},
{numbers['module1_triage']['seed42_recall']:.3f}). Data: CDC NHAMCS 2019 ED.

**Fig. 3 (fig2, `paper_fig2_shap.png`, label `fig:shap`)** — SHAP summary for
the XGBoost triage model: age, pulse, respiratory rate and pain scale dominate;
reason-for-visit codes contribute through clinical-pattern splits. Data: CDC
NHAMCS 2019 ED.

**Fig. 4 (fig3, `paper_fig3_tradeoff.png`, label `fig:tradeoff`)** — OOF
threshold trade-offs (recall, precision, FPR, under-/over-triage, abstention)
with the pre-declared selections t=0.40 (LogReg) and
t={numbers['module1_triage']['tuned_xgb_threshold']:.2f} (primary pooled).
Data: CDC NHAMCS 2019 ED.

**Fig. 5 (fig4, `paper_fig4_conformal_hist.png`, label `fig:conformal`)** —
Split-conformal test recall over 200 seeds at alpha=0.15: marginal rule mean
{numbers['module1_triage']['conformal_mean_recall']:.3f} vs the 0.85 target,
with {100 * numbers['module1_triage']['conformal_frac_below_085']:.1f}% of
individual splits below target; the PAC rule (delta
{numbers['module1_triage']['pac_delta']:g}) raises the mean to
{numbers['module1_triage']['pac_mean_recall']:.3f} and cuts the below-target
share to {100 * numbers['module1_triage']['pac_frac_below_085']:.1f}%. Data: CDC
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
{numbers['module1_triage']['seed42_brier']:.4f} vs
{numbers['module1_triage']['seed42_brier_constant']:.4f} for a constant at the
train-split prevalence): isotonic calibration improves on the constant
baseline; probabilities drive the threshold rule, not quoted as risks. Data:
CDC NHAMCS 2019 ED.

## Tables (`ds/models/paper/paper_T*.csv` + `.tex`)

**Table I (T1, label `tab:triage_models`)** — Triage model comparison, 5-fold
CV (t=0.5): primary XGBoost+isotonic AUC
{numbers['module1_triage']['auc_xgb_cv']:.3f} vs LogReg
{numbers['module1_triage']['auc_logreg_cv']:.3f}; OOF Brier
{numbers['module1_triage']['brier_primary_oof']:.3f} vs
{numbers['module1_triage']['brier_constant_oof']:.3f} for a constant predictor;
under-triage {numbers['module1_triage']['xgb_cv_t05_under_triage']:.3f} at the
default threshold. Data: CDC NHAMCS 2019 ED.

**Table II (T2, label `tab:operating_points`)** — Operating points: default
t=0.50, tuned t={numbers['module1_triage']['tuned_xgb_threshold']:.2f} (recall
{numbers['module1_triage']['tuned_xgb_recall']:.3f}, FPR
{numbers['module1_triage']['tuned_xgb_fpr']:.3f}), cross-fitted nested
(recall {numbers['module1_triage']['nested_recall']:.3f}, FPR
{numbers['module1_triage']['nested_fpr']:.3f}), LogReg t=0.40, and conformal
seed-42 (recall {numbers['module1_triage']['seed42_recall']:.3f}). Data: CDC
NHAMCS 2019 ED.

**Table III (T3, label `tab:threshold_sweep`)** — OOF threshold sweep for both
models (supplementary). Data: CDC NHAMCS 2019 ED.

**Table IV (T4, label `tab:conformal`)** — Conformal recall across 200 splits
per alpha, marginal vs PAC rules; at alpha=0.15 marginal mean
{numbers['module1_triage']['conformal_mean_recall']:.3f} (SD
{numbers['module1_triage']['conformal_sd_recall']:.3f}) with
{100 * numbers['module1_triage']['conformal_frac_below_085']:.1f}% of splits
below target vs PAC mean {numbers['module1_triage']['pac_mean_recall']:.3f}
with {100 * numbers['module1_triage']['pac_frac_below_085']:.1f}% below. Data:
CDC NHAMCS 2019 ED.

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
