"""
ds/paper_figures.py — IEEE paper figures at 300 DPI (new files only).

Reads committed artifacts; originals untouched. New computation ONLY:
  Fig 1 ROC curves (5-fold CV, seed 42) and seed-42 refit for the conformal
  point + calibration bins (asserted equal to severity_extras.json).
Forecast re-plot recomputes HW/LGBM forecasts with the trainer's exact settings
(asserted equal to demand_forecast_metrics.csv) — see train_forecast.py (canonical).

IEEE double-column sizing: 3.5in single column, 7.16in page width; fonts >= 9pt.

Usage: python ds/paper_figures.py
Outputs: ds/models/paper/paper_fig{0,1,2,3,4,5,S1,S1b}_*.png + per-figure source CSVs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
PAPER_DIR = MODEL_DIR / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "figure.dpi": 300})
COLORS = {"blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c",
          "red": "#d62728", "black": "black"}


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------- Fig 0: schematic
def fig0_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    def box(x, y, w, h, text, sub="", color="#dce9f7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="black"))
        ax.text(x + w / 2, y + h / 2 + 0.12, text, ha="center", va="center",
                fontsize=9, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
                    fontsize=7)

    def arrow(x1, y1, x2, y2, label="", lab_xy=None, ha="center"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2))
        if label:
            lx, ly = lab_xy if lab_xy else ((x1 + x2) / 2, (y1 + y2) / 2 + 0.12)
            ax.text(lx, ly, label, ha=ha, fontsize=7)

    box(0.1, 1.0, 1.6, 1.0, "Voice intake", "village call")
    box(2.1, 1.0, 1.6, 1.0, "Features", "vitals + RFV")
    box(4.1, 1.7, 2.0, 1.0, "Module 1: triage", "XGB + abstain", "#f7e6c5")
    box(4.1, 0.4, 2.0, 1.0, "Module 2: outbreak", "IsolationForest", "#d9ead3")
    box(6.5, 1.7, 1.6, 1.0, "Doctor", "review")
    box(8.4, 1.0, 1.5, 1.0, "Delivery", "medicine")
    arrow(1.7, 1.5, 2.1, 1.5)
    arrow(3.7, 1.7, 4.1, 2.2)
    arrow(3.7, 1.15, 4.1, 0.8, "district counts", lab_xy=(4.02, 0.62), ha="right")
    arrow(6.1, 2.2, 6.5, 2.2)
    arrow(8.1, 2.2, 8.55, 1.6)
    arrow(5.3, 1.7, 5.3, 1.4, "low conf.", lab_xy=(5.45, 1.55), ha="left")
    ax.text(0.1, 0.1, "Data: CDC NHAMCS 2019 (public domain) | covid19india archive "
            "(static) | Kaggle pharma sales (CC BY-NC, suppl.)", fontsize=7)
    ax.set_title("Curago data-science layer: patient-level triage + population-level surveillance")
    savefig(PAPER_DIR / "paper_fig0_pipeline.png")


# ---------------------------------------------------------------- Fig 1: ROC
def fig1_roc():
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from xgboost import XGBClassifier
    import sys
    sys.path.insert(0, str(HERE))
    from train_triage import load_real

    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    neg, pos = np.bincount(y)
    pipes = {
        "LogReg": Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(class_weight="balanced",
                                                       max_iter=1000, random_state=42))]),
        "XGBoost": Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("clf", XGBClassifier(
                                 scale_pos_weight=float(neg / pos), n_estimators=200,
                                 max_depth=6, learning_rate=0.1, random_state=42,
                                 eval_metric="logloss"))]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = np.linspace(0, 1, 200)
    curves, aucs = {}, {}
    oof = {}
    for name, pipe in pipes.items():
        tprs, a = [], []
        oof_p = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            oof_p[te] = p
            fpr, tpr, _ = roc_curve(y[te], p)
            tprs.append(np.interp(grid, fpr, tpr))
            a.append(auc(fpr, tpr))
        curves[name] = (np.mean(tprs, axis=0), np.std(tprs, axis=0))
        aucs[name] = float(np.mean(a))
        oof[name] = oof_p
    # assert against committed CV means
    mt = pd.read_csv(MODEL_DIR / "metric_table.csv")
    assert abs(aucs["LogReg"] - mt.loc[0, "AUC-ROC"]) < 0.01, aucs
    assert abs(aucs["XGBoost"] - mt.loc[1, "AUC-ROC"]) < 0.01, aucs

    def point_at(p, t):
        pred = (p >= t).astype(int)
        fpr = float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))
        tpr = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
        return fpr, tpr

    op = json.load(open(MODEL_DIR / "operating_point.json"))
    pts = {"t=0.5": point_at(oof["XGBoost"], 0.5),
           f"tuned t={op['xgboost']['threshold']:.2f}":
               point_at(oof["XGBoost"], op["xgboost"]["threshold"]),
           "conformal s42": (None, None)}  # filled below
    # seed-42 conformal point (recompute; assert vs severity_extras.json)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    xp = pipes["XGBoost"]
    xp.fit(X_tr, y_tr)
    pc = xp.predict_proba(X_cal)[:, 1]
    ppos = np.sort(pc[y_cal == 1])
    k = int(np.floor(0.15 * (len(ppos) + 1)))
    tc = float(ppos[k - 1])
    pte = xp.predict_proba(X_te)[:, 1]
    ex = json.load(open(MODEL_DIR / "severity_extras.json"))
    assert abs(tc - ex["split"]["threshold"]) < 1e-6, (tc, ex["split"]["threshold"])
    assert abs(recall_score(y_te, (pte >= tc).astype(int)) - ex["xgboost_test"]["recall"]) < 1e-3
    pts["conformal s42"] = (float(((pte >= tc) & (y_te == 0)).sum() / (y_te == 0).sum()),
                            float(((pte >= tc) & (y_te == 1)).sum() / (y_te == 1).sum()))

    src = pd.DataFrame({"fpr_grid": grid,
                        "logreg_tpr_mean": curves["LogReg"][0],
                        "xgb_tpr_mean": curves["XGBoost"][0]})
    src.to_csv(PAPER_DIR / "paper_fig1_roc_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    for name, col in [("LogReg", COLORS["blue"]), ("XGBoost", COLORS["orange"])]:
        m, s = curves[name]
        ax.plot(grid, m, color=col, label=f"{name} AUC={aucs[name]:.3f}")
        ax.fill_between(grid, m - s, m + s, color=col, alpha=0.2)
    for lab, (fx, fy) in pts.items():
        ax.plot(fx, fy, "o", color=COLORS["red"] if "tuned" in lab else "black",
                ms=5, label=f"XGB {lab} ({fx:.2f},{fy:.2f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("Triage ROC, 5-fold CV (n=13,595)")
    ax.legend(fontsize=7, loc="lower right")
    savefig(PAPER_DIR / "paper_fig1_roc.png")


# ---------------------------------------------------------------- Fig 2: SHAP (re-render)
def fig2_shap():
    import shap
    import joblib
    model = joblib.load(MODEL_DIR / "triage.pkl")
    feats = json.load(open(MODEL_DIR / "triage_features.json"))
    data = pd.read_csv(HERE / "data" / "processed" / "nhamcs_triage_features.csv")
    X = data.drop(columns=["is_high_severity"])
    X_imp = model.named_steps["imputer"].transform(X)
    expl = shap.TreeExplainer(model.named_steps["clf"])
    sv = expl.shap_values(X_imp[:500])
    plt.figure(figsize=(7.16, 5.5))
    shap.summary_plot(sv, X_imp[:500], feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(PAPER_DIR / "paper_fig2_shap.png", dpi=300)
    plt.close()
    print(f"Saved {PAPER_DIR / 'paper_fig2_shap.png'}")


# ---------------------------------------------------------------- Fig 3: tradeoff re-plot
def fig3_tradeoff():
    df = pd.read_csv(MODEL_DIR / "threshold_tradeoff.csv")
    op = json.load(open(MODEL_DIR / "operating_point.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
    for ax, (name, sub) in zip(axes, df.groupby("model")):
        for col, c in [("recall", COLORS["blue"]), ("precision", COLORS["orange"]),
                       ("under_triage", COLORS["green"]), ("over_triage", COLORS["red"])]:
            ax.plot(sub["threshold"], sub[col], marker="o", ms=2, label=col, color=c)
        t = op[name]["threshold"]
        ax.axvline(t, color="black", ls="--", label=f"selected t={t:.2f}")
        ax.set_title(f"{name} (OOF)")
        ax.set_xlabel("threshold on P(High)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("rate")
    savefig(PAPER_DIR / "paper_fig3_tradeoff.png")


# ---------------------------------------------------------------- Fig 4: conformal hist re-plot
def fig4_conformal_hist():
    df = pd.read_csv(MODEL_DIR / "conformal_repeats.csv")
    rec = df["test_recall"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.hist(rec, bins=20, edgecolor="black", color=COLORS["blue"])
    ax.axvline(0.85, color=COLORS["red"], ls="--", label="target 1-alpha=0.85")
    ax.axvline(rec.mean(), color="black", ls="-",
               label=f"mean={rec.mean():.4f} (n={len(rec)})")
    ax.set_xlabel("test recall (per split)")
    ax.set_ylabel("splits")
    ax.set_title("Conformal recall over 200 splits (marginal guarantee)")
    ax.legend(fontsize=7)
    savefig(PAPER_DIR / "paper_fig4_conformal_hist.png")


# ---------------------------------------------------------------- Fig 5: outbreak re-plot
def fig5_outbreak():
    ev = json.load(open(MODEL_DIR / "outbreak_eval_v2.json"))
    names = list(ev["methods"])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, wname in zip(axes, ["delta", "omicron"]):
        vals = [ev["methods"][m][wname]["flag_rate_in_window"] for m in names]
        covs = [ev["methods"][m][wname]["district_coverage"] for m in names]
        ax.bar(x - 0.2, vals, 0.4, label="in-window flag rate", color=COLORS["blue"])
        ax.bar(x + 0.2, covs, 0.4, label="district coverage", color=COLORS["orange"])
        ax.set_ylim(0, max(max(vals), max(covs)) * 1.35)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_title(f"{wname}")
        ax.legend(fontsize=7)
    fig.suptitle("Outbreak methods vs epidemic windows (descriptive)")
    savefig(PAPER_DIR / "paper_fig5_outbreak.png")


# ---------------------------------------------------------------- Suppl: forecast re-plot (recompute, assert vs metrics CSV)
def figS1_forecast():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error
    import sys
    sys.path.insert(0, str(HERE))
    from train_forecast import load_real, DRUGS
    df = load_real()
    met = pd.read_csv(MODEL_DIR / "demand_forecast_metrics.csv").set_index("Drug (ATC)")
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 9))
    axes = axes.flatten()
    rows = []
    for idx, drug in enumerate(DRUGS):
        series = df.set_index("date")[drug].dropna()
        series = series.asfreq(pd.infer_freq(series.index) or "ME")
        train, test = series[:-12], series[-12:]
        hw = ExponentialSmoothing(train, seasonal="add",
                                  seasonal_periods=12).fit(optimized=True, use_brute=True)
        hw_pred = np.asarray(hw.forecast(12))
        dfl = pd.DataFrame({"y": series})
        for lag in range(1, 13):
            dfl[f"lag_{lag}"] = dfl["y"].shift(lag)
        dfl["month"] = dfl.index.month
        dfl = dfl.dropna()
        X, yy = dfl.drop(columns=["y"]), dfl["y"]
        lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb.fit(X[:-12], yy[:-12])
        lgb_pred = lgb.predict(X[-12:])
        mae_hw = round(float(mean_absolute_error(test, hw_pred)), 2)
        mae_lgb = round(float(mean_absolute_error(yy[-12:], lgb_pred)), 2)
        assert abs(mae_hw - met.loc[drug, "MAE Holt-Winters"]) < 0.02, (drug, mae_hw)
        assert abs(mae_lgb - met.loc[drug, "MAE LightGBM"]) < 0.02, (drug, mae_lgb)
        rows.append({"drug": drug, "mae_hw": mae_hw, "mae_lgb": mae_lgb})
        ax = axes[idx]
        ax.plot(test.index, test.values, "k-", label="Actual", lw=1.5)
        ax.plot(test.index, hw_pred, "--", color=COLORS["blue"], label=f"HW ({mae_hw})", lw=1)
        ax.plot(test.index, lgb_pred, "--", color=COLORS["red"], label=f"LGB ({mae_lgb})", lw=1)
        ax.set_title(drug)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    pd.DataFrame(rows).to_csv(PAPER_DIR / "paper_figS1_forecast_data.csv", index=False)
    savefig(PAPER_DIR / "paper_figS1_forecast.png")


# ---------------------------------------------------------------- Suppl: calibration re-plot (recompute, assert Brier)
def figS1b_calibration():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss
    import sys
    sys.path.insert(0, str(HERE))
    from train_triage import load_real
    from conformal_triage import split_60_20_20, build_pipe, conformal_threshold, MAIN_ALPHA
    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = split_60_20_20(X, y, 42)
    neg, pos = np.bincount(y_tr)
    pipe = build_pipe(float(neg / pos))
    pipe.fit(X_tr, y_tr)
    pte = pipe.predict_proba(X_te)[:, 1]
    brier = round(float(brier_score_loss(y_te, pte)), 4)
    ex = json.load(open(MODEL_DIR / "severity_extras.json"))
    assert abs(brier - ex["xgboost_test"]["brier"]) < 1e-3, (brier, ex)
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(pte, bins) - 1, 0, 9)
    rows = []
    for b in range(10):
        m = which == b
        if m.sum():
            rows.append({"bin": b, "mean_pred": round(float(pte[m].mean()), 4),
                         "obs_frac": round(float(y_te[m].mean()), 4), "n": int(m.sum())})
    pd.DataFrame(rows).to_csv(PAPER_DIR / "paper_figS1b_calibration_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    d = pd.DataFrame(rows)
    ax.scatter(d["mean_pred"], d["obs_frac"],
               s=[max(10, n / 8) for n in d["n"]], color=COLORS["blue"])
    for _, r in d.iterrows():
        ax.annotate(str(int(r["n"])), (r["mean_pred"], r["obs_frac"]), fontsize=7)
    ax.set_xlabel("mean predicted P(High)")
    ax.set_ylabel("observed fraction High")
    ax.set_title(f"Reliability, seed-42 test (Brier={brier:.4f})")
    ax.legend(fontsize=7)
    savefig(PAPER_DIR / "paper_figS1b_calibration.png")


if __name__ == "__main__":
    fig0_pipeline()
    fig1_roc()
    fig2_shap()
    fig3_tradeoff()
    fig4_conformal_hist()
    fig5_outbreak()
    figS1_forecast()
    figS1b_calibration()
    print("ALL PAPER FIGURES DONE")
