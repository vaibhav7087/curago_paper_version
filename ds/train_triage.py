"""
ds/train_triage.py — Triage severity classifier (CDC NHAMCS 2019 ED, real data only).

FAIL-CLOSED: no synthetic fallback. Missing raw data = hard error.

Raw data (download before running, US public domain, no auth):
  https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/ed2019_sas.zip
  (fallback: .../NHAMCS/stata/ED2019-stata.zip -> inner ED2019.dta)
Unzip into ds/data/raw/ (any depth) — inner filename is probed by glob, not hardcoded.

Outputs: models/triage.pkl, models/triage_features.json,
         models/metric_table.csv, models/triage_oof_stats.json,
         models/shap_summary.png

Protocol: manual 5-fold OOF loop (StratifiedKFold, shuffle, seed 42) over the
candidates in ds/triage_models.py; fold-mean metrics in metric_table.csv
(incl. Brier and a constant-at-train-prevalence reference row), pooled OOF
predictions retained for bootstrap AUC CIs. Primary model chosen by the
pre-declared Brier rule (asserted, fail-closed) and saved as triage.pkl.
"""
import json
import joblib
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
DATA_PROCESSED = HERE / "data" / "processed"
DATA_RAW = HERE / "data" / "raw"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DATA_SOURCE_LABEL = "CDC NHAMCS 2019 ED"
RAW_URL = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/"
           "Dataset_Documentation/NHAMCS/sas/ed2019_sas.zip")


def load_real():
    """Load real NHAMCS 2019 ED data. Fail closed if unavailable."""
    candidates = list(DATA_RAW.glob("**/*.sas7bdat")) + list(DATA_RAW.glob("**/*.dta"))
    if not candidates:
        raise FileNotFoundError(
            "No NHAMCS 2019 ED raw file found in ds/data/raw/.\n"
            "Download (public domain, ~3.5 MB):\n"
            f"  {RAW_URL}\n"
            "Stata fallback: .../NHAMCS/stata/ED2019-stata.zip (inner: ED2019.dta)\n"
            "Unzip into ds/data/raw/ then re-run. No synthetic data is used in this repo."
        )
    last_err = None
    for cand in candidates:
        try:
            if str(cand).endswith(".sas7bdat"):
                df, _ = pyreadstat.read_sas7bdat(str(cand))
            else:
                df, _ = pyreadstat.read_dta(str(cand))
            print(f"Loaded real CDC file {cand} shape={df.shape}")
            return prepare_nhamcs(df)
        except Exception as e:
            last_err = e
            print(f"Load failed for {cand}: {e}")
    raise RuntimeError(f"All raw candidates failed. Last error: {last_err}")


def prepare_nhamcs(df):
    if "IMMEDR" not in df.columns or "AGE" not in df.columns:
        raise ValueError(f"Unexpected NHAMCS schema: {sorted(df.columns)[:20]}")
    df_clean = df[df["IMMEDR"].isin([1, 2, 3, 4, 5])].copy()
    df_clean["is_high_severity"] = (df_clean["IMMEDR"].isin([1, 2])).astype(int)
    feature_cols = []
    df_clean["age"] = df_clean["AGE"].clip(0, 120)
    df_clean["is_female"] = (df_clean["SEX"] == 1).astype(int)
    feature_cols += ["age", "is_female"]
    for col in ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]:
        if col in df_clean.columns:
            low = col.lower()
            df_clean[low] = df_clean[col].apply(lambda x: x if x > 0 else pd.NA)
            feature_cols.append(low)
    if "PAINSCALE" in df_clean.columns:
        df_clean["pain_scale"] = df_clean["PAINSCALE"].apply(
            lambda x: x if 0 <= x <= 10 else pd.NA)
        feature_cols.append("pain_scale")
    if "RFV1" in df_clean.columns:
        top_rfv = df_clean["RFV1"].value_counts().head(20).index.tolist()
        for rfv in top_rfv:
            try:
                col_name = f"rfv_{int(rfv)}"
            except Exception:
                continue
            df_clean[col_name] = (df_clean["RFV1"] == rfv).astype(int)
            feature_cols.append(col_name)
    out = DATA_PROCESSED / "nhamcs_triage_features.csv"
    df_clean[feature_cols + ["is_high_severity"]].to_csv(out, index=False)
    print(f"Saved CDC processed to {out} shape={df_clean.shape}")
    return pd.read_csv(out)


def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (recall_score, precision_score, f1_score,
                                 roc_auc_score, brier_score_loss)
    from triage_models import (CANDIDATES, LABELS, PRIMARY, CALIBRATED,
                               SELECTION_RULE, make_estimator, extract_base)

    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    feature_cols = list(X.columns)
    print(f"X shape: {X.shape}, y: {dict(zip(*np.unique(y, return_counts=True)))}")
    neg, pos = np.bincount(y)
    scale_pos_weight = float(neg / max(1, pos))
    print(f"scale_pos_weight (weighted predecessor): {scale_pos_weight:.2f}")
    n = len(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def fold_metrics(est_name, oof, fold_briers, fold_aucs, fold_rec,
                     fold_prec, fold_f1):
        return {
            "Model": LABELS[est_name],
            "AUC-ROC": float(np.mean(fold_aucs)),
            "Recall (Sensitivity)": float(np.mean(fold_rec)),
            "Precision": float(np.mean(fold_prec)),
            "F1 Score": float(np.mean(fold_f1)),
            "Under-Triage Rate": float(1.0 - np.mean(fold_rec)),
            "Brier (OOF)": float(np.mean(fold_briers)),
            "Data Source": f"{DATA_SOURCE_LABEL} (n={n})",
        }

    rows, oof_store = [], {}
    for name in CANDIDATES:
        oof = np.zeros(n)
        briers, aucs, recs, precs, f1s = [], [], [], [], []
        for tr, te in cv.split(X, y):
            est = make_estimator(name, scale_pos_weight=scale_pos_weight)
            est.fit(X.iloc[tr], y[tr])
            p = est.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            pred = (p >= 0.5).astype(int)
            yte = y[te]
            briers.append(brier_score_loss(yte, p))
            aucs.append(roc_auc_score(yte, p))
            recs.append(recall_score(yte, pred, zero_division=0))
            precs.append(precision_score(yte, pred, zero_division=0))
            f1s.append(f1_score(yte, pred, zero_division=0))
        oof_store[name] = oof
        rows.append(fold_metrics(name, oof, briers, aucs, recs, precs, f1s))
        print(f"{LABELS[name]}: AUC {rows[-1]['AUC-ROC']:.4f} "
              f"recall {rows[-1]['Recall (Sensitivity)']:.4f} "
              f"brier {rows[-1]['Brier (OOF)']:.4f}")

    # Constant-at-train-prevalence reference row (per-fold prevalence -> OOF)
    oof_c = np.zeros(n)
    briers, aucs, recs, precs, f1s = [], [], [], [], []
    for tr, te in cv.split(X, y):
        pbar = float(y[tr].mean())
        oof_c[te] = pbar
        pred = np.zeros(len(te), dtype=int) if pbar < 0.5 else np.ones(len(te), dtype=int)
        yte = y[te]
        briers.append(brier_score_loss(yte, np.full(len(te), pbar)))
        aucs.append(roc_auc_score(yte, np.full(len(te), pbar)))
        recs.append(recall_score(yte, pred, zero_division=0))
        precs.append(precision_score(yte, pred, zero_division=0))
        f1s.append(f1_score(yte, pred, zero_division=0))
    oof_store["constant"] = oof_c
    rows.append(fold_metrics("constant", oof_c, briers, aucs, recs, precs, f1s))
    print(f"Constant (train prevalence): brier {rows[-1]['Brier (OOF)']:.4f} "
          f"auc {rows[-1]['AUC-ROC']:.4f}")

    metric_table = pd.DataFrame(rows)
    print("\nMETRIC TABLE:")
    try:
        print(metric_table.to_markdown(index=False))
    except Exception:
        print(metric_table.to_string(index=False))
    metric_table.to_csv(MODEL_DIR / "metric_table.csv", index=False)

    # ---- pre-declared primary selection rule (fail-closed) ------------------
    brier_by_cal = {m: float(metric_table.loc[metric_table.Model == LABELS[m],
                                              "Brier (OOF)"].iloc[0])
                    for m in CALIBRATED}
    rule_winner = min(sorted(brier_by_cal), key=lambda m: brier_by_cal[m])
    if rule_winner != PRIMARY:
        raise RuntimeError(
            f"PRE-DECLARED RULE VIOLATION: rule picks {rule_winner} "
            f"(Brier {brier_by_cal}), but PRIMARY is locked to {PRIMARY}. "
            "Update PRIMARY deliberately or investigate the drift."
        )
    print(f"Primary selection: {PRIMARY} ({SELECTION_RULE}; "
          f"Brier {brier_by_cal})")

    # ---- OOF bootstrap AUC CIs (paired, seed 42) ---------------------------
    rng = np.random.default_rng(42)
    n_boot = 1000
    idx_all = np.arange(n)

    def boot_auc(p, idx):
        return roc_auc_score(y[idx], p[idx])

    ci_store, diff = {}, []
    for m in ["logreg", PRIMARY]:
        vals = [boot_auc(oof_store[m], rng.choice(idx_all, n, replace=True))
                for _ in range(n_boot)]
        ci_store[m] = [round(float(np.percentile(vals, 2.5)), 4),
                       round(float(np.percentile(vals, 97.5)), 4)]
    for _ in range(n_boot):
        b = rng.choice(idx_all, n, replace=True)
        diff.append(roc_auc_score(y[b], oof_store[PRIMARY][b])
                    - roc_auc_score(y[b], oof_store["logreg"][b]))
    diff_ci = [round(float(np.percentile(diff, 2.5)), 4),
               round(float(np.percentile(diff, 97.5)), 4)]
    stats = {
        "protocol": "5-fold OOF (StratifiedKFold, shuffle, seed 42); "
                    "bootstrap 1000 resamples of OOF rows, rng 42",
        "bootstrap_resamples": n_boot,
        "primary": PRIMARY,
        "selection_rule": SELECTION_RULE,
        "selection_brier": {k: round(v, 4) for k, v in brier_by_cal.items()},
        "selection_chosen": rule_winner,
        "oof_auc": {k: round(float(roc_auc_score(y, v)), 4)
                    for k, v in oof_store.items()},
        "oof_brier": {k: round(float(brier_score_loss(y, v)), 4)
                      for k, v in oof_store.items()},
        "auc_ci95": ci_store,
        "auc_diff_primary_minus_logreg_ci95": diff_ci,
    }
    with open(MODEL_DIR / "triage_oof_stats.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"AUC 95% CI: {PRIMARY} {ci_store[PRIMARY]}, logreg {ci_store['logreg']}, "
          f"diff {diff_ci}")

    # ---- final artifact: primary model fitted on all data -------------------
    model = make_estimator(PRIMARY, scale_pos_weight=scale_pos_weight)
    model.fit(X, y)
    joblib.dump(model, MODEL_DIR / "triage.pkl")
    with open(MODEL_DIR / "triage_features.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved ds/models/triage.pkl ({PRIMARY}) + triage_features.json "
          "+ triage_oof_stats.json")

    # SHAP on the underlying booster of the primary model (with fallback)
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        base = extract_base(model)
        X_imp = base.named_steps["imputer"].transform(X)
        explainer = shap.TreeExplainer(base.named_steps["clf"])
        sv = explainer.shap_values(X_imp[:500])
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_imp[:500], feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150)
        plt.close()
        print("Saved shap_summary.png (SHAP)")
    except Exception as e:
        print(f"SHAP failed ({e}), using importance fallback...")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        imp = extract_base(model).named_steps["clf"].feature_importances_
        idx = np.argsort(imp)[-12:][::-1]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_cols[i] for i in idx][::-1], imp[idx][::-1])
        plt.title("Feature importance (XGBoost fallback)")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150)
        plt.close()
        print("Saved shap_summary.png (fallback)")


if __name__ == "__main__":
    main()
