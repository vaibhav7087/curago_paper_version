"""
ds/train_triage.py — Triage severity classifier (CDC NHAMCS 2019 ED, real data only).

FAIL-CLOSED: no synthetic fallback. Missing raw data = hard error.

Raw data (download before running, US public domain, no auth):
  https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/ed2019_sas.zip
  (fallback: .../NHAMCS/stata/ED2019-stata.zip -> inner ED2019.dta)
Unzip into ds/data/raw/ (any depth) — inner filename is probed by glob, not hardcoded.

Outputs: models/triage.pkl, models/triage_features.json,
         models/metric_table.csv, models/shap_summary.png
"""
import json
import joblib
import numpy as np
import pandas as pd
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
    global pyreadstat
    import pyreadstat
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
    from xgboost import XGBClassifier

    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"]
    feature_cols = list(X.columns)
    print(f"X shape: {X.shape}, y: {y.value_counts().to_dict()}")
    neg, pos = y.value_counts().values
    scale_pos_weight = float(neg / max(1, pos))
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
        "f1": make_scorer(f1_score),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    print("\n=== Logistic Regression (baseline) ===")
    results_lr = cross_validate(pipe_lr, X, y, cv=cv, scoring=scoring)
    for m in scoring:
        v = results_lr[f"test_{m}"]
        print(f"  {m}: {v.mean():.4f} ± {v.std():.4f}")

    pipe_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        )),
    ])
    print("\n=== XGBoost (class-weighted) ===")
    results_xgb = cross_validate(pipe_xgb, X, y, cv=cv, scoring=scoring)
    for m in scoring:
        v = results_xgb[f"test_{m}"]
        print(f"  {m}: {v.mean():.4f} ± {v.std():.4f}")

    under_lr = 1 - results_lr["test_recall"].mean()
    under_xgb = 1 - results_xgb["test_recall"].mean()
    print(f"\nUnder-triage (LogReg): {under_lr:.4f}")
    print(f"Under-triage (XGBoost): {under_xgb:.4f}")

    pipe_xgb.fit(X, y)
    joblib.dump(pipe_xgb, MODEL_DIR / "triage.pkl")
    with open(MODEL_DIR / "triage_features.json", "w") as f:
        json.dump(feature_cols, f)
    print("Saved ds/models/triage.pkl + triage_features.json")

    n = len(X)
    metric_table = pd.DataFrame({
        "Model": ["Logistic Regression (baseline)", "XGBoost (class-weighted)"],
        "AUC-ROC": [results_lr["test_roc_auc"].mean(), results_xgb["test_roc_auc"].mean()],
        "Recall (Sensitivity)": [results_lr["test_recall"].mean(), results_xgb["test_recall"].mean()],
        "Precision": [results_lr["test_precision"].mean(), results_xgb["test_precision"].mean()],
        "F1 Score": [results_lr["test_f1"].mean(), results_xgb["test_f1"].mean()],
        "Under-Triage Rate": [under_lr, under_xgb],
        "Data Source": [f"{DATA_SOURCE_LABEL} (n={n})", f"{DATA_SOURCE_LABEL} (n={n})"],
    })
    print("\nMETRIC TABLE:")
    try:
        print(metric_table.to_markdown(index=False))
    except Exception:
        print(metric_table.to_string(index=False))
    metric_table.to_csv(MODEL_DIR / "metric_table.csv", index=False)

    # SHAP with fallback to importance bar
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X_imp = pipe_xgb.named_steps["imputer"].transform(X)
        explainer = shap.TreeExplainer(pipe_xgb.named_steps["clf"])
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
        imp = pipe_xgb.named_steps["clf"].feature_importances_
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
