"""
ds/tune_threshold.py — Recall-tuned operating point for triage (real data only).

FAIL-CLOSED: reuses train_triage.load_real() — missing raw data = hard error.

Method (no leakage, no synthetic data):
  1. Same features, same pipelines, same StratifiedKFold(5, seed 42) as
     train_triage.py (the 0.5-threshold baseline stays untouched there).
  2. cross_val_predict(method="predict_proba") -> OUT-OF-FOLD P(High) per row.
  3. Sweep thresholds 0.05..0.95: recall / precision / F1 / under-triage /
     over-triage (=1-precision) / abstention-rate (max proba < 0.60, the
     predict.py rule, kept unchanged).
  4. Selection rule (pre-declared): lowest threshold with OOF recall >= 0.80,
     tie-break by max precision.

This moves ALONG the fixed ROC curve (AUC unchanged) — it trades over-triage
for under-triage. Both operating points are reported; nothing is hidden.

Outputs: models/threshold_tradeoff.csv, models/threshold_tradeoff.png,
         models/operating_point.json (consumed by ds/predict.py).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, precision_score, f1_score
from xgboost import XGBClassifier

from train_triage import load_real, DATA_SOURCE_LABEL

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RECALL_TARGET = 0.80
ABSTAIN_CUTOFF = 0.60  # must match ds/predict.py


def build_pipes(scale_pos_weight):
    # MUST match train_triage.py pipelines exactly.
    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipe_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        )),
    ])
    return {"logreg": pipe_lr, "xgboost": pipe_xgb}


def sweep(oof_proba, y_true):
    rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        pred = (oof_proba >= round(float(t), 2)).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    rdf = pd.DataFrame(rows)
    rdf["under_triage"] = 1.0 - rdf["recall"]
    rdf["over_triage"] = 1.0 - rdf["precision"]
    rdf["abstention_rate"] = float((np.maximum(oof_proba, 1 - oof_proba) < ABSTAIN_CUTOFF).mean())
    return rdf


def pick(rdf):
    ok = rdf[rdf["recall"] >= RECALL_TARGET]
    if len(ok):
        return ok.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return rdf.sort_values("recall", ascending=False).iloc[0]


def main():
    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    neg, pos = np.bincount(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipes = build_pipes(float(neg / max(1, pos)))

    trade, operating = [], {}
    for name, pipe in pipes.items():
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        rdf = sweep(oof, y)
        rdf["model"] = name
        trade.append(rdf)
        sel = pick(rdf)
        operating[name] = {
            "threshold": float(sel["threshold"]),
            "expected_recall": round(float(sel["recall"]), 4),
            "expected_precision": round(float(sel["precision"]), 4),
            "expected_f1": round(float(sel["f1"]), 4),
            "expected_under_triage": round(float(sel["under_triage"]), 4),
            "expected_over_triage": round(float(sel["over_triage"]), 4),
            "abstention_rate": round(float(rdf["abstention_rate"].iloc[0]), 4),
        }
        print(f"{name}: t={sel['threshold']:.2f} recall={sel['recall']:.3f} "
              f"precision={sel['precision']:.3f} under={sel['under_triage']:.3f} "
              f"over={sel['over_triage']:.3f}")

    operating["rule"] = (f"lowest threshold with OOF recall>={RECALL_TARGET}, "
                         "tie-break max precision")
    operating["selected_model"] = "xgboost"  # matches ds/predict.py model_used
    trade_df = pd.concat(trade, ignore_index=True)
    trade_df["Data Source"] = f"{DATA_SOURCE_LABEL} (n={len(X)})"
    trade_df.to_csv(MODEL_DIR / "threshold_tradeoff.csv", index=False)
    with open(MODEL_DIR / "operating_point.json", "w") as f:
        json.dump(operating, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (name, sub) in zip(axes, trade_df.groupby("model")):
        for col in ["recall", "precision", "under_triage", "over_triage"]:
            ax.plot(sub["threshold"], sub[col], marker="o", ms=3, label=col)
        t = operating[name]["threshold"]
        ax.axvline(t, color="k", ls="--", label=f"selected t={t:.2f}")
        ax.set_title(f"{name} (OOF, n={len(X)})")
        ax.set_xlabel("threshold on P(High)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rate")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "threshold_tradeoff.png", dpi=150)
    plt.close()
    print("Saved threshold_tradeoff.csv/.png + operating_point.json")


if __name__ == "__main__":
    main()
