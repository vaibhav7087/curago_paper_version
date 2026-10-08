"""
ds/tune_threshold.py — Recall-tuned operating point for triage (real data only).

FAIL-CLOSED: reuses train_triage.load_real() — missing raw data = hard error.

Method (no leakage, no synthetic data):
  1. Same features and estimators from ds/triage_models.py; the primary
     (calibrated) model and the logistic baseline are swept separately.
  2. Pooled OOF sweep (descriptive): cross_val_predict on
     StratifiedKFold(5, shuffle, seed 42) -> OUT-OF-FOLD P(High) per row;
     thresholds 0.05..0.95 -> recall / precision / F1 / FPR / under-triage /
     over-triage (=1-precision) / abstention-rate (max proba < 0.60, the
     predict.py rule, kept unchanged).
  3. Selection rule (pre-declared, matches pick()): among thresholds whose
     OOF recall reaches at least 0.80, take the one with the highest
     precision; ties are broken by the higher threshold.
  4. NESTED evaluation (honest headline): outer StratifiedKFold(5, seed 42);
     inside each outer-training fold the selection rule is re-applied to an
     inner 5-fold OOF sweep, and the chosen threshold is evaluated on the
     held-out outer fold. Tune and evaluate never see the same rows.

Moving along the fixed ROC curve does not change AUC. Both operating points
(pooled descriptive and nested cross-fitted) are reported; nothing is hidden.

Outputs: models/threshold_tradeoff.csv, models/threshold_tradeoff.png,
         models/operating_point.json (consumed by ds/predict.py).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score)

from train_triage import load_real, DATA_SOURCE_LABEL
from triage_models import make_estimator, PRIMARY

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RECALL_TARGET = 0.80
ABSTAIN_CUTOFF = 0.60  # must match ds/predict.py
RULE_TEXT = ("max precision among thresholds with OOF recall>=0.80 "
             "(ties: higher threshold)")


def rates(pred, y_true):
    neg = y_true == 0
    pos = y_true == 1
    return {
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(((pred == 1) & neg).sum() / max(1, neg.sum())),
        "under_triage": 1.0 - float(recall_score(y_true, pred, zero_division=0)),
        "over_triage": 1.0 - float(precision_score(y_true, pred, zero_division=0)),
    }


def sweep(oof_proba, y_true):
    # 0.01 grid: calibrated (isotonic) scores are stepped, so a 0.05 grid
    # jumps across large probability mass near the recall floor.
    rows = []
    for t in np.arange(0.01, 1.0, 0.01):
        t = round(float(t), 2)
        r = rates((oof_proba >= t).astype(int), y_true)
        r["threshold"] = t
        rows.append(r)
    rdf = pd.DataFrame(rows)
    rdf["abstention_rate"] = float(
        (np.maximum(oof_proba, 1 - oof_proba) < ABSTAIN_CUTOFF).mean())
    return rdf


def pick(rdf):
    """Pre-declared rule: max precision among recall>=RECALL_TARGET rows,
    ties broken by higher threshold (see RULE_TEXT)."""
    ok = rdf[rdf["recall"] >= RECALL_TARGET]
    if len(ok):
        return ok.sort_values(["precision", "threshold"],
                              ascending=[False, False]).iloc[0]
    return rdf.sort_values("recall", ascending=False).iloc[0]


def pooled_point(oof_proba, y_true, t):
    r = rates((oof_proba >= t).astype(int), y_true)
    r["threshold"] = float(t)
    r["abstention_rate"] = float(
        (np.maximum(oof_proba, 1 - oof_proba) < ABSTAIN_CUTOFF).mean())
    return r


def nested_crossfit(X, y, name, spw):
    """Outer 5-fold: tune on inner OOF, evaluate on the outer fold."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    chosen_t = np.zeros(len(y))
    per_fold = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y[tr]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_oof = cross_val_predict(
            make_estimator(name, scale_pos_weight=spw),
            X_tr, y_tr, cv=inner, method="predict_proba")[:, 1]
        sel = pick(sweep(inner_oof, y_tr))
        t = float(sel["threshold"])
        est = make_estimator(name, scale_pos_weight=spw)
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        chosen_t[te] = t
        r = rates((p >= t).astype(int), y[te])
        r.update({"fold": i, "threshold": t, "n_test": int(len(te)),
                  "inner_tuned_recall": round(float(sel["recall"]), 4),
                  "inner_tuned_precision": round(float(sel["precision"]), 4)})
        per_fold.append(r)
        print(f"  nested fold {i}: t={t:.2f} "
              f"test recall={r['recall']:.3f} precision={r['precision']:.3f} "
              f"fpr={r['fpr']:.3f}")
    pred = (oof >= chosen_t).astype(int)
    pooled = rates(pred, y)
    pooled["abstention_rate"] = float(
        (np.maximum(oof, 1 - oof) < ABSTAIN_CUTOFF).mean())
    pooled["auc"] = float(roc_auc_score(y, oof))
    return {
        "protocol": ("outer StratifiedKFold(5, shuffle, seed 42); inner "
                     "5-fold OOF selection on the outer-training folds; "
                     "outer-test evaluation"),
        "thresholds": [f["threshold"] for f in per_fold],
        "pooled": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in pooled.items()},
        "per_fold": [{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in f.items()} for f in per_fold],
    }


def main():
    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    neg, pos = np.bincount(y)
    spw = float(neg / max(1, pos))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipes = {"logreg": make_estimator("logreg"),
             "xgboost": make_estimator(PRIMARY)}

    trade, operating = [], {}
    for name, pipe in pipes.items():
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        rdf = sweep(oof, y)
        rdf["model"] = name
        trade.append(rdf)
        sel = pick(rdf)
        operating[name] = dict(
            threshold=round(float(sel["threshold"]), 4),
            expected_recall=round(float(sel["recall"]), 4),
            expected_precision=round(float(sel["precision"]), 4),
            expected_f1=round(float(sel["f1"]), 4),
            expected_fpr=round(float(sel["fpr"]), 4),
            expected_under_triage=round(float(sel["under_triage"]), 4),
            expected_over_triage=round(float(sel["over_triage"]), 4),
            abstention_rate=round(float(sel["abstention_rate"]), 4),
        )
        print(f"{name}: t={sel['threshold']:.2f} recall={sel['recall']:.3f} "
              f"precision={sel['precision']:.3f} fpr={sel['fpr']:.3f} "
              f"under={sel['under_triage']:.3f} over={sel['over_triage']:.3f}")

    print("Nested cross-fitted evaluation (primary model):")
    operating["nested_xgboost"] = nested_crossfit(X, y, PRIMARY, spw)
    np_ = operating["nested_xgboost"]["pooled"]
    print(f"  pooled: recall={np_['recall']:.3f} precision={np_['precision']:.3f} "
          f"fpr={np_['fpr']:.3f} abstention={np_['abstention_rate']:.3f}")

    operating["rule"] = RULE_TEXT
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
        for col in ["recall", "precision", "fpr", "under_triage", "over_triage"]:
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
