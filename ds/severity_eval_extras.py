"""
ds/severity_eval_extras.py — Credibility extras for Module 1 (real data only).

FAIL-CLOSED: loads via train_triage.load_real() — missing raw data = hard error.
Uses the fixed seed-42 60/20/20 split via conformal_triage helpers (no drift).

1. Bootstrap 95% CIs (1000 resamples of the TEST set, rng 42) for recall,
   precision, AUC at the seed-42 conformal threshold (alpha=0.15).
2. NEWS2-style rule baseline on available vitals (tempF, pulse, resp, BP sys,
   SpO2): standard NEWS2 cutoffs, missing vitals score 0 (rate disclosed).
   Cutoff = lowest NEWS total with calibration recall >= 0.85 (same bar as
   conformal); report NEWS test recall/precision beside XGBoost.
3. Calibration plot (10 uniform bins reliability diagram) + Brier score.

Usage: python ds/severity_eval_extras.py
Outputs: models/severity_extras.json, models/calibration_curve.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, roc_auc_score, brier_score_loss

from train_triage import load_real, DATA_SOURCE_LABEL
from conformal_triage import split_60_20_20, build_pipe, conformal_threshold, MAIN_ALPHA

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
SEED = 42
N_BOOT = 1000


def news2_scores(X):
    """Standard NEWS2 cutoffs. tempf is Fahrenheit -> convert to Celsius."""
    temp_c = (X["tempf"] - 32) * 5.0 / 9.0
    pulse = X["pulse"]
    resp = X["respr"]
    sysbp = X["bpsys"]
    spo2 = X["popct"]

    def pts(s, bands):
        out = pd.Series(np.nan, index=s.index)
        for (lo, hi, p) in bands:
            out = out.where(~((s >= lo) & (s <= hi)), p)
        return out.fillna(0)

    score = (
        pts(resp, [(0, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 300, 3)])
        + pts(spo2, [(0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 200, 0)])
        + pts(temp_c, [(-99, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0),
                       (38.1, 39.0, 1), (39.1, 99, 2)])
        + pts(sysbp, [(0, 90, 3), (91, 100, 2), (101, 110, 1),
                      (111, 219, 0), (220, 999, 3)])
        + pts(pulse, [(0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1),
                      (111, 130, 2), (131, 999, 3)])
    )
    n_missing = int((X[["tempf", "pulse", "respr", "bpsys", "popct"]].isna().any(axis=1)).sum())
    return score.to_numpy(), n_missing


def bootstrap_ci(y_true, proba, threshold, rng, n=N_BOOT):
    recs, precs, aucs = [], [], []
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        pred = (proba[b] >= threshold).astype(int)
        recs.append(recall_score(y_true[b], pred, zero_division=0))
        precs.append(precision_score(y_true[b], pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_true[b], proba[b]))
        except ValueError:
            aucs.append(float("nan"))
    def ci(v):
        v = np.array(v)
        return [round(float(np.nanpercentile(v, 2.5)), 4),
                round(float(np.nanpercentile(v, 97.5)), 4)]
    return {"recall": ci(recs), "precision": ci(precs), "auc": ci(aucs)}


def main():
    rng = np.random.default_rng(42)
    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = split_60_20_20(X, y, SEED)
    neg, pos = np.bincount(y_tr)
    pipe = build_pipe(float(neg / max(1, pos)))
    pipe.fit(X_tr, y_tr)
    cal_proba = pipe.predict_proba(X_cal)[:, 1]
    test_proba = pipe.predict_proba(X_te)[:, 1]
    t, n_cal_pos = conformal_threshold(cal_proba[y_cal == 1], MAIN_ALPHA)
    pred = (test_proba >= t).astype(int)

    test_metrics = {
        "recall": round(float(recall_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "auc": round(float(roc_auc_score(y_te, test_proba)), 4),
        "brier": round(float(brier_score_loss(y_te, test_proba)), 4),
        "abstention_rate": round(float((np.maximum(test_proba, 1 - test_proba) < 0.60).mean()), 4),
    }
    cis = bootstrap_ci(y_te, test_proba, t, rng)
    print(f"seed-42: t={t:.4f} recall={test_metrics['recall']} "
          f"(95% CI {cis['recall']}) precision={test_metrics['precision']} "
          f"(95% CI {cis['precision']}) auc={test_metrics['auc']} "
          f"(95% CI {cis['auc']}) brier={test_metrics['brier']}")

    # NEWS2 baseline
    cal_news, _ = news2_scores(X_cal)
    te_news, n_missing = news2_scores(X_te)
    # Try to match conformal recall (0.85) on calibration; if no cutoff >= 1
    # reaches it, record infeasibility and fall back to the clinical standard
    # (NEWS >= 5 urgent review) plus the max-recall cutoff-1 point.
    feasible = [c for c in range(1, 21)
                if recall_score(y_cal, (cal_news >= c).astype(int), zero_division=0) >= 0.85]
    cut = max(feasible) if feasible else 5
    news_pred = (te_news >= cut).astype(int)
    news_pred1 = (te_news >= 1).astype(int)
    news = {
        "cutoff": int(cut),
        "matched_recall_feasible": bool(feasible),
        "cal_recall": round(float(recall_score(y_cal, (cal_news >= cut).astype(int))), 4),
        "test_recall": round(float(recall_score(y_te, news_pred)), 4),
        "test_precision": round(float(precision_score(y_te, news_pred, zero_division=0)), 4),
        "cutoff1_test_recall": round(float(recall_score(y_te, news_pred1)), 4),
        "cutoff1_test_precision": round(float(precision_score(y_te, news_pred1, zero_division=0)), 4),
        "test_missing_vitals_rows": n_missing,
    }
    print(f"NEWS2: cutoff={cut} cal_recall={news['cal_recall']} "
          f"test_recall={news['test_recall']} test_precision={news['test_precision']} "
          f"(missing-vit rows={n_missing}/{len(X_te)})")

    out = {
        "data_source": f"{DATA_SOURCE_LABEL} (n={len(X)})",
        "split": {"seed": SEED, "n_train": len(y_tr), "n_cal": len(y_cal),
                  "n_test": len(y_te), "threshold": round(t, 6),
                  "n_cal_pos": n_cal_pos},
        "xgboost_test": test_metrics,
        "bootstrap_95ci": cis,
        "bootstrap_resamples": N_BOOT,
        "news2_baseline": news,
    }
    with open(MODEL_DIR / "severity_extras.json", "w") as f:
        json.dump(out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(test_proba, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = which == b
        if m.sum():
            xs.append(float(test_proba[m].mean()))
            ys.append(float(y_te[m].mean()))
            ns.append(int(m.sum()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.scatter(xs, ys, s=[max(10, n / 8) for n in ns])
    for x, yy, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, yy), fontsize=7)
    plt.xlabel("mean predicted P(High)")
    plt.ylabel("observed fraction High")
    plt.title(f"XGBoost reliability, seed-42 test (Brier={test_metrics['brier']:.4f}, "
              "dot labels = n)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "calibration_curve.png", dpi=150)
    plt.close()
    print("Saved severity_extras.json + calibration_curve.png")


if __name__ == "__main__":
    main()
