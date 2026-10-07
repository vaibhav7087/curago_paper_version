"""
ds/conformal_triage.py — Conformal recall-controlled triage (real data only).

FAIL-CLOSED: loads via train_triage.load_real() — missing raw data = hard error.

Protocol (split-conformal, pre-declared):
  1. Same 13,595 x 29 NHAMCS features as train_triage.py.
  2. Stratified split 60% train / 20% calibration / 20% test (seed = split seed).
     XGBoost (existing config + tree_method="hist", n_jobs=-1 for speed;
     scale_pos_weight recomputed on the TRAIN split only) fit on train.
  3. Calibration: scores = P(High) on calibration positives, sorted ascending;
     n = #positives, k = floor(alpha*(n+1)); threshold t = scores[k-1] (k>=1).
     Predict high severity iff p >= t. (Standard split-conformal construction:
     empirical miscoverage on calibration ~= alpha; marginal guarantee 1-alpha
     holds on average over calibration draws — NOT per split.)
  4. Report on the untouched test split: recall, under/over-triage, precision,
     AUC, abstention rate (predict.py rule max(p)<0.60, untouched), chosen t.
  5. Repeat for seeds 0..199 (same fits serve the alpha sweep {0.05,0.10,0.15,
     0.20,0.30} — no refits per alpha).

Writes incrementally (resume-safe): each finished seed appends its rows
(flushed); existing complete seeds are skipped unless --force. Final files are
sorted by seed and rewritten, so completed reruns are byte-identical.

Usage: python ds/conformal_triage.py [--seeds 0,1,42] [--force]

Outputs: models/conformal_repeats.csv, models/conformal_alpha_sweep.csv,
         models/conformal_recall_hist.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

from train_triage import load_real, DATA_SOURCE_LABEL

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPEATS_CSV = MODEL_DIR / "conformal_repeats.csv"
SWEEP_CSV = MODEL_DIR / "conformal_alpha_sweep.csv"
HIST_PNG = MODEL_DIR / "conformal_recall_hist.png"

MAIN_ALPHA = 0.15
SWEEP_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30]
ABSTAIN_CUTOFF = 0.60  # must match ds/predict.py

REPEAT_COLS = ["seed", "threshold", "spw", "test_recall", "test_under_triage",
               "test_precision", "test_over_triage", "test_auc",
               "test_abstention_rate", "n_cal_pos", "n_test", "n_test_pos",
               "Data Source"]
SWEEP_COLS = ["seed", "alpha", "threshold", "test_recall", "test_precision",
              "Data Source"]


def r6(x):
    return round(float(x), 6)


def build_pipe(spw):
    # Existing trainer config + hist/n_jobs for speed (disclosed in docs).
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            scale_pos_weight=spw, n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42, eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
        )),
    ])


def split_60_20_20(X, y, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def conformal_threshold(cal_pos_proba, alpha):
    p = np.sort(np.asarray(cal_pos_proba))
    n = len(p)
    k = int(np.floor(alpha * (n + 1)))
    return float(p[k - 1]) if k >= 1 else 0.0, n


def eval_at_threshold(test_proba, y_test, t):
    pred = (test_proba >= t).astype(int)
    return {
        "recall": recall_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, test_proba),
        "abstention": float((np.maximum(test_proba, 1 - test_proba) < ABSTAIN_CUTOFF).mean()),
    }


def run_seed(X, y, seed):
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = split_60_20_20(X, y, seed)
    neg, pos = np.bincount(y_tr)
    spw = float(neg / max(1, pos))
    pipe = build_pipe(spw)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    cal_proba = pipe.predict_proba(X_cal)[:, 1]
    test_proba = pipe.predict_proba(X_te)[:, 1]
    cal_pos = cal_proba[y_cal == 1]

    rep_rows, sweep_rows = [], []
    for alpha in SWEEP_ALPHAS:
        t, n = conformal_threshold(cal_pos, alpha)
        m = eval_at_threshold(test_proba, y_te, t)
        sweep_rows.append({
            "seed": seed, "alpha": alpha, "threshold": r6(t),
            "test_recall": r6(m["recall"]), "test_precision": r6(m["precision"]),
            "Data Source": f"{DATA_SOURCE_LABEL} (n={len(X)})",
        })
        if alpha == MAIN_ALPHA:
            rep_rows.append({
                "seed": seed, "threshold": r6(t), "spw": r6(spw),
                "test_recall": r6(m["recall"]),
                "test_under_triage": r6(1 - m["recall"]),
                "test_precision": r6(m["precision"]),
                "test_over_triage": r6(1 - m["precision"]),
                "test_auc": r6(m["auc"]),
                "test_abstention_rate": r6(m["abstention"]),
                "n_cal_pos": n, "n_test": len(y_te), "n_test_pos": int(y_te.sum()),
                "Data Source": f"{DATA_SOURCE_LABEL} (n={len(X)})",
            })
    return rep_rows, sweep_rows, fit_s


def load_done(path, key):
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(df[key].tolist())
    except Exception:
        return set()  # malformed/partial file -> rerun everything


def append_rows(path, rows, cols):
    import csv
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()


def finalize():
    for path, cols, keys in [(REPEATS_CSV, REPEAT_COLS, ["seed"]),
                             (SWEEP_CSV, SWEEP_COLS, ["seed", "alpha"])]:
        if path.exists():
            df = (pd.read_csv(path)
                    .drop_duplicates(subset=keys, keep="last")  # --force replaces
                    .sort_values(keys))
            df.to_csv(path, index=False)
    print(f"Finalized {REPEATS_CSV} + {SWEEP_CSV} (deduped, sorted)")


def plot_hist():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(REPEATS_CSV)
    rec = df["test_recall"].to_numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(rec, bins=20, edgecolor="black")
    plt.axvline(1 - MAIN_ALPHA, color="red", ls="--",
                label=f"target 1-alpha={1 - MAIN_ALPHA:.2f}")
    plt.axvline(rec.mean(), color="blue", ls="-",
                label=f"mean={rec.mean():.4f} (n={len(rec)})")
    plt.xlabel("test recall (per split)")
    plt.ylabel("splits")
    plt.title("Conformal triage: test recall over splits\n"
              "guarantee is marginal (average over calibration draws); "
              "single splits may fall below 0.85")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=150)
    plt.close()
    print(f"Saved {HIST_PNG} mean_recall={rec.mean():.4f} "
          f"frac_below_target={float((rec < 1 - MAIN_ALPHA).mean()):.3f}")


def print_sweep_means():
    df = pd.read_csv(SWEEP_CSV)
    print("\nAlpha sweep (mean over seeds):")
    for a, sub in df.groupby("alpha"):
        print(f"  alpha={a:.2f} target_recall={1 - a:.2f} "
              f"mean_recall={sub['test_recall'].mean():.4f} "
              f"mean_precision={sub['test_precision'].mean():.4f} (n={len(sub)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, range(200))),
                    help="comma-separated seeds (default 0..199)")
    ap.add_argument("--force", action="store_true",
                    help="recompute seeds even if already in CSV")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    data = load_real()
    X = data.drop(columns=["is_high_severity"])
    y = data["is_high_severity"].to_numpy()
    print(f"X shape: {X.shape}, positives: {int(y.sum())}")

    done = set() if args.force else load_done(REPEATS_CSV, "seed")
    todo = [s for s in seeds if s not in done]
    print(f"seeds requested={len(seeds)} done={len(done)} todo={len(todo)}")
    t_all = time.time()
    for i, seed in enumerate(todo):
        rep_rows, sweep_rows, fit_s = run_seed(X, y, seed)
        append_rows(REPEATS_CSV, rep_rows, REPEAT_COLS)
        append_rows(SWEEP_CSV, sweep_rows, SWEEP_COLS)
        r = rep_rows[0]
        el = time.time() - t_all
        print(f"[{i + 1}/{len(todo)}] seed={seed} fit={fit_s:.1f}s "
              f"t={r['threshold']} recall={r['test_recall']} "
              f"prec={r['test_precision']} abst={r['test_abstention_rate']} "
              f"elapsed={el:.0f}s", flush=True)
    finalize()
    plot_hist()
    print_sweep_means()


if __name__ == "__main__":
    main()
