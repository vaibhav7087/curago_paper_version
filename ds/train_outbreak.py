"""
ds/train_outbreak.py — IsolationForest outbreak detection.

FAIL-CLOSED: no synthetic fallback. Missing raw data = hard error.

Raw data (choose one, machine-readable CSV only — no ProMED/WHO PDFs):
  Preferred: https://data.incovid19.org/csv/latest/districts.csv
    (static archive, observed window 2020-04-26 to 2023-08-22 - cite window + retrieval date;
     schema has district + Confirmed/Recovered/Deceased timeseries; resample to weekly)
  Or place ds/data/raw/district_weekly_cases.csv with schema:
    district,week,case_count,disease

Outputs: models/outbreak_iso.pkl, models/outbreak_features.json
"""
import joblib
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
RAW = HERE / "data" / "raw" / "district_weekly_cases.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_SOURCE_LABEL = "data.incovid19.org district timeseries (2020-04-26..2023-08-22, retrieved 2026-10-07)"


def load_real():
    if not RAW.exists():
        raise FileNotFoundError(
            "ds/data/raw/district_weekly_cases.csv not found.\n"
            "Preferred source (open, no auth):\n"
            "  https://data.incovid19.org/csv/latest/districts.csv\n"
            "Resample to district,week,case_count,disease weekly schema and save.\n"
            "NOTE: static archive (observed window 2020-04-26..2023-08-22) - cite window + retrieval date.\n"
            "No synthetic outbreak data is used in this repo."
        )
    df = pd.read_csv(RAW)
    required = {"district", "week", "case_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"district_weekly_cases.csv missing columns: {missing}")
    if df.empty:
        raise ValueError("district_weekly_cases.csv is empty")
    print(f"Loaded {RAW} rows={len(df)} ({DATA_SOURCE_LABEL})")
    return df


def main():
    df = load_real().sort_values(["district", "week"]).copy()
    df["rolling_mean"] = df.groupby("district")["case_count"].transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    df["rolling_std"] = df.groupby("district")["case_count"].transform(
        lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    df["week_of_year"] = pd.to_datetime(df["week"]).dt.isocalendar().week.astype(int)
    df["ratio_to_mean"] = df["case_count"] / df["rolling_mean"].clip(lower=1)
    features = ["case_count", "rolling_mean", "rolling_std", "week_of_year", "ratio_to_mean"]
    X = df[features].fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    df["anomaly_score"] = iso.fit_predict(X)
    df["is_outbreak"] = (df["anomaly_score"] == -1).astype(int)
    print(f"Flagged outbreaks: {int(df['is_outbreak'].sum())}/{len(df)}")
    print(df[df["is_outbreak"] == 1].head(10).to_string())
    import json
    joblib.dump(iso, MODEL_DIR / "outbreak_iso.pkl")
    with open(MODEL_DIR / "outbreak_features.json", "w") as f:
        json.dump(features, f)
    # Evidence artifact: every flagged district-week (small; full frame stays in raw).
    # Enables the Delta-window evaluation in ds/evaluate_outbreak.py.
    flagged = df.loc[df["is_outbreak"] == 1,
                     ["district", "week", "case_count", "disease"]].copy()
    flagged.to_csv(MODEL_DIR / "outbreak_flagged_weeks.csv", index=False)
    print("Saved outbreak_iso.pkl + outbreak_features.json + "
          f"outbreak_flagged_weeks.csv ({len(flagged)} rows)")


if __name__ == "__main__":
    from sklearn.ensemble import IsolationForest
    main()
