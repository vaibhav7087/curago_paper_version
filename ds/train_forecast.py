"""
ds/train_forecast.py — Demand forecast (notebook-only, no pkl).

FAIL-CLOSED: no synthetic fallback. Missing raw data = hard error.

Raw data (manual download, needs Kaggle account — CC BY-NC 4.0, Milan Zdravkovic):
  https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data
  -> place salesmonthly.csv in ds/data/raw/ (schema: datum + ATC drug columns)

Outputs: models/demand_forecast_comparison.png, models/demand_forecast_metrics.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
RAW = HERE / "data" / "raw" / "salesmonthly.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DRUGS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
DATA_SOURCE_LABEL = "Kaggle milanzdravkovic/pharma-sales-data (CC BY-NC 4.0)"


def load_real():
    if not RAW.exists():
        raise FileNotFoundError(
            "ds/data/raw/salesmonthly.csv not found.\n"
            "Manual download (Kaggle account required):\n"
            "  https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data\n"
            "License: CC BY-NC 4.0 (attribution required). No synthetic data is used in this repo."
        )
    df = pd.read_csv(RAW)
    for cand in ["datum", "date", "Date", "DATUM"]:
        if cand in df.columns:
            df["date"] = pd.to_datetime(df[cand])
            break
    if "date" not in df.columns:
        raise ValueError(f"No date column in {list(df.columns)}")
    df = df.sort_values("date")
    missing = [d for d in DRUGS if d not in df.columns]
    if missing:
        raise ValueError(
            f"salesmonthly.csv missing expected drug columns: {missing}. "
            f"Found: {list(df.columns)}. Refusing to synthesize."
        )
    if len(df) < 24:
        raise ValueError(f"Need >=24 monthly rows for 12-step holdout, got {len(df)}")
    print(f"Loaded {RAW} rows={len(df)} drugs={DRUGS} ({DATA_SOURCE_LABEL})")
    return df[["date"] + DRUGS]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error

    df = load_real()
    print(df.head().to_string())
    print(df.shape, df.columns.tolist())

    results = []
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    for idx, drug in enumerate(DRUGS):
        series = df.set_index("date")[drug].dropna().asfreq("MS")
        train, test = series[:-12], series[-12:]

        naive_pred = train[-12:].values
        mae_naive = mean_absolute_error(test, naive_pred)

        try:
            hw = ExponentialSmoothing(train, seasonal="add", seasonal_periods=12).fit(optimized=True, use_brute=True)
            hw_pred = np.asarray(hw.forecast(12))
            mae_hw = mean_absolute_error(test, hw_pred)
        except Exception as e:
            print(f"{drug} HW failed ({e}), using naive")
            hw_pred, mae_hw = naive_pred, mae_naive

        def make_lag(s, n_lags=12):
            d = pd.DataFrame({"y": s})
            for lag in range(1, n_lags + 1):
                d[f"lag_{lag}"] = d["y"].shift(lag)
            d["month"] = d.index.month
            return d.dropna()

        dfl = make_lag(series)
        X, y = dfl.drop(columns=["y"]), dfl["y"]
        Xtr, ytr, Xte, yte = X[:-12], y[:-12], X[-12:], y[-12:]
        lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb.fit(Xtr, ytr)
        lgb_pred = lgb.predict(Xte)
        mae_lgb = mean_absolute_error(yte, lgb_pred)

        results.append({"Drug (ATC)": drug, "MAE Seasonal Naive": round(float(mae_naive), 2),
                        "MAE Holt-Winters": round(float(mae_hw), 2), "MAE LightGBM": round(float(mae_lgb), 2)})
        ax = axes[idx]
        ax.plot(test.index, test.values, "k-", label="Actual", linewidth=2)
        ax.plot(test.index, hw_pred, "b--", label=f"HW (MAE={mae_hw:.1f})")
        ax.plot(test.index, lgb_pred, "r--", label=f"LGB (MAE={mae_lgb:.1f})")
        ax.set_title(drug)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "demand_forecast_comparison.png", dpi=150)
    plt.close()
    rdf = pd.DataFrame(results)
    rdf["Data Source"] = DATA_SOURCE_LABEL
    print("\nDEMAND FORECAST METRICS:")
    try:
        print(rdf.to_markdown(index=False))
    except Exception:
        print(rdf.to_string(index=False))
    rdf.to_csv(MODEL_DIR / "demand_forecast_metrics.csv", index=False)
    print("Saved demand_forecast_comparison.png + demand_forecast_metrics.csv")


if __name__ == "__main__":
    main()
