"""
ds/outbreak_eval_v2.py — Module 2 evaluation upgrade (real data only).

FAIL-CLOSED: requires ds/data/raw/district_weekly_cases.csv and
ds/models/outbreak_flagged_weeks.csv (from train_outbreak.py).

Methods (IsolationForest model itself untouched):
  1. iso_single    — original flags (any flagged district-week = alert).
  2. iso_confirmed — alert only if a district is flagged 2 consecutive weeks
                     (exact 7-day spacing) AND >=2 other districts in the same
                     state are flagged that same week. Alert week = 2nd week.
  3. zscore        — baseline: z >= 2 on case_count vs 4-week rolling mean/std
                     (min_periods=4; std==0 -> z=0).

Windows: Delta 2021-04-01..2021-06-30, Omicron 2022-01-01..2022-03-31.
Background = district-weeks outside BOTH windows (called background rate, NOT
false-alarm rate — other real waves occurred outside the windows).

Per method per window: in-window flag rate, background rate, district coverage
(districts with >=1 alert in window / 840), median lead time in weeks from
first in-window alert to the district's peak week (max case_count week in
window, first on ties; districts without alerts excluded, n reported).

All results are DESCRIPTIVE (no labeled ground truth -> no precision/recall).

Usage: python ds/outbreak_eval_v2.py
Outputs: models/outbreak_eval_v2.json, models/outbreak_methods_compare.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
RAW = HERE / "data" / "raw" / "district_weekly_cases.csv"
FLAGGED = MODEL_DIR / "outbreak_flagged_weeks.csv"

WINDOWS = {"delta": ("2021-04-01", "2021-06-30"),
           "omicron": ("2022-01-01", "2022-03-31")}


def main():
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. See ds/data/README.md. "
            "No synthetic outbreak data is used in this repo.")
    if not FLAGGED.exists():
        raise FileNotFoundError(
            f"{FLAGGED} not found. Run ds/train_outbreak.py first.")
    full = pd.read_csv(RAW, parse_dates=["week"])
    flagged = pd.read_csv(FLAGGED, parse_dates=["week"])
    full[["state", "dist"]] = full["district"].str.split(" \\| ", n=1, expand=True)
    flagged[["state", "dist"]] = flagged["district"].str.split(" \\| ", n=1, expand=True)
    full = full.sort_values(["district", "week"]).copy()
    flagged = flagged.sort_values(["district", "week"]).copy()

    # Week universe per district (for rates: every district-week cell).
    full["in_delta"] = ((full["week"] >= WINDOWS["delta"][0])
                        & (full["week"] <= WINDOWS["delta"][1]))
    full["in_omicron"] = ((full["week"] >= WINDOWS["omicron"][0])
                          & (full["week"] <= WINDOWS["omicron"][1]))
    full["in_any"] = full["in_delta"] | full["in_omicron"]

    # --- Method 1: iso_single ---
    iso_weeks = set(zip(flagged["district"], flagged["week"].dt.date))

    # --- Method 2: iso_confirmed ---
    state_week_counts = (flagged.groupby([flagged["state"],
                                          flagged["week"].dt.date])["district"]
                         .nunique().rename("n_state"))
    confirmed = set()
    for dist, sub in flagged.groupby("district"):
        days = sorted(sub["week"].dt.date.tolist())
        dayset = set(days)
        for d in days:
            prev = d - pd.Timedelta(days=7)
            if prev not in dayset:
                continue
            st = sub["state"].iloc[0]
            others = int(state_week_counts.get((st, d), 0)) - 1
            if others >= 2:
                confirmed.add((dist, d))

    # --- Method 3: rolling z-score baseline (baseline = PRIOR 4 weeks only;
    # including the current week would let a surge inflate its own mean/std).
    g = full.groupby("district")["case_count"]
    roll_mean = g.transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean())
    roll_std = g.transform(lambda x: x.shift(1).rolling(4, min_periods=4).std())
    z = (full["case_count"] - roll_mean) / roll_std.replace(0, np.nan)
    z = z.fillna(0)
    z_weeks = set(zip(full.loc[(z >= 2).to_numpy(), "district"],
                      full.loc[(z >= 2).to_numpy(), "week"].dt.date))

    methods = {"iso_single": iso_weeks, "iso_confirmed": confirmed, "zscore": z_weeks}
    n_districts = int(full["district"].nunique())
    results = {"windows": {k: list(v) for k, v in WINDOWS.items()},
               "n_districts": n_districts, "methods": {}}
    for mname, alert_weeks in methods.items():
        alert = pd.Series(
            list(zip(full["district"], full["week"].dt.date))).isin(alert_weeks)
        stats = {}
        for wname, (s, e) in WINDOWS.items():
            in_w = full["in_delta"] if wname == "delta" else full["in_omicron"]
            a_in = alert & in_w
            cov_dists = full.loc[a_in, "district"].nunique()
            # lead time: first alert week -> peak week within window
            leads = []
            win_weeks = full.loc[in_w, ["district", "week", "case_count"]]
            peak_idx = win_weeks.groupby("district")["case_count"].idxmax()
            peak = dict(zip(win_weeks.loc[peak_idx, "district"],
                            pd.to_datetime(win_weeks.loc[peak_idx, "week"])))
            first = (full.loc[a_in, ["district", "week"]].groupby("district")["week"]
                     .min() if a_in.any()
                     else pd.Series(dtype="datetime64[ns]"))
            for dist, fa in first.items():
                pw = peak.get(dist)
                if pw is not None:
                    leads.append((pw - fa).days / 7.0)
            stats[wname] = {
                "alert_district_weeks": int(a_in.sum()),
                "window_district_weeks": int(in_w.sum()),
                "flag_rate_in_window": round(float(a_in.sum() / max(1, in_w.sum())), 4),
                "district_coverage": round(float(cov_dists / n_districts), 4),
                "districts_alerted": int(cov_dists),
                "median_lead_weeks": (round(float(np.median(leads)), 2) if leads else None),
                "n_lead_districts": len(leads),
            }
        bg = alert & ~full["in_any"]
        stats["background"] = {
            "alert_district_weeks": int(bg.sum()),
            "background_district_weeks": int((~full["in_any"]).sum()),
            "background_rate": round(float(bg.sum() / max(1, (~full["in_any"]).sum())), 4),
        }
        results["methods"][mname] = stats
        print(f"{mname}: " + "; ".join(
            f"{w} rate={stats[w]['flag_rate_in_window']} cov={stats[w]['district_coverage']} "
            f"lead={stats[w]['median_lead_weeks']}" for w in WINDOWS)
            + f"; background={stats['background']['background_rate']}")

    with open(MODEL_DIR / "outbreak_eval_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(methods)
    x = np.arange(len(names))
    dw, ow = 0.35, 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, wname in zip(axes, WINDOWS):
        vals = [results["methods"][m][wname]["flag_rate_in_window"] for m in names]
        covs = [results["methods"][m][wname]["district_coverage"] for m in names]
        ax.bar(x - dw / 2, vals, dw, label="in-window flag rate")
        ax.bar(x + dw / 2, covs, dw, label="district coverage")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_title(f"{wname} {WINDOWS[wname][0]}..{WINDOWS[wname][1]}")
        ax.legend(fontsize=8)
    fig.suptitle("Outbreak methods vs epidemic windows (descriptive — no ground truth)")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "outbreak_methods_compare.png", dpi=150)
    plt.close()
    print("Saved outbreak_eval_v2.json + outbreak_methods_compare.png")


if __name__ == "__main__":
    main()
