"""
ds/evaluate_outbreak.py — Delta-window evaluation for the IsolationForest flags.

FAIL-CLOSED: requires ds/models/outbreak_flagged_weeks.csv (from train_outbreak.py)
and ds/data/raw/district_weekly_cases.csv (denominator).

Method (honest, descriptive — NOT precision/recall, there is no labeled ground
truth): compare the flag rate inside India's Delta-wave window against the flag
rate outside it, plus district coverage (share of districts with >=1 flag in
window). If the detector is picking up real surges, the in-window rate should
be substantially above baseline.

Delta window: 2021-04-01..2021-06-30 (second wave rise/peak/decline, widely
documented; district peaks vary within it, so the whole wave is used rather
than a single peak week).

Output: models/outbreak_delta_eval.json (committed evidence).
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
RAW = HERE / "data" / "raw" / "district_weekly_cases.csv"
FLAGGED = MODEL_DIR / "outbreak_flagged_weeks.csv"

DELTA_START = "2021-04-01"
DELTA_END = "2021-06-30"


def main():
    if not FLAGGED.exists():
        raise FileNotFoundError(
            f"{FLAGGED} not found. Run ds/train_outbreak.py first.")
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. See ds/data/README.md. "
            "No synthetic outbreak data is used in this repo.")
    flagged = pd.read_csv(FLAGGED, parse_dates=["week"])
    full = pd.read_csv(RAW, parse_dates=["week"])
    full["in_delta"] = (full["week"] >= DELTA_START) & (full["week"] <= DELTA_END)
    flagged["in_delta"] = (flagged["week"] >= DELTA_START) & (flagged["week"] <= DELTA_END)

    in_win = full[full["in_delta"]]
    out_win = full[~full["in_delta"]]
    in_flag = flagged[flagged["in_delta"]]
    out_flag = flagged[~flagged["in_delta"]]

    eval_out = {
        "delta_window": [DELTA_START, DELTA_END],
        "district_weeks_in_window": int(len(in_win)),
        "flagged_in_window": int(len(in_flag)),
        "flag_rate_in_window": round(len(in_flag) / max(1, len(in_win)), 4),
        "district_weeks_outside_window": int(len(out_win)),
        "flagged_outside_window": int(len(out_flag)),
        "flag_rate_outside_window": round(len(out_flag) / max(1, len(out_win)), 4),
        "districts_total": int(full["district"].nunique()),
        "districts_with_flag_in_window": int(in_flag["district"].nunique()),
    }
    eval_out["district_coverage_in_window"] = round(
        eval_out["districts_with_flag_in_window"] / max(1, eval_out["districts_total"]), 4)
    with open(MODEL_DIR / "outbreak_delta_eval.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    print(json.dumps(eval_out, indent=2))
    print("Saved outbreak_delta_eval.json")


if __name__ == "__main__":
    main()
