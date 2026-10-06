"""
ds/data/incovid_to_weekly.py — Convert incovid19 district timeseries to weekly new cases.

FAIL-CLOSED: requires ds/data/raw/districts.csv (download from
https://data.incovid19.org/csv/latest/districts.csv — open, no auth).

Schema note: incovid Confirmed/Recovered/Deceased are CUMULATIVE. This script:
  1. sorts by district+date (dropping 68 exact-duplicate rows present in the file),
  2. diffs to daily new cases (negative corrections clipped to 0;
     each district's first observation has no predecessor, so week 1
     slightly undercounts — negligible over a ~170-week series),
  3. aggregates to Sunday-ending weeks (W-SUN, CDC epi-week style),
  4. writes ds/data/raw/district_weekly_cases.csv
     (district,week,case_count,disease) for train_outbreak.py.

Window observed in file: 2020-04-26 to 2023-08-22 (retrieved 2026-10-07) - static archive.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "raw" / "districts.csv"
DST = HERE / "raw" / "district_weekly_cases.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found. Download:\n"
            "  https://data.incovid19.org/csv/latest/districts.csv\n"
            "No synthetic outbreak data is used in this repo."
        )
    df = pd.read_csv(SRC, parse_dates=["Date"])
    df = df.drop_duplicates()  # file contains exact-duplicate rows (verified: 68)
    df = df.sort_values(["State", "District", "Date"])
    df["new_cases"] = df.groupby(["State", "District"])["Confirmed"].diff()
    df["new_cases"] = df["new_cases"].clip(lower=0)
    df["week"] = df["Date"].dt.to_period("W-SUN").dt.start_time.dt.date
    # qualify district names with state — "Unknown"/same names exist across states
    df["district"] = df["State"].astype(str) + " | " + df["District"].astype(str)
    out = (df.groupby(["district", "week"], as_index=False)["new_cases"].sum()
           .rename(columns={"new_cases": "case_count"}))
    out["disease"] = "covid19"
    out = out.sort_values(["district", "week"])
    out.to_csv(DST, index=False)
    print(f"Wrote {DST} rows={len(out)} "
          f"window={df['Date'].min().date()}..{df['Date'].max().date()} "
          f"districts={out['district'].nunique()}")


if __name__ == "__main__":
    main()
