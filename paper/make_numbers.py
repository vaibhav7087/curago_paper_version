#!/usr/bin/env python3
"""paper/make_numbers.py — generate numbers.tex from paper_numbers.json.

Every numeric claim in the paper must come through a macro defined here, so
numbers are never hand-typed in main.tex. Fail-closed: any missing JSON key
aborts with exit 1. Deterministic: byte-identical output on every run.

Usage:
  python paper/make_numbers.py            # writes paper/numbers.tex
  python paper/make_numbers.py --check    # verify existing numbers.tex is current

Also importable by audit scripts:
  from make_numbers import MACROS, iter_numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parent / "ds" / "models" / "paper" / "paper_numbers.json"
OUT_PATH = HERE / "numbers.tex"

# macro -> (json dotted path, format spec or None, transform or None)
#   format spec: Python format() spec applied to the value (e.g. ".3f", ".1f")
#   transform:   "pct" multiplies by 100 first (for percent-phrased claims)
MACROS: dict[str, tuple[str, str | None, str | None]] = {
    # ---- module 1: triage (CDC NHAMCS 2019 ED)
    "Nraw":        ("module1_triage.n_raw", None, None),
    "Nanalysis":   ("module1_triage.n_analysis", None, None),
    "Prev":        ("module1_triage.prevalence", None, None),
    "AUCxgb":      ("module1_triage.auc_xgb_cv", None, None),
    "AUClogreg":   ("module1_triage.auc_logreg_cv", None, None),
    "RecallT":     ("module1_triage.xgb_cv_t05_recall", None, None),
    "UnderT":      ("module1_triage.xgb_cv_t05_under_triage", None, None),
    "TunedT":      ("module1_triage.tuned_xgb_threshold", None, None),
    "TunedRecall": ("module1_triage.tuned_xgb_recall", None, None),
    "TunedPrec":   ("module1_triage.tuned_xgb_precision", None, None),
    "TunedUnder":  ("module1_triage.tuned_xgb_under_triage", None, None),
    "TunedAbst":   ("module1_triage.tuned_xgb_abstention", None, None),
    # percent-phrased variants of the same source values
    "UnderTPct":   ("module1_triage.xgb_cv_t05_under_triage", ".1f", "pct"),
    "TunedUnderPct": ("module1_triage.tuned_xgb_under_triage", ".1f", "pct"),
    # ---- module 1: split-conformal recall control
    "AlphaC":      ("module1_triage.conformal_alpha", None, None),
    "CMeanR":      ("module1_triage.conformal_mean_recall", ".3f", None),
    "CSdR":        ("module1_triage.conformal_sd_recall", ".3f", None),
    "CBelow":      ("module1_triage.conformal_frac_below_085", None, None),
    "CBelowPct":   ("module1_triage.conformal_frac_below_085", ".0f", "pct"),
    "CMeanT":      ("module1_triage.conformal_mean_threshold", None, None),
    "CMeanA":      ("module1_triage.conformal_mean_abstention", None, None),
    "CNseeds":     ("module1_triage.conformal_n_seeds", None, None),
    # ---- module 1: seed-42 reference split + NEWS2 baseline
    "SdT":         ("module1_triage.seed42_threshold", None, None),
    "SdR":         ("module1_triage.seed42_recall", None, None),
    "SdCIlo":      ("module1_triage.seed42_recall_ci95.0", None, None),
    "SdCIhi":      ("module1_triage.seed42_recall_ci95.1", None, None),
    "SdP":         ("module1_triage.seed42_precision", None, None),
    "SdAUC":       ("module1_triage.seed42_auc", None, None),
    "SdAbst":      ("module1_triage.seed42_abstention", None, None),
    "SdBrier":     ("module1_triage.seed42_brier", None, None),
    "NewsAtFiveR": ("module1_triage.news2_cutoff5_recall", None, None),
    "NewsAtFiveP": ("module1_triage.news2_cutoff5_precision", None, None),
    "NewsAtOneR":  ("module1_triage.news2_cutoff1_recall", None, None),
    "NewsAtOneP":  ("module1_triage.news2_cutoff1_precision", None, None),
    # ---- module 2: outbreak surveillance (covid19india archive)
    "DWeeks":      ("module2_outbreak.district_weeks", None, None),
    "NDistricts":  ("module2_outbreak.districts", None, None),
    "DeltaFlag":   ("module2_outbreak.iso_single_delta_flag_rate", None, None),
    "DeltaCov":    ("module2_outbreak.iso_single_delta_coverage", None, None),
    "LeadWk":      ("module2_outbreak.iso_single_delta_lead_weeks", "g", None),
    "DeltaEnr":    ("module2_outbreak.delta_enrichment", ".1f", None),
    "DeltaOut":    ("module2_outbreak.delta_outside_flag_rate", ".3f", None),
    "BgIso":       ("module2_outbreak.iso_single_background_rate", None, None),
    "BgIsoC":      ("module2_outbreak.iso_confirmed_background_rate", None, None),
    "BgZ":         ("module2_outbreak.zscore_background_rate", None, None),
    # ---- supplementary: demand forecast
    "Months":      ("supplement_forecast.months_analyzed", None, None),
    "HWWins":      ("supplement_forecast.hw_wins", None, None),
    "NaiveWins":   ("supplement_forecast.naive_wins", None, None),
    "LGBMWins":    ("supplement_forecast.lgbm_wins", None, None),
}


def dig(obj, dotted: str):
    """Fetch value at dotted path (supports .0/.1 indexing into lists)."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
    return cur


def fmt(value, spec: str | None, transform: str | None) -> str:
    if transform == "pct":
        value = value * 100
    if spec is not None:
        return format(value, spec)
    if isinstance(value, bool):            # guard: bool is int subclass
        raise TypeError("boolean values are not allowed in numbers.tex")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"unsupported value type: {type(value)!r}")


def load_numbers() -> dict:
    if not JSON_PATH.exists():
        sys.exit(
            f"FAIL-CLOSED: {JSON_PATH} missing — run `python ds/paper_tables.py` first."
        )
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def resolve_all(data: dict) -> dict[str, tuple[str, str]]:
    """Macro -> (json path, display string). Raises on any missing key."""
    out: dict[str, tuple[str, str]] = {}
    missing = []
    for macro, (path, spec, transform) in MACROS.items():
        try:
            value = dig(data, path)
        except (KeyError, IndexError, TypeError):
            missing.append(f"{macro} -> {path}")
            continue
        out[macro] = (path, fmt(value, spec, transform))
    if missing:
        sys.exit(
            "FAIL-CLOSED: keys missing from paper_numbers.json:\n  "
            + "\n  ".join(missing)
        )
    return out


def render(data: dict) -> str:
    resolved = resolve_all(data)
    lines = [
        "% paper/numbers.tex — GENERATED by make_numbers.py from",
        "% ds/models/paper/paper_numbers.json. DO NOT EDIT BY HAND.",
        "% Regenerate: python paper/make_numbers.py",
    ]
    for macro in MACROS:                       # stable, declared order
        path, display = resolved[macro]
        lines.append(f"% source: {path}")
        lines.append(f"\\newcommand{{\\{macro}}}{{{display}}}")
    return "\n".join(lines) + "\n"


def iter_numbers():
    """Yield (macro, json_path, display) for audits (imports MACROS order)."""
    data = load_numbers()
    for macro, (path, display) in resolve_all(data).items():
        yield macro, path, display


def main() -> None:
    text = render(load_numbers())
    if "--check" in sys.argv:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != text:
            sys.exit("numbers.tex is OUT OF DATE — run `python paper/make_numbers.py`")
        print("numbers.tex is current")
        return
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH} ({len(MACROS)} macros)")


if __name__ == "__main__":
    main()
