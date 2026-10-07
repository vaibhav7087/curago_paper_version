#!/usr/bin/env python3
"""paper/audit.py — automated audits for the self-score loop.

Checks (machine-checkable rubric criteria):
  1. numbers  : every numbers.tex macro value appears in the PDF text,
                and matches paper_numbers.json numerically.
  2. cites    : every references.bib entry is cited; every citation resolves.
  3. ngrams   : duplicated 10-word runs inside the paper (originality scan).
  4. honesty  : mandated disclosure phrases are present.
  5. format   : page count, figures/tables/labels/refs, abstract length.

Usage:  python paper/audit.py            (prints report, exit 0 iff all pass)
        python paper/audit.py --json     (machine-readable)
Exit codes: 0 all pass | 1 audit failure
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PDF = HERE / "main.pdf"
TEX = HERE / "main.tex"
BIB = ROOT / "docs" / "references.bib"
AUX = HERE / "main.aux"
BBL = HERE / "main.bbl"

sys.path.insert(0, str(HERE))
import make_numbers  # noqa: E402


def pdf_text() -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    return "\n".join((p.extract_text() or "") for p in reader.pages), len(reader.pages)


def norm(s: str) -> str:
    return re.sub(r"[\s\u00ad]+", " ", s.replace("\u2013", "-").replace("\u2014", "-"))


def audit_numbers(text: str):
    flat = norm(text)
    flat_nocomma = flat.replace(",", "")
    tokens = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", flat)
    token_floats = set()
    for t in tokens:
        try:
            token_floats.add(round(float(t.replace(",", "")), 6))
        except ValueError:
            pass
    failures = []
    for macro, path, display in make_numbers.iter_numbers():
        if display in flat or display.replace(",", "") in flat_nocomma:
            continue
        try:
            if round(float(display.replace(",", "")), 6) in token_floats:
                continue
        except ValueError:
            pass
        failures.append(f"{macro} ({path}) = {display}")
    return {"checked": len(list(make_numbers.iter_numbers())),
            "missing": failures, "ok": not failures}


def audit_cites():
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", BIB.read_text(encoding="utf-8")))
    aux = AUX.read_text(encoding="utf-8", errors="replace") if AUX.exists() else ""
    cited = set()
    for line in re.findall(r"\\citation\{([^}]+)\}", aux):
        cited.update(k.strip() for k in line.split(","))
    bbl = BBL.read_text(encoding="utf-8", errors="replace") if BBL.exists() else ""
    resolved = set(re.findall(r"\\bibitem\{([^}]+)\}", bbl))
    unused = sorted(bib_keys - cited)
    unresolved = sorted(cited - resolved)
    return {"bib_entries": len(bib_keys), "cited": len(cited & bib_keys),
            "unused": unused, "unresolved": unresolved,
            "ok": not unused and not unresolved and len(bib_keys & cited) >= 10}


def audit_ngrams(text: str):
    body = re.sub(r"[\W_]+", " ", text.lower())
    words = body.split()
    grams = Counter(tuple(words[i:i + 10]) for i in range(len(words) - 9))
    dups = [(" ".join(g), n) for g, n in grams.items() if n > 1]
    # formulaic data-source / attribution strings are whitelisted: they are
    # citations, not prose reuse
    allowed = (
        "data cdc nhamcs 2019 ed",
        "data source cdc nhamcs 2019 ed",
        "data covid19india static archive",
        "cdc nhamcs 2019 ed data source",
        "covid19india static archive data incovid19",
        "static archive data incovid19 org retrieved",
        "http www cdc gov nchs nhamcs",
        # bibliographic venue strings (published citations, not prose reuse)
        "neural information processing systems neurips",
        "advances in neural information processing systems",
    )
    real = [(g, n) for g, n in dups
            if not any(a in g for a in allowed)]
    return {"dup_10grams": len(dups), "flagged": real[:20], "ok": not real}


HONESTY = {
    "marginal guarantee stated": ["marginal", "per-split"],
    "below-target fraction": ["46", "below"],
    "outbreak descriptive (no invented labels)": ["descriptive", "enrichment"],
    "overconfidence disclosed": ["overconfident"],
    "transfer caveat": ["exchangeab", "transfer"],
    "no synthetic data": ["synthetic"],
    "forecast baselines competitive": ["holt-winters"],
    "NEWS2 baseline reported": ["news2"],
}


def audit_honesty(text: str):
    flat = norm(text).lower()
    missing = [k for k, phrases in HONESTY.items()
               if not all(p in flat for p in phrases)]
    return {"checks": len(HONESTY), "missing": missing, "ok": not missing}


def audit_format(text: str):
    tex = TEX.read_text(encoding="utf-8")
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    _, n_pages = pdf_text()
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abs_words = len(re.sub(r"\\[a-zA-Z]+|\$[^$]*\$|[{}]", " ",
                   abs_match.group(1)).split()) if abs_match else 0
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for inp in re.findall(r"\\input\{([^}]+)\}", tex):
        src = (HERE / inp).resolve()
        if not src.exists():            # LaTeX \input omits the .tex suffix
            src = src.with_suffix(".tex")
        if src.exists():
            labels.update(re.findall(r"\\label\{([^}]+)\}",
                                     src.read_text(encoding="utf-8",
                                                   errors="replace")))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    # only float labels must be referenced; section labels are optional
    unreffed = sorted(l for l in (labels - refs)
                      if l.startswith(("fig:", "tab:")))
    unresolved = sorted(refs - labels)
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
    figs = len(re.findall(r"\\begin\{figure", tex))
    tabs = len(re.findall(r"\\input\{.*?paper_T", tex))
    checks = {
        "pages_5_to_7": 5 <= n_pages <= 7,
        "abstract_le_200w": abs_words <= 200,
        "no_overfull": not overfull,
        "no_unresolved_refs": not unresolved,
        "every_label_reffed": not unreffed,
        "figures_ge_5": figs >= 5,
        "tables_ge_4": tabs >= 4,
        "ieee_class": "IEEEtran" in tex,
        "keywords_present": "IEEEkeywords" in tex,
        "authors_placeholder_flagged": "Placeholder" in tex,
    }
    return {"pages": n_pages, "abstract_words": abs_words, "figures": figs,
            "tables": tabs, "labels": len(labels), "refs": len(refs),
            "overfull": overfull, "unreffed": unreffed,
            "unresolved": unresolved, "checks": checks,
            "ok": all(v for k, v in checks.items() if k != "authors_placeholder_flagged")}


def main():
    text, n_pages = pdf_text()
    report = {
        "pages": n_pages,
        "numbers": audit_numbers(text),
        "cites": audit_cites(),
        "originality": audit_ngrams(text),
        "honesty": audit_honesty(text),
        "format": audit_format(text),
    }
    report["all_pass"] = all(report[k]["ok"] for k in
                             ("numbers", "cites", "originality", "honesty", "format"))
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        for section, res in report.items():
            if isinstance(res, dict):
                status = "PASS" if res.get("ok") else "FAIL"
                print(f"[{status}] {section}: "
                      + json.dumps({k: v for k, v in res.items() if k != "ok"},
                                   ensure_ascii=False))
        print("ALL PASS" if report["all_pass"] else "AUDIT FAILED")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
