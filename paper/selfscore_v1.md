# Self-score v1 — iteration 1 (draft)

**Build:** `paper/main.pdf`, 6 pages, 5 figures + 4 tables, 13 references.
**Audits:** `python paper/audit.py` → ALL PASS (numbers 48/48, cites 13/13,
originality 0 flagged, honesty 8/8, format checks pass).
**Date:** 2026-10-08.

## Rubric scores (0–5 each, weighted to 100)

| # | Criterion | W | Score | Weighted | Critical | Evidence |
|---|-----------|---|-------|----------|----------|----------|
| 1 | Number fidelity | 20 | 5 | 20 | yes | 48/48 macros present in PDF text and equal to `paper_numbers.json`; all values generator-injected (`make_numbers.py`), zero hand-typing |
| 2 | IEEE format compliance | 20 | 3 | 12 | yes | IEEEtran conference, 6 pp (target 5–7), abstract 171 w, numbered refs, 21 labels / all floats referenced, 0 overfull, 0 undefined refs — **but: NO page numbers anywhere** (IEEEtran conference blanks head/foot by design; not yet overridden) **and page 6 is unbalanced** (only refs [8]–[13], right column empty) |
| 3 | Citation/acknowledgement coverage | 15 | 5 | 15 | yes | 13/13 bib entries cited and resolved; every dataset (NHAMCS, covid19india, Kaggle+license), library (scikit-learn, XGBoost, LightGBM, statsmodels, SHAP), method (conformal, NEWS2, OpenDengue) cited; Data Availability + licenses section present |
| 4 | Originality | 15 | 5 | 15 | yes | 0 flagged duplicated 10-grams after whitelist (whitelisted = dataset attribution strings + published venue strings); prose is original, written from repo artifacts; no verbatim quotes |
| 5 | Honesty/accuracy of claims | 10 | 5 | 10 | yes | 8/8 mandated disclosures present: marginal guarantee, 46% below target, descriptive-only outbreak, overconfidence/Brier, exchangeability/transfer caveat, no synthetic data, naive baselines competitive, NEWS2 baseline |
| 6 | Readability/clarity | 10 | 4 | 8 | no | Abstract tight, sections flow, notation consistent — **defect:** Discussion `\paragraph` renders doubled punctuation ("What holds.:", "Limitations.:"); a couple of caption/body echoes remain |
| 7 | Results completeness | 10 | 4 | 8 | no | Both modules + limitations + availability + supplementary pointers in main body; SHAP, calibration, cohort table T7, NEWS2 table T5, forecast table TS1 live as repo artifacts (referenced by artifact name), byline still placeholder (user choice) |
| | **Total** | 100 | | **88** | | |

## Gate check

- Weighted total **88 ≥ 85** ✓
- Criticals: #2 = **3 < 4** ✗
- **Result: FAIL gate → iterate** (format critical below floor)

## Gap list driving iteration 2

1. **G1 (critical): page numbers missing on all 6 pages.** Fix: `\pagestyle{plain}`
   after `\maketitle` (IEEEtran conference blanks `\@oddhead`/`\@oddfoot` in
   `ps@headings`; kernel `plain` restores centered footer number).
2. **G2 (critical-adjacent, readability): doubled punctuation** in Discussion
   run-in paragraphs: strip trailing periods from `\paragraph{...}` args so
   IEEEtran's appended colon reads correctly.
3. **G3 (format): page 6 nearly empty** (refs [8]–[13] left column only).
   Fix: tighten float spacing (`\textfloatsep`/`\floatsep` reductions), trim
   ~0.5 column of redundant prose (Related Work compression, caption/body
   de-echo) to land a balanced 5-page or well-filled 6-page layout.
4. **G4 (polish): caption/body echo** — Fig. 5 caption and §IV-D both say
   "enrich Delta 7.0×"; rephrase body once more.

## Decisions

- Placeholder byline retained (user's choice) — tracked as known non-blocking gap.
- SHAP/calibration/forecast figures stay supplementary (space budget).
