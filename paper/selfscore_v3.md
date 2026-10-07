# Self-score v3 — iteration 3 (polish + format exports) — FINAL

**Build:** `paper/main.pdf`, 5 pages, 5 figures + 4 tables, 13 references.
**Audits:** `python paper/audit.py` → ALL PASS (numbers 48/48, cites 13/13,
originality 0 flagged, honesty 8/8, format: 5 pp, no overfull, all 21 labels
referenced, no unresolved refs).
**Exports:** `main.md` (0 unhandled LaTeX), `main.docx` (94 paras, 4 tables,
5 images, 23,095 chars) — both validate via `--check`.
**Date:** 2026-10-08.

## Fixes applied since v2

- **G5 (readability): caption↔body de-echo.** Fig. 2 caption no longer
  restates AUC 0.719/0.732 (legend carries them; body states and interprets
  them); Fig. 4 caption no longer restates mean 0.850 / 46 % (body and
  abstract own those numbers). Captions now describe what is plotted
  (protocol, markers, target line) plus the mandatory `Data:` line.
  Verified in PDF text: `chance diagonal` present, old caption strings absent.
- **G6 (format): float distribution.** Table I's `table*` input moved into
  page-2 text (end of §III-B) so it catches the p2→p3 float break. Result:
  Tables I / II+III / IV now land on pages 3 / 4 / 5 (was 1+2+3 stacked on
  page 4). Page count stays 5; column fills unchanged (L 96 %, R 91/91/91/91/85 %);
  audit ALL PASS. (A first attempt moving Table I within page-3 text had no
  effect — LaTeX defers double-column floats past their source page — hence
  the second move.)
- **Content accuracy (honesty):** §III-C said the conformal threshold was the
  ⌊α(n+1)⌋-th smallest *nonconformity quantile* among positives. The
  implementation (`ds/conformal_triage.py: conformal_threshold`) sorts the
  calibration **probabilities** ascending and takes `p[⌊α(n+1)⌋-1]` — i.e. the
  ⌊α(n+1)⌋-th smallest **predicted probability** (the α lower tail ⇒ nominal
  recall 1−α). Wording corrected to match code exactly; `nonconformity` no
  longer appears in the PDF.
- **Format exports (user request):** `export_md.py` (LaTeX→Markdown,
  macro-injected numbers, `.bbl`-ordered citations, fail-closed on unknown
  commands) and `export_docx.py` (Markdown→Word with native headings,
  tables, figures, bullets; validates sections/numbers/images). README
  documents all deliverable formats.

## Rubric scores (0–5 each, weighted to 100)

| # | Criterion | W | Score | Weighted | Critical | Evidence |
|---|-----------|---|-------|----------|----------|----------|
| 1 | Number fidelity | 20 | 5 | 20 | yes | 48/48 macros in PDF = `paper_numbers.json`; same generator feeds `numbers.tex` and `main.md`; zero hand-typing in any format; seed-42 byte-identical reruns |
| 2 | IEEE format compliance | 20 | 5 | 20 | yes | IEEEtran conference, 5 pp with page numbers 1–5, abstract 171 w, 21 labels/all referenced, 0 overfull, 0 unresolved refs; **floats distributed 1/2/1 (Tables I / II+III / IV on pp. 3/4/5)**, columns fill 85–96 %; G1–G3, G6 closed |
| 3 | Citation/acknowledgement coverage | 15 | 5 | 15 | yes | 13/13 bib entries cited and resolved in PDF, MD (`[n]` from `.bbl` order), and DOCX; every dataset/library/method cited; Data Availability + licenses section present |
| 4 | Originality | 15 | 5 | 15 | yes | 0 flagged duplicated 10-grams after whitelist; prose written from repo artifacts; all three text formats derive from one source (no divergent hand copies) |
| 5 | Honesty/accuracy of claims | 10 | 5 | 10 | yes | 8/8 mandated disclosures present; methods wording now matches implementation exactly (probability quantile fix); descriptive-only outbreak, marginal-guarantee, overconfidence caveats intact |
| 6 | Readability/clarity | 10 | 5 | 10 | no | G2 punctuation + G5 caption echoes closed; quotes/accents normalized in exports; captions self-describing, body owns the numbers; MD/DOCX render clean prose with 0 stray LaTeX |
| 7 | Results completeness | 10 | 4 | 8 | no | Both modules + limitations + availability + supplementary pointers in main body; SHAP, calibration, T7 cohort, T5 NEWS2, TS1 forecast live as referenced repo artifacts; byline placeholder (user choice) — only remaining deduction |
| | **Total** | 100 | | **98** | | |

## Gate check

- Weighted total **98 ≥ 85** ✓
- Criticals: #1=5, #2=5, #3=5, #4=5, #5=5 — **all ≥ 4** ✓
- Strict increase: 88 (v1) → 92 (v2) → **98 (v3)** ✓
- **Result: PASS — loop complete (3 of 3 iterations).**

## Remaining non-blocking items

1. Placeholder byline (user's decision — replace before hard copy if real
   names become available).
2. Supplementary figures (SHAP, calibration, forecast) remain repository
   artifacts by space budget; captions in body point to them by name.
3. Legacy `.doc` not emitted (no Word on build machine); DOCX is the
   editable format.
