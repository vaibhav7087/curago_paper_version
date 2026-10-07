# Self-score v2 — iteration 2 (format fixes)

**Build:** `paper/main.pdf`, 5 pages, 5 figures + 4 tables, 13 references.
**Audits:** `python paper/audit.py` → ALL PASS (numbers 48/48, cites 13/13,
originality 0 flagged, honesty 8/8, format checks pass: 5 pp, no overfull,
all 21 labels referenced, no unresolved refs).
**Date:** 2026-10-08.

## Fixes applied since v1

- **G1 (page numbers):** `\pagestyle{plain}` after `\maketitle` + `\thispagestyle{plain}`
  → centered footer numbers verified on pages 1–5 (text extraction + render).
- **G2 (doubled punctuation):** trailing periods stripped from Discussion
  `\paragraph` args → PDF text contains `What holds:` / `Limitations:` with
  zero `\w.:` matches.
- **G3 (page balance):** `\IEEEtriggeratref{8}` removed, float spacing tightened
  (`\textfloatsep` 14 pt, `\floatsep`/`\intextsep` 10 pt), figure width 3.4→3.15 in,
  ~0.5 column of redundant prose trimmed → 5 pages, both columns of p4 fill
  96 %/91 % and p5 fills 96 %/85 % of page height (was: near-empty page 6).
- **G4 (caption echo):** §IV-D body no longer repeats "7.0×"; it now reads
  "hit rate 0.25 inside … versus 0.036 outside — the largest concentration of
  any method". `7.0×` remains only where it is the headline: abstract,
  contribution bullet, Fig. 5 caption, Table IV caption, conclusion.

## Rubric scores (0–5 each, weighted to 100)

| # | Criterion | W | Score | Weighted | Critical | Evidence |
|---|-----------|---|-------|----------|----------|----------|
| 1 | Number fidelity | 20 | 5 | 20 | yes | 48/48 macros in PDF match `paper_numbers.json`; all injected by `make_numbers.py`, zero hand-typing; reruns byte-identical |
| 2 | IEEE format compliance | 20 | 4 | 16 | yes | IEEEtran conference, **5 pp** with **page numbers 1–5**, abstract 171 w, numbered refs, 21 labels / all floats referenced, 0 overfull, 0 undefined refs, columns well filled — **deduction:** float clustering, Tables I–III all stack at the top of page 4 (compliant but table-dense; could distribute better) |
| 3 | Citation/acknowledgement coverage | 15 | 5 | 15 | yes | 13/13 bib entries cited and resolved; every dataset, library, method cited; Data Availability + licenses section present |
| 4 | Originality | 15 | 5 | 15 | yes | 0 flagged duplicated 10-grams after whitelist (attribution/venue strings); prose written from repo artifacts, no verbatim quotes |
| 5 | Honesty/accuracy of claims | 10 | 5 | 10 | yes | 8/8 mandated disclosures present (marginal guarantee, 46 % below target, descriptive-only outbreak, Brier overconfidence, transfer caveat, no synthetic data, naive baselines, NEWS2) |
| 6 | Readability/clarity | 10 | 4 | 8 | no | Punctuation defect fixed, prose tightened, §IV-D de-echoed — **residual:** Fig. 2 caption restates AUC 0.719/0.732 already given in §IV-A body; Fig. 4 caption restates 0.850/46 % already given in §IV-C body |
| 7 | Results completeness | 10 | 4 | 8 | no | Both modules + limitations + availability + supplementary pointers in main body; SHAP, calibration, T7 cohort, T5 NEWS2, TS1 forecast live as referenced repo artifacts; byline placeholder (user choice) |
| | **Total** | 100 | | **92** | | |

## Gate check

- Weighted total **92 ≥ 85** ✓
- Criticals: #1=5, #2=4, #3=5, #4=5, #5=5 — **all ≥ 4** ✓
- Strict increase over v1 (88 → 92) ✓
- **Result: PASS gate for iteration 2** — iterate once more for polish.

## Gap list driving iteration 3

1. **G5 (readability): caption↔body numeric echoes** for Fig. 2 (0.719/0.732)
   and Fig. 4 (0.850, 46 %). Rephrase captions to describe what is plotted
   (axes, protocol, marker meaning) and point to the text for the numbers,
   keeping the mandatory `Data:` lines.
2. **G6 (format polish): table clustering on page 4.** Attempt moving the
   Table I environment earlier in source (and/or `[!t]` placement hints) so
   Tables I–II spread across pages 3–4; revert if it disturbs the 5-page
   balance or audit.

## Decisions

- Placeholder byline retained (user's choice) — non-blocking, tracked.
- Float spacing stays at iteration-2 values (working, balanced).
