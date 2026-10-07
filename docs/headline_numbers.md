# Headline numbers (for paper + slides — all from committed artifacts)

## Module 1 — severity scoring (CDC NHAMCS 2019 ED, n=13,595)

- Baseline (t=0.5, 5-fold CV): XGB AUC **0.732**, recall 0.513, under-triage **0.487**.
- Tuned (t=0.25, pre-declared recall≥0.80 rule): recall **0.845**, under-triage
  **0.155**, precision 0.207, abstention 0.235.
- Conformal (alpha=0.15, 200 splits): mean test recall **0.850**; seed-42 test
  recall **0.913** (95% CI 0.886–0.940), precision 0.187, AUC 0.739, Brier 0.158.
- NEWS2 vitals-only baseline: cutoff ≥5 → recall 0.138; best recall (cutoff 1)
  0.669 — cannot match 0.85 at any cutoff.
- Honest limits: no held-out set for the baseline; conformal guarantee is
  marginal and US-data-internal; model overconfident (probabilities not risks).

## Module 2 — outbreak detection (126,712 district-weeks, 840 districts)

- 5% flag budget by construction (never a "detection rate").
- Delta window: **7× enrichment** (0.250 vs 0.036), **39.8%** district coverage.
- Method comparison: confirmed-rule halves background (0.013) at −1 wk lead;
  z-baseline leads earliest (4 wk) at 4× background (0.093).
- Honest limits: descriptive only, static archive, no labeled ground truth.

## Supplementary — demand forecast (69 months, 12-step holdout)

- Holt-Winters 4/8, seasonal naive 2/8, LightGBM 2/8 (MAE). Naive baselines
  compete — reported as a robustness check.
