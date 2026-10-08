# Headline numbers (for paper + slides — all from committed artifacts)

## Module 1 — severity scoring (CDC NHAMCS 2019 ED, n=13,595)

- Primary model (pre-declared rule: min 5-fold OOF Brier among calibrated
  candidates): unweighted XGBoost + isotonic — AUC **0.742** (95% CI
  0.728–0.751), OOF Brier **0.113** vs **0.129** for a constant at train
  prevalence. Class-weighted predecessor 0.732 / Brier 0.165; LogReg 0.719
  (CI 0.708–0.731); paired AUC difference CI [0.012, 0.031].
- Default t=0.5 (primary): recall 0.141, under-triage **0.859**, FPR 0.016.
- Tuned pooled (rule: max precision with OOF recall ≥ 0.80): t=**0.11**,
  recall 0.800, precision 0.226, FPR 0.493, under-triage 0.200, abstention 0.042.
- **Nested cross-fitted** (outer 5-fold, threshold re-tuned on the other four
  folds only): thresholds **0.09–0.10**, recall **0.829**, precision 0.214,
  FPR **0.545**, under-triage **0.172**, abstention 0.042 — headline operating
  point in the paper.
- Split-conformal (alpha=0.15, 200 seeds, primary model):
  - marginal rule: mean recall **0.884** (SD 0.033), **12.5%** of splits below
    the 0.85 target, mean precision 0.196, mean threshold 0.094;
  - **PAC rule (Beta order statistic, delta=10%)**: mean recall **0.900**
    (SD 0.030), below-target share **3.5%** (≤ delta), precision 0.191, mean
    threshold 0.089.
- Seed-42 conformal reference split: t=0.0985, recall **0.886** (95% CI
  0.856–0.914), precision 0.199 (0.179–0.215), FPR 0.643, AUC 0.755,
  Brier 0.112 (constant 0.129), abstention 0.036.
- NEWS2 vitals-only baseline: cutoff ≥5 → recall 0.138; best recall (cutoff 1)
  0.669 — cannot match 0.85 at any cutoff.
- Limits: single-cohort (no external test set); conformal rests on
  exchangeability — the marginal rule is average-case, PAC bounds the
  below-target share at delta; probabilities beat the constant on Brier but
  are not validated as absolute risks.

## Module 2 — outbreak detection (126,712 district-weeks, 840 districts)

- 5% flag budget by construction (never a "detection rate").
- Delta window: **7× enrichment** (0.250 vs 0.036), **39.8%** district coverage.
- Method comparison: confirmed-rule halves background (0.013) at −1 wk lead;
  z-baseline leads earliest (4 wk) at 4× background (0.093).
- Limits: descriptive only, static archive, no labeled ground truth.

## Supplementary — demand forecast (69 months, 12-step holdout)

- Holt-Winters 4/8, seasonal naive 2/8, LightGBM 2/8 (MAE). Naive baselines
  compete — reported as a robustness check.
