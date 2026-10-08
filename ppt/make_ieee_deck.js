/* IEEE conference paper deck â€” ICFACT-template structure, our paper's content.
 * Structure mirrors ppt/Updated-Paper ID_ICFACT_2026_Final_Template (1).pptx
 * (23 slides: Title, Outline, Abstract, Intro, Problem, Lit x3, Methodology,
 *  Flowchart, detail slides, Datasets, Results, Tables x3, Graphs x2,
 *  Conclusion, References, Thank You).
 * Numbers: ds/models/paper/paper_numbers.json + committed CSV artifacts.
 * References: paper/main.md "## References" (same [1]-[13] as the PDF).
 * Run: node ppt/make_ieee_deck.js  ->  ppt/ieee_presentation.pptx
 */
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT = path.join(__dirname, 'ieee_presentation.pptx');

/* ---------- numbers (fail loud) ---------- */
const RAW = JSON.parse(fs.readFileSync(path.join(ROOT, 'ds', 'models', 'paper', 'paper_numbers.json'), 'utf8'));
const flat = (o, p = '') => Object.entries(o).reduce((acc, [k, val]) => {
  const kk = p ? p + '.' + k : k;
  if (val && typeof val === 'object' && !Array.isArray(val)) Object.assign(acc, flat(val, kk));
  else acc[kk] = val;
  return acc;
}, {});
const F = flat(RAW);
const v = (k) => { if (!(k in F)) throw new Error('missing number key: ' + k); return F[k]; };
const fx = (k, d = 3) => Number(v(k)).toFixed(d);
const fxa = (k, i, d = 3) => { const a = v(k); if (!Array.isArray(a) || i >= a.length) throw new Error('bad array key: ' + k + '[' + i + ']'); return Number(a[i]).toFixed(d); };
const pc = (k, d = 1) => (v(k) * 100).toFixed(d) + '%';

/* ---------- committed CSV artifacts ---------- */
const csv = (rel) => {
  const txt = fs.readFileSync(path.join(ROOT, rel), 'utf8').trim();
  const lines = txt.split(/\r?\n/);
  const parse = (line) => {
    const out = []; let cur = ''; let q = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') { q = !q; continue; }
      if (c === ',' && !q) { out.push(cur); cur = ''; } else cur += c;
    }
    out.push(cur); return out;
  };
  const head = parse(lines[0]);
  return lines.slice(1).map((l) => {
    const cells = parse(l); const row = {};
    head.forEach((h, i) => { row[h] = cells[i]; });
    return row;
  });
};

const metric = csv('ds/models/metric_table.csv');
const trade = csv('ds/models/threshold_tradeoff.csv');
const t4 = csv('ds/models/paper/paper_T4_conformal.csv');
const op = JSON.parse(fs.readFileSync(path.join(ROOT, 'ds', 'models', 'operating_point.json'), 'utf8'));

const mrow = (name) => { const r = metric.find((m) => m.Model === name); if (!r) throw new Error('metric row: ' + name); return r; };
const trow = (model, thr) => {
  const r = trade.find((t) => t.model === model && Math.abs(Number(t.threshold) - thr) < 1e-9);
  if (!r) throw new Error('tradeoff row: ' + model + ' @' + thr); return r;
};
const r3 = (x) => Number(x).toFixed(3);

const LOGREG = mrow('Logistic Regression (baseline)');
const WEIGHTED = mrow('XGBoost (class-weighted)');
const PRIMARY = mrow('XGBoost (unweighted + isotonic)');
const CONST = mrow('Constant (train prevalence)');
const DEF = trow('xgboost', 0.5);
const NX = op.nested_xgboost;

/* ---------- references (identical to the paper) ---------- */
const md = fs.readFileSync(path.join(ROOT, 'paper', 'main.md'), 'utf8');
const refs = md.split('## References')[1].split('\n').map((l) => l.trim()).filter((l) => /^\[\d+\]/.test(l));
if (refs.length !== 13) throw new Error('expected 13 references, got ' + refs.length);

const FIG = (n) => path.join(ROOT, 'ds', 'models', 'paper', n);
const LOGO = path.join(__dirname, 'assets', 'ieee_logo.png');

/* ---------- design: faithful to the template (4:3, serif, black on white) ---------- */
const SERIF = 'Times New Roman';
const INK = '000000', GRAY = '595959', BAND = 'F2F2F2', WHITE = 'FFFFFF', IEEE = '00629B';

const pres = new pptxgen();
pres.layout = 'LAYOUT_4x3'; /* 10 x 7.5 â€” same canvas as the template */
pres.author = 'Vaibhav Kadam, Tanvi Gandhi, Omkar Shinolikar';
pres.subject = 'Nexathon II paper presentation (IEEE format)';
pres.title = 'Real-Data Machine Learning for Telemedicine Triage and Outbreak Surveillance';

const title = (s, txt, size = 30) =>
  s.addText(txt, { x: 0.45, y: 0.28, w: 9.1, h: 0.9, fontFace: SERIF, fontSize: size, color: INK, align: 'center', valign: 'middle', margin: 0 });

const bullets = (s, items, opts = {}) => {
  const { x = 0.7, y = 1.3, w = 8.6, h = 5.8, size = 20, gap = 12 } = opts;
  const arr = items.map((it) => {
    const base = { breakLine: true, paraSpaceAfter: gap, fontSize: size, color: INK };
    if (typeof it === 'string') return { text: it, options: { ...base, bullet: true } };
    return {
      text: it.t,
      options: {
        ...base, bullet: it.bullet === undefined ? true : it.bullet,
        bold: !!it.bold, italic: !!it.italic,
        fontSize: it.size || size,
      },
    };
  });
  s.addText(arr, { x, y, w, h, fontFace: SERIF, valign: 'top', margin: 0.03 });
};

const caption = (s, txt) =>
  s.addText(txt, { x: 0.5, y: 6.85, w: 9.0, h: 0.45, fontFace: SERIF, fontSize: 12, italic: true, color: GRAY, align: 'center', valign: 'top', margin: 0 });

const tblStyle = {
  fontFace: SERIF, fontSize: 13.5, color: INK, valign: 'middle',
  border: { type: 'solid', color: '000000', pt: 0.75 },
  autoPage: false,
};

/* ================================================================ 1 title */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addImage({ path: LOGO, x: 0.5, y: 0.42, w: 1.75, h: 0.306 });
  s.addText('Nexathon II \u00B7 Paper Presentation \u00B7 AIKTC',
    { x: 5.5, y: 0.42, w: 4.0, h: 0.4, fontFace: SERIF, fontSize: 14, bold: true, color: GRAY, align: 'right', valign: 'middle', margin: 0 });
  s.addText('Real-Data Machine Learning for Telemedicine Triage and Outbreak Surveillance',
    { x: 0.6, y: 1.75, w: 8.8, h: 2.1, fontFace: SERIF, fontSize: 30, bold: true, italic: true, color: INK, align: 'center', valign: 'middle', margin: 0 });
  s.addText('Vaibhav Kadam  \u00B7  Tanvi Gandhi  \u00B7  Omkar Shinolikar',
    { x: 0.6, y: 4.15, w: 8.8, h: 0.45, fontFace: SERIF, fontSize: 19, color: INK, align: 'center', valign: 'middle', margin: 0 });
  s.addText('Bylance Technologies, India',
    { x: 0.6, y: 4.68, w: 8.8, h: 0.4, fontFace: SERIF, fontSize: 16, color: GRAY, align: 'center', valign: 'middle', margin: 0 });
  s.addText('9 October 2026  \u00B7  IEEE conference format',
    { x: 0.6, y: 5.35, w: 8.8, h: 0.4, fontFace: SERIF, fontSize: 15, bold: true, color: INK, align: 'center', valign: 'middle', margin: 0 });
  s.addText('Real public data only \u00B7 no synthetic records \u00B7 seed-42 byte-identical reruns \u00B7 fail-closed pipeline',
    { x: 0.6, y: 5.85, w: 8.8, h: 0.4, fontFace: SERIF, fontSize: 13, italic: true, color: GRAY, align: 'center', valign: 'middle', margin: 0 });
  s.addNotes('Title slide. Three authors, Bylance Technologies. The paper and deck are generated from committed artifacts \u2014 nothing is hand-typed.');
}

/* ================================================================ 2 outline */
{
  const s = pres.addSlide();
  title(s, 'Presentation Outline');
  bullets(s, ['Introduction', 'Problem Statement', 'Methodology', 'Results', 'Conclusion', 'Future Work', 'References'],
    { x: 1.4, y: 1.4, w: 7.2, h: 5.6, size: 24, gap: 16 });
  s.addNotes('Standard conference outline, same as the template.');
}

/* ================================================================ 3 abstract */
{
  const s = pres.addSlide();
  title(s, 'Abstract');
  bullets(s, [
    'Telemedicine platforms face two simultaneous decisions: which visit needs urgent escalation, and is a district-level outbreak beginning.',
    'Real public data only \u2014 CDC NHAMCS 2019 ED (n = ' + Number(v('module1_triage.n_analysis')).toLocaleString('en-US') + '), COVID-19 India district-weeks (' + Number(v('module2_outbreak.district_weeks')).toLocaleString('en-US') + '), Kaggle pharma sales; fail-closed if raw data is missing.',
    'Primary model fixed by a pre-declared rule (min OOF Brier): unweighted XGBoost + isotonic \u2014 AUC ' + fx('module1_triage.auc_xgb_cv') + ' [' + fxa('module1_triage.auc_xgb_ci95', 0) + ', ' + fxa('module1_triage.auc_xgb_ci95', 1) + '], Brier ' + fx('module1_triage.brier_primary_oof') + ' vs ' + fx('module1_triage.brier_constant_oof') + ' constant.',
    'Nested cross-fitted threshold: recall ' + fx('module1_triage.nested_recall') + ', under-triage ' + fx('module1_triage.nested_under_triage') + ', FPR ' + fx('module1_triage.nested_fpr') + '.',
    'PAC split-conformal rule (\u03B4 = 0.10): mean recall ' + fx('module1_triage.pac_mean_recall') + ' with ' + pc('module1_triage.pac_frac_below_085') + ' of splits below target (marginal: ' + fx('module1_triage.conformal_mean_recall') + ' / ' + pc('module1_triage.conformal_frac_below_085') + ').',
    'Outbreak module: ' + fx('module2_outbreak.delta_enrichment', 1) + '\u00D7 descriptive enrichment at a 5% alert budget \u2014 reported as enrichment, never as detection rates.',
  ], { size: 17.5, gap: 10 });
  s.addNotes('Condensed abstract with the locked headline numbers. Every figure here is also in the 6-page paper.');
}

/* ================================================================ 4 introduction */
{
  const s = pres.addSlide();
  title(s, 'Introduction');
  bullets(s, [
    'Two everyday decisions on a telemedicine platform, answered only with public data.',
    'Decision 1 \u2014 patient level: route an encounter to urgent care or routine advice. Errors are asymmetric: under-triage can be fatal.',
    'Decision 2 \u2014 population level: detect district-week outbreak signals under a fixed alert budget.',
    'Supplementary: pharmaceutical demand forecasting as a robustness check (naive baselines win 4/8 drugs \u2014 reported).',
    'Safety-first framing: sensitivity is prioritized, and every claim is audit-locked to a committed artifact.',
  ], { size: 19, gap: 13 });
  s.addNotes('Frame the two decisions and the honest-data philosophy before any method detail.');
}

/* ================================================================ 5 problem statement */
{
  const s = pres.addSlide();
  title(s, 'Problem Statement');
  bullets(s, [
    'Default threshold t = 0.50 is clinically unsafe: recall ' + fx('module1_triage.xgb_cv_t05_recall') + ' \u2192 under-triage ' + pc('module1_triage.xgb_cv_t05_under_triage') + ' (FPR ' + fx('module1_triage.xgb_cv_t05_fpr') + ').',
    'Thresholds tuned and evaluated on the same data overstate performance \u2014 nested cross-fitting is required for an honest estimate.',
    'The textbook conformal guarantee is only marginal: ' + pc('module1_triage.conformal_frac_below_085') + ' of splits miss the 0.85 target.',
    'Outbreak data has no gold-standard labels \u2014 precision, recall and ROC cannot be honestly computed.',
    'Objective: conservative operating points + calibrated claims, all from real public data.',
  ], { size: 18.5, gap: 13 });
  s.addNotes('Three honest problems: unsafe default, evaluation leakage, missing labels. Everything else in the deck is a response to these.');
}

/* ================================================================ 6 literature survey 1 */
{
  const s = pres.addSlide();
  title(s, 'Literature Survey 1');
  bullets(s, [
    { t: 'Clinical severity scores (NEWS2):', bold: true, bullet: false },
    'Vitals-only bedside score \u2014 cutoff \u22655: recall ' + fx('module1_triage.news2_cutoff5_recall') + ', precision ' + fx('module1_triage.news2_cutoff5_precision') + '; best cutoff 1: ' + fx('module1_triage.news2_cutoff1_recall') + ' / ' + fx('module1_triage.news2_cutoff1_precision') + '.',
    'No cutoff \u22651 reaches 0.85 recall on calibration \u2014 matched comparison infeasible (documented).',
    { t: 'Learned triage models:', bold: true, bullet: false },
    'Models over reason-for-visit + vitals outperform vitals-only scoring at comparable precision.',
    'NHAMCS is the standard public ED benchmark for triage research [1].',
    'Limitation: US data \u2014 non-transferability to deployment settings is stated explicitly.',
  ], { size: 17.5, gap: 9 });
  s.addNotes('Anchor against the clinical baseline (NEWS2) â€” the paper reports we cannot match 0.85 recall with any NEWS2 cutoff.');
}

/* ================================================================ 7 literature survey 2 */
{
  const s = pres.addSlide();
  title(s, 'Literature Survey 2');
  bullets(s, [
    { t: 'Conformal prediction:', bold: true, bullet: false },
    'Split-conformal prediction gives distribution-free marginal coverage [11], [12].',
    'Marginal \u2260 per-split: the guarantee holds in expectation over calibration draws.',
    { t: 'Risk control:', bold: true, bullet: false },
    'PAC (probably-approximately-correct) guarantees bound the share of bad splits via \u03B2 order statistics [11].',
    { t: 'Our contribution:', bold: true, bullet: false },
    'Apply PAC rank selection to triage recall control on a calibrated primary model \u2014 marginal and PAC rules both reported over 200 seeds.',
  ], { size: 18, gap: 10 });
  s.addNotes('Position the PAC rule as the honest upgrade of textbook conformal recall control.');
}

/* ================================================================ 8 literature survey 3 */
{
  const s = pres.addSlide();
  title(s, 'Literature Survey 3');
  bullets(s, [
    { t: 'Outbreak surveillance without labels:', bold: true, bullet: false },
    'Open volunteer district archives (e.g. OpenDengue [13]) are rich in time series, poor in ground-truth labels.',
    'Standard practice: alert-budget concentration in verified wave windows (EARS-style) instead of ROC curves.',
    { t: 'Forecasting baselines:', bold: true, bullet: false },
    'Seasonal naive and Holt-Winters are strong competitors on low-variance series \u2014 the free-lunch finding is reported, not hidden [7].',
    'Limitation: descriptive enrichment only; retrospective on a static archive.',
  ], { size: 18, gap: 11 });
  s.addNotes('Justify why the outbreak module reports enrichment instead of precision/recall.');
}

/* ================================================================ 9 proposed methodology */
{
  const s = pres.addSlide();
  title(s, 'Proposed Methodology');
  bullets(s, [
    'Six triage pipelines: LogReg, class-weighted XGB, unweighted XGB, + isotonic, + Platt, constant baseline.',
    'Primary chosen by pre-declared rule: minimum 5-fold OOF Brier among calibrated candidates (ties \u2192 isotonic).',
    'Threshold rule: argmax precision subject to OOF recall \u2265 0.80 (0.01 grid; ties \u2192 higher threshold).',
    'Nested cross-fitting: each outer fold re-selects its threshold on the other four folds only.',
    'Abstention band: calibrated p \u2208 (0.4, 0.6) \u2192 human review (' + pc('module1_triage.nested_abstention') + ' of cases).',
    { t: 'Workflow: real data \u2192 calibrated model \u2192 cross-fitted threshold \u2192 PAC recall control \u2192 triage advice', italic: true, bullet: false },
  ], { size: 17.5, gap: 9 });
  s.addNotes('The methodology in one slide: model selection, thresholding, abstention, conformal layer.');
}

/* ================================================================ 10 flowchart (pipeline figure) */
{
  const s = pres.addSlide();
  title(s, 'Flowchart of Proposed Method');
  s.addImage({ path: FIG('paper_fig0_pipeline.png'), x: 0.5, y: 1.5, w: 9.0, h: 9.0 * (848 / 2114) });
  caption(s, 'Curago data-science layer: patient-level triage + population-level surveillance (Fig. 1 of the paper). Data: NHAMCS 2019 ED, inCOVID19 district archive, Kaggle pharma sales.');
  s.addNotes('Walk left to right: intake, Module 1 triage with abstention, Module 2 surveillance, forecasting.');
}

/* ================================================================ 11 primary model & calibration */
{
  const s = pres.addSlide();
  title(s, 'Primary Model & Calibration');
  bullets(s, [
    'Primary: unweighted XGBoost + isotonic calibration \u2014 AUC ' + fx('module1_triage.auc_xgb_cv') + ' [' + fxa('module1_triage.auc_xgb_ci95', 0) + ', ' + fxa('module1_triage.auc_xgb_ci95', 1) + '].',
    'Logistic regression: AUC ' + fx('module1_triage.auc_logreg_cv') + ' [' + fxa('module1_triage.auc_logreg_ci95', 0) + ', ' + fxa('module1_triage.auc_logreg_ci95', 1) + ']; paired difference CI [' + fxa('module1_triage.auc_diff_ci95', 0) + ', ' + fxa('module1_triage.auc_diff_ci95', 1) + '] excludes zero.',
    'Class weighting distorts probabilities: Brier ' + fx('module1_triage.brier_weighted_oof') + ' vs ' + fx('module1_triage.brier_primary_oof') + ' (isotonic).',
    'Calibrated model beats the constant baseline: ' + fx('module1_triage.brier_primary_oof') + ' < ' + fx('module1_triage.brier_constant_oof') + '.',
    'Probabilities drive a threshold rule \u2014 they are not quoted as absolute risks.',
  ], { size: 18, gap: 12 });
  s.addNotes('The calibration story: why isotonic was pre-selected and what the probabilities are (and are not) used for.');
}

/* ================================================================ 12 threshold & nested cross-fitting */
{
  const s = pres.addSlide();
  title(s, 'Threshold Selection & Cross-Fitting', 28);
  bullets(s, [
    'Rule: argmax precision subject to OOF recall \u2265 0.80 (0.01 grid; ties \u2192 higher threshold).',
    'Pooled: t = ' + fx('module1_triage.tuned_xgb_threshold', 2) + ' \u2192 recall ' + fx('module1_triage.tuned_xgb_recall') + ', FPR ' + fx('module1_triage.tuned_xgb_fpr') + ', under-triage ' + fx('module1_triage.tuned_xgb_under_triage') + '.',
    'Nested (headline): outer 5-fold \u2014 threshold re-tuned on the other four folds only, then evaluated on its own test fold.',
    'Nested result: t = ' + fx('module1_triage.nested_threshold_min', 2) + '\u2013' + fx('module1_triage.nested_threshold_max', 2) + ' \u2192 recall ' + fx('module1_triage.nested_recall') + ', FPR ' + fx('module1_triage.nested_fpr') + ', under-triage ' + fx('module1_triage.nested_under_triage') + ', abstention ' + fx('module1_triage.nested_abstention') + '.',
    'No test-fold peeking \u2014 the conservative operating point we report.',
  ], { size: 18.5, gap: 13 });
  s.addNotes('Emphasise the nested protocol: this is what makes the 0.829 recall number honest.');
}

/* ================================================================ 13 PAC conformal */
{
  const s = pres.addSlide();
  title(s, 'PAC Split-Conformal Guarantee', 28);
  bullets(s, [
    'Protocol: 200 seeds, 60/20/20 stratified splits, \u03B1 = ' + fx('module1_triage.conformal_alpha', 2) + ' (target recall 0.85).',
    'Marginal rule (rank k = \u230A\u03B1(n+1)\u230B): mean ' + fx('module1_triage.conformal_mean_recall') + ' (SD ' + fx('module1_triage.conformal_sd_recall') + '), ' + pc('module1_triage.conformal_frac_below_085') + ' of splits below target.',
    'PAC rule: largest k with BetaCDF(1\u2212\u03B1; n+1\u2212k, k) \u2264 \u03B4, \u03B4 = ' + fx('module1_triage.pac_delta', 2) + '.',
    'PAC result: mean ' + fx('module1_triage.pac_mean_recall') + ' (SD ' + fx('module1_triage.pac_sd_recall') + '), ' + pc('module1_triage.pac_frac_below_085') + ' below target (\u2264 \u03B4), precision ' + fx('module1_triage.pac_mean_precision') + '.',
    'PAC bounds the share of below-target splits \u2014 still not a per-split guarantee, and we say so.',
  ], { size: 18, gap: 12 });
  s.addNotes('Core methodological contribution: PAC rank selection. Lower rank â†’ lower threshold â†’ higher recall, at a small precision cost.');
}

/* ================================================================ 14 datasets */
{
  const s = pres.addSlide();
  title(s, 'Datasets Used');
  bullets(s, [
    'CDC NHAMCS 2019 ED \u2014 ' + Number(v('module1_triage.n_analysis')).toLocaleString('en-US') + ' encounters after screening (US public domain).',
    'COVID-19 India district time-series \u2014 ' + Number(v('module2_outbreak.district_weeks')).toLocaleString('en-US') + ' district-weeks, ' + Number(v('module2_outbreak.districts')) + ' districts (open archive).',
    'Kaggle pharma sales \u2014 ' + v('supplement_forecast.months_analyzed') + ' monthly points, 8 ATC groups (CC BY-NC 4.0).',
    'Evaluation: 5-fold OOF CV, nested cross-fitting, 200-seed conformal study, 12-step forecast holdout.',
    'Every CSV carries a Data Source column; licenses and citations in docs/citations.md.',
  ], { size: 18.5, gap: 13 });
  s.addNotes('All three datasets are public; raw data lives outside git and the pipeline fails closed without it.');
}

/* ================================================================ 15 results & discussion */
{
  const s = pres.addSlide();
  title(s, 'Results & Discussion');
  bullets(s, [
    'Triage: under-triage ' + pc('module1_triage.xgb_cv_t05_under_triage') + ' \u2192 ' + pc('module1_triage.nested_under_triage') + ' via cross-fitted thresholds (recall ' + fx('module1_triage.nested_recall') + ', FPR ' + fx('module1_triage.nested_fpr') + ').',
    'Calibration: Brier ' + fx('module1_triage.brier_primary_oof') + ' beats constant ' + fx('module1_triage.brier_constant_oof') + '; probabilities are threshold inputs, not quoted risks.',
    'Guarantees: marginal ' + fx('module1_triage.conformal_mean_recall') + ' / ' + pc('module1_triage.conformal_frac_below_085') + ' below target; PAC ' + fx('module1_triage.pac_mean_recall') + ' / ' + pc('module1_triage.pac_frac_below_085') + ' (\u03B4 = 10%).',
    'Seed-42 split: recall ' + fx('module1_triage.seed42_recall') + ' [' + fxa('module1_triage.seed42_recall_ci95', 0) + ', ' + fxa('module1_triage.seed42_recall_ci95', 1) + '], precision ' + fx('module1_triage.seed42_precision') + ', FPR ' + fx('module1_triage.seed42_fpr') + '.',
    'Outbreak: ' + fx('module2_outbreak.delta_enrichment', 1) + '\u00D7 enrichment (' + fx('module2_outbreak.iso_single_delta_flag_rate') + ' vs ' + fx('module2_outbreak.delta_outside_flag_rate') + '), ' + pc('module2_outbreak.iso_single_delta_coverage') + ' district coverage \u2014 descriptive.',
  ], { size: 17.5, gap: 11 });
  s.addNotes('Results with their honest qualifiers attached. Numbers follow directly from the tables on the next slides.');
}

/* ================================================================ 16 table 1 */
{
  const s = pres.addSlide();
  title(s, 'Table 1. Triage model comparison (5-fold OOF, t = 0.50)', 24);
  const hdr = { options: { bold: true, fill: { color: BAND }, align: 'center' } };
  const cell = (t, o = {}) => ({ text: t, options: o });
  const rows = [
    ['Model', 'AUC', 'Recall', 'Precision', 'Brier'].map((t) => cell(t, hdr.options)),
    ['Logistic regression', r3(LOGREG['AUC-ROC']), r3(LOGREG['Recall (Sensitivity)']), r3(LOGREG['Precision']), r3(LOGREG['Brier (OOF)'])],
    ['XGBoost (class-weighted)', r3(WEIGHTED['AUC-ROC']), r3(WEIGHTED['Recall (Sensitivity)']), r3(WEIGHTED['Precision']), r3(WEIGHTED['Brier (OOF)'])],
    ['XGBoost + isotonic (primary)', r3(PRIMARY['AUC-ROC']), r3(PRIMARY['Recall (Sensitivity)']), r3(PRIMARY['Precision']), r3(PRIMARY['Brier (OOF)'])].map((t, i) => cell(t, { bold: true })),
    ['Constant (train prevalence)', r3(CONST['AUC-ROC']), r3(CONST['Recall (Sensitivity)']), r3(CONST['Precision']), r3(CONST['Brier (OOF)'])],
  ];
  s.addTable(rows, { ...tblStyle, x: 0.55, y: 1.5, w: 8.9, colW: [3.1, 1.35, 1.45, 1.5, 1.5], rowH: 0.62,
    align: 'center', valign: 'middle', margin: 0.06 });
  caption(s, 'Primary chosen by min OOF Brier among calibrated candidates. Data: CDC NHAMCS 2019 ED (n = 13,595).');
  s.addNotes('Calibration is the differentiator: the primary model has the best Brier while staying at the top of the AUC table.');
}

/* ================================================================ 17 table 2 */
{
  const s = pres.addSlide();
  title(s, 'Table 2. Operating points', 26);
  const hdr = { bold: true, fill: { color: BAND }, align: 'center' };
  const cell = (t, o = {}) => ({ text: t, options: o });
  const rows = [
    ['Setting', 't', 'Recall', 'Precision', 'FPR', 'Under-triage'].map((t) => cell(t, hdr)),
    ['Default, primary', '0.50', r3(DEF.recall), r3(DEF.precision), r3(DEF.fpr), r3(DEF.under_triage)],
    ['Tuned, pooled OOF', fx('module1_triage.tuned_xgb_threshold', 2), fx('module1_triage.tuned_xgb_recall'), fx('module1_triage.tuned_xgb_precision'), fx('module1_triage.tuned_xgb_fpr'), fx('module1_triage.tuned_xgb_under_triage')],
    ['Nested cross-fitted (headline)', fx('module1_triage.nested_threshold_min', 2) + '\u2013' + fx('module1_triage.nested_threshold_max', 2), fx('module1_triage.nested_recall'), fx('module1_triage.nested_precision'), fx('module1_triage.nested_fpr'), fx('module1_triage.nested_under_triage')],
    ['Logistic regression, tuned', r3(op.logreg.threshold), r3(op.logreg.expected_recall), r3(op.logreg.expected_precision), r3(op.logreg.expected_fpr), r3(op.logreg.expected_under_triage)],
    ['Conformal, seed-42 split', fx('module1_triage.seed42_threshold'), fx('module1_triage.seed42_recall'), fx('module1_triage.seed42_precision'), fx('module1_triage.seed42_fpr'), r3(1 - v('module1_triage.seed42_recall'))],
  ];
  rows[3] = rows[3].map((c) => cell(typeof c === 'string' ? c : c.text, { bold: true }));
  s.addTable(rows, { ...tblStyle, x: 0.4, y: 1.5, w: 9.2, colW: [3.0, 1.0, 1.2, 1.35, 1.15, 1.5], rowH: 0.58,
    align: 'center', valign: 'middle', margin: 0.05, fontSize: 13 });
  caption(s, 'The nested row is the conservative headline: thresholds selected without test-fold access. Data: CDC NHAMCS 2019 ED.');
  s.addNotes('Read the default row first (unsafe), then the nested row (the number we stand behind).');
}

/* ================================================================ 18 table 3 */
{
  const s = pres.addSlide();
  title(s, 'Table 3. Conformal \u03B1 sweep \u2014 marginal vs PAC (200 splits)', 24);
  const hdr = { bold: true, fill: { color: BAND }, align: 'center' };
  const cell = (t, o = {}) => ({ text: t, options: o });
  const rows = [
    ['\u03B1', 'Rule', 'Mean recall', 'SD', 'Below target', 'Mean precision'].map((t) => cell(t, hdr)),
    ...t4.map((r) => {
      const rule = r['Rule'].startsWith('PAC') ? 'PAC (\u03B4 = 0.10)' : 'marginal';
      const row = [Number(r['alpha (miss target)']).toFixed(2), rule, Number(r['Mean recall']).toFixed(3),
        Number(r['SD']).toFixed(3), (Number(r['Frac. below $1-\\alpha$']) * 100).toFixed(1) + '%', Number(r['Mean precision']).toFixed(3)];
      const isPac = rule.startsWith('PAC');
      const isA15 = Math.abs(Number(r['alpha (miss target)']) - 0.15) < 1e-9;
      return row.map((t) => cell(t, isA15 ? { bold: true } : {}));
    }),
  ];
  s.addTable(rows, { ...tblStyle, x: 0.5, y: 1.35, w: 9.0, colW: [0.8, 2.1, 1.7, 1.2, 1.6, 1.6], rowH: 0.42,
    align: 'center', valign: 'middle', margin: 0.04, fontSize: 12.5 });
  caption(s, '\u03B1 = 0.15 rows (bold) are the paper\u2019s headline: marginal 12.5% below target vs PAC 3.5% \u2264 \u03B4. Data: CDC NHAMCS 2019 ED.');
  s.addNotes('The PAC rule trades a little precision for a bounded below-target share at every alpha.');
}

/* ================================================================ 19 ROC graph */
{
  const s = pres.addSlide();
  title(s, 'ROC Curve', 30);
  s.addImage({ path: FIG('paper_fig1_roc.png'), x: 2.4, y: 1.3, w: 5.2, h: 5.2 * (939 / 1029) });
  caption(s, '5-fold OOF ROC: primary AUC ' + fx('module1_triage.auc_xgb_cv') + ' [' + fxa('module1_triage.auc_xgb_ci95', 0) + ', ' + fxa('module1_triage.auc_xgb_ci95', 1) + '] vs LogReg ' + fx('module1_triage.auc_logreg_cv') + ' (Fig. 2 of the paper).');
  s.addNotes('The paired difference CI [0.012, 0.031] excludes zero â€” the AUC gain is real but modest; calibration is the bigger win.');
}

/* ================================================================ 20 conformal graph */
{
  const s = pres.addSlide();
  title(s, 'Conformal Recall Distribution', 28);
  s.addImage({ path: FIG('paper_fig4_conformal_hist.png'), x: 2.3, y: 1.3, w: 5.4, h: 5.4 * (879 / 1028) });
  caption(s, 'Test recall over 200 splits at \u03B1 = 0.15: marginal vs PAC rule (Fig. 5 of the paper).');
  s.addNotes('Visual of the honest core: the marginal rule scatters below 0.85; the PAC rule shifts mass above the target.');
}

/* ================================================================ 21 conclusion */
{
  const s = pres.addSlide();
  title(s, 'Conclusion');
  bullets(s, [
    'Real public data, no synthetic records \u2014 full pipeline fail-closed and seed-42 reproducible.',
    'Calibrated primary + nested cross-fitting: recall ' + fx('module1_triage.nested_recall') + ' at under-triage ' + fx('module1_triage.nested_under_triage') + ' (FPR ' + fx('module1_triage.nested_fpr') + ').',
    'PAC conformal control: below-target share ' + pc('module1_triage.pac_frac_below_085') + ' \u2264 \u03B4 = 10% (marginal ' + pc('module1_triage.conformal_frac_below_085') + ').',
    'Outbreak module: descriptive ' + fx('module2_outbreak.delta_enrichment', 1) + '\u00D7 enrichment only \u2014 stated as such.',
    { t: 'Future Work:', bold: true, bullet: false },
    'External validation under covariate shift; outcome-linked labels (T4); surveillance baselines (T5); prospective live evaluation.',
  ], { size: 18, gap: 11 });
  s.addNotes('Close on honesty: what we proved, what we deliberately did not claim, and what comes next.');
}

/* ================================================================ 22 references */
{
  const s = pres.addSlide();
  title(s, 'References');
  const opt = (r) => ({ text: r, options: { breakLine: true, paraSpaceAfter: 7, fontSize: 10.5, color: INK } });
  s.addText(refs.slice(0, 7).map(opt), { x: 0.5, y: 1.25, w: 4.45, h: 5.9, fontFace: SERIF, margin: 0, valign: 'top' });
  s.addText(refs.slice(7).map(opt), { x: 5.1, y: 1.25, w: 4.45, h: 5.9, fontFace: SERIF, margin: 0, valign: 'top' });
  s.addNotes('Same 13-item bibliography as the compiled paper \u2014 citation numbering is identical.');
}

/* ================================================================ 23 thank you */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText('Thank You', { x: 0.6, y: 2.6, w: 8.8, h: 1.3, fontFace: SERIF, fontSize: 54, bold: true, italic: true, color: INK, align: 'center', valign: 'middle', margin: 0 });
  s.addText('Questions?', { x: 0.6, y: 4.0, w: 8.8, h: 0.7, fontFace: SERIF, fontSize: 26, color: GRAY, align: 'center', valign: 'middle', margin: 0 });
  s.addText('Paper: Real-Data Machine Learning for Telemedicine Triage and Outbreak Surveillance \u00B7 github.com/vaibhav7087/curago_paper_version',
    { x: 0.6, y: 5.3, w: 8.8, h: 0.5, fontFace: SERIF, fontSize: 13, italic: true, color: GRAY, align: 'center', valign: 'middle', margin: 0 });
  s.addNotes('Open for questions. Repo link points to the committed artifacts behind every number.');
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log('wrote', OUT, '| slides: 23 | refs:', refs.length, '| numbers: paper_numbers.json + committed CSVs');
});
