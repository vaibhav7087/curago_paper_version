"""
ds/triage_models.py — Single source of truth for triage model definitions.

Every consumer (train_triage, tune_threshold, severity_eval_extras,
conformal_triage, paper_figures, predict) builds estimators from here so the
paper, the figures and the deployed artifact cannot drift apart.

Candidates (same features from train_triage.load_real, same StratifiedKFold(5,
shuffle, seed 42) protocol):
  logreg           class-weighted logistic regression + scaling (baseline)
  xgb_weighted     XGBoost, class-weighted (predecessor; kept for comparison)
  xgb_unw          XGBoost, unweighted (raw tree scores)
  xgb_unw_isotonic unweighted + isotonic calibration
  xgb_unw_platt    unweighted + Platt (sigmoid) calibration

Calibration protocol (sklearn CalibratedClassifierCV, ensemble=False):
  inner StratifiedKFold(5, shuffle, seed 42) out-of-fold raw scores fit the
  calibrator; the base booster is then fit on the full training partition.
  Probabilities entering thresholds, conformal quantiles and abstention are
  therefore calibrated scores, not raw margin transforms.

PRE-DECLARED primary selection rule (evaluated + asserted by train_triage.py):
  among {xgb_unw_isotonic, xgb_unw_platt} the primary model is the one with
  the lower 5-fold out-of-fold Brier score; ties go to isotonic.
  PRIMARY below records the locked outcome of that rule.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CALIB_CV_SEED = 42
CALIB_CV_FOLDS = 5

CANDIDATES = ["logreg", "xgb_weighted", "xgb_unw",
              "xgb_unw_isotonic", "xgb_unw_platt"]
CALIBRATED = {"xgb_unw_isotonic", "xgb_unw_platt"}
LABELS = {
    "logreg": "Logistic Regression (baseline)",
    "xgb_weighted": "XGBoost (class-weighted)",
    "xgb_unw": "XGBoost (unweighted)",
    "xgb_unw_isotonic": "XGBoost (unweighted + isotonic)",
    "xgb_unw_platt": "XGBoost (unweighted + Platt)",
    "constant": "Constant (train prevalence)",
}
# Locked outcome of the pre-declared Brier rule; train_triage.py re-checks it.
PRIMARY = "xgb_unw_isotonic"
SELECTION_RULE = ("min 5-fold OOF Brier among {isotonic, Platt}; ties -> isotonic")

BASE_XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="logloss",
)


def base_xgb(scale_pos_weight=None):
    """Imputer + unweighted (or weighted) XGBoost — the base learner."""
    params = dict(BASE_XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(**params)),
    ])


def logreg_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=42)),
    ])


def _calibrated(method, scale_pos_weight=None):
    cv = StratifiedKFold(CALIB_CV_FOLDS, shuffle=True, random_state=CALIB_CV_SEED)
    return CalibratedClassifierCV(
        estimator=base_xgb(scale_pos_weight), cv=cv, method=method,
        ensemble=False,
    )


def make_estimator(name, scale_pos_weight=None):
    """Fresh estimator for `name` (clone-safe: nothing is shared)."""
    if name == "logreg":
        return logreg_pipe()
    if name == "xgb_weighted":
        return base_xgb(scale_pos_weight)
    if name == "xgb_unw":
        return base_xgb(None)
    if name == "xgb_unw_isotonic":
        return _calibrated("isotonic")
    if name == "xgb_unw_platt":
        return _calibrated("sigmoid")
    raise KeyError(f"unknown model: {name}")


def make_primary(scale_pos_weight=None):
    return make_estimator(PRIMARY, scale_pos_weight)


def extract_base(model):
    """Underlying imputer+booster Pipeline (fitted on the full train split)."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model
