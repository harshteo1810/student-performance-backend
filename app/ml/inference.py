import os

import joblib
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

_model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
_gender_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "gender_encoder.pkl"))
_target_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "target_encoder.pkl"))
_feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

# Class order in target_encoder.classes_ determines which probability column is which
_CLASS_INDEX = {cls: i for i, cls in enumerate(_target_encoder.classes_)}


def predict_risk(features: dict) -> dict:
    """
    features: dict matching StudentFeaturesIn fields (minus external_student_id).
    Returns: {"risk_category": str, "prob_high_risk": float,
              "prob_moderate_risk": float, "prob_good_performance": float}
    """
    row = {k: features[k] for k in features if k in _feature_columns}

    # Impute missing extracurricular score the same way training did: median.
    # (At inference time we don't have the training median handy, so this
    # uses a fixed fallback of 5.0 -- the scale midpoint -- for a single
    # missing value. Acceptable for a single-row inference call; if you
    # batch-predict many rows with missing values, impute with the
    # training-set median saved as an artifact instead.)
    if row.get("extracurricular_involvement_score") is None:
        row["extracurricular_involvement_score"] = 5.0

    row["gender"] = _gender_encoder.transform([row["gender"]])[0]

    X = pd.DataFrame([row])[_feature_columns]  # enforce exact column order

    probs = _model.predict_proba(X)[0]
    pred_idx = probs.argmax()
    risk_category = _target_encoder.inverse_transform([pred_idx])[0]

    return {
        "risk_category": risk_category,
        "prob_high_risk": float(probs[_CLASS_INDEX["High-Risk"]]),
        "prob_moderate_risk": float(probs[_CLASS_INDEX["Moderate-Risk"]]),
        "prob_good_performance": float(probs[_CLASS_INDEX["Good-Performance"]]),
    }
