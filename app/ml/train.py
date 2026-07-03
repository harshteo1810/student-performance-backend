"""
Model training script for the Student Performance Risk Classifier.

Run manually whenever the dataset changes:
    python3 -m app.ml.train

Produces (in app/ml/artifacts/):
    model.pkl           -- trained classifier
    label_encoder.pkl   -- encodes/decodes risk_category <-> gender
    feature_columns.pkl -- exact column order the model expects at inference time
    metrics.json         -- test-set metrics, for your report

Design notes:
  - RandomForestClassifier: no scaling needed, handles the mixed
    continuous/count features fine, gives feature_importances_ for free
    (useful for your report and for the "personalized suggestions" objective
    later -- you can point to which features drove a given prediction).
  - Stratified train/test split: risk_category is imbalanced (see class
    counts below), so a plain random split risks skewed test proportions.
  - student_id is dropped before training -- it's an identifier, not signal.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "student_performance_dataset.csv")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TARGET_COL = "risk_category"
DROP_COLS = ["student_id"]  # identifier, never a feature


def main():
    df = pd.read_csv(DATA_PATH)

    # Impute the one column with missingness (extracurricular_involvement_score)
    df["extracurricular_involvement_score"] = df["extracurricular_involvement_score"].fillna(
        df["extracurricular_involvement_score"].median()
    )

    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL]

    # Encode the one categorical feature (gender)
    gender_encoder = LabelEncoder()
    X["gender"] = gender_encoder.fit_transform(X["gender"])

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",  # dataset is imbalanced across the 3 classes
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=target_encoder.classes_, output_dict=True
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    print(f"Macro F1: {macro_f1:.4f}")

    # Feature importances -- useful for the "personalized suggestions" objective later
    importances = sorted(
        zip(feature_columns, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop feature importances:")
    for name, score in importances[:8]:
        print(f"  {name:35s} {score:.4f}")

    # Save artifacts
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(gender_encoder, os.path.join(ARTIFACTS_DIR, "gender_encoder.pkl"))
    joblib.dump(target_encoder, os.path.join(ARTIFACTS_DIR, "target_encoder.pkl"))
    joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"macro_f1": macro_f1, "classification_report": report}, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
