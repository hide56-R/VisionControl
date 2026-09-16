"""Evaluate a saved candidate model against the collected dataset."""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_extraction import FEATURE_COUNT

DATASET_PATH = ROOT / "data" / "gesture_dataset.csv"
MODEL_PATH = ROOT / "models" / "gesture_model_new.pkl"


def main():
    if not DATASET_PATH.exists() or not MODEL_PATH.exists():
        print("[ERROR] Candidate dataset or models/gesture_model_new.pkl is missing.")
        return
    data = pd.read_csv(DATASET_PATH)
    if "label" not in data or data.shape[1] != FEATURE_COUNT + 1 or data.isna().any().any():
        print("[ERROR] Dataset is invalid or contains missing values.")
        return
    features = data.drop(columns="label").to_numpy()
    labels = data["label"].to_numpy()
    try:
        _, x_test, _, y_test = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)
        model = joblib.load(MODEL_PATH)
        if getattr(model, "n_features_in_", FEATURE_COUNT) != FEATURE_COUNT:
            raise ValueError("Model feature count is incompatible.")
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        predicted = model.predict(x_test)
    except Exception as error:
        print(f"[ERROR] Evaluation failed: {error}")
        return
    print(f"Accuracy: {accuracy_score(y_test, predicted) * 100:.2f}%")
    print(classification_report(y_test, predicted, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predicted, labels=model.classes_))


if __name__ == "__main__":
    main()
