"""Train a candidate VisionControl model without touching production."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_COUNT


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "gesture_dataset.csv"
DEFAULT_MODEL = ROOT / "models" / "gesture_model_new.pkl"
MIN_RECOMMENDED_SAMPLES_PER_CLASS = 30


def load_dataset(path):
    if not path.exists():
        raise ValueError(f"Dataset not found: {path}. Run tools/collect_data.py first.")
    data = pd.read_csv(path)
    if "label" not in data or data.shape[1] != FEATURE_COUNT + 1:
        raise ValueError(f"Dataset must contain exactly {FEATURE_COUNT} feature columns and a label column.")
    feature_columns = [column for column in data.columns if column != "label"]
    if len(feature_columns) != FEATURE_COUNT or data[feature_columns].isna().any().any() or data["label"].isna().any():
        raise ValueError("Dataset has invalid feature columns or missing values.")
    try:
        features = data[feature_columns].to_numpy(dtype=np.float32)
    except ValueError as error:
        raise ValueError(f"All feature values must be numeric: {error}") from error
    if not np.isfinite(features).all():
        raise ValueError("Dataset contains non-finite feature values.")
    labels = data["label"].astype(str).str.strip().to_numpy()
    if not np.all(labels):
        raise ValueError("Dataset contains empty labels.")
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Train a non-production VisionControl gesture model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        features, labels = load_dataset(args.dataset)
    except ValueError as error:
        print(f"[VALIDATION ERROR] {error}")
        return 1

    counts = pd.Series(labels).value_counts().sort_index()
    print("\nVISIONCONTROL CANDIDATE MODEL TRAINING")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(labels)} | Features: {features.shape[1]} | Classes: {len(counts)}")
    print("\nClass distribution:")
    print(counts.to_string())
    if len(counts) < 2 or counts.min() < 2:
        print("\n[VALIDATION ERROR] At least two classes and two samples per class are required.")
        return 1
    weak = counts[counts < MIN_RECOMMENDED_SAMPLES_PER_CLASS]
    if not weak.empty:
        print(f"\n[WARNING] Classes below {MIN_RECOMMENDED_SAMPLES_PER_CLASS} samples are not production-ready:")
        print(weak.to_string())

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.20, random_state=42, stratify=labels
        )
    except ValueError as error:
        print(f"[VALIDATION ERROR] Train/test split failed: {error}")
        return 1
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=1, class_weight="balanced"
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nPrecision / Recall / F1:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"\n[SAVED] Candidate model: {args.output}")
    print("Production model was not changed. Review metrics, then run tools/promote_model.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
