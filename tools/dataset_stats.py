"""Report data quality for the user-collected VisionControl dataset."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_extraction import FEATURE_COUNT

DATASET_PATH = ROOT / "data" / "gesture_dataset.csv"


def main():
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        return
    try:
        data = pd.read_csv(DATASET_PATH)
    except Exception as error:
        print(f"[ERROR] Could not read dataset: {error}")
        return
    if data.shape[1] != FEATURE_COUNT + 1 or "label" not in data:
        print(f"[ERROR] Expected {FEATURE_COUNT} features plus label; found {data.shape[1]} columns.")
        return
    counts = data["label"].value_counts().sort_index()
    print("\nVISIONCONTROL DATASET STATISTICS")
    print(f"Total samples: {len(data)}")
    print(f"Number of gestures: {len(counts)}")
    print("\nSamples per gesture:")
    print(counts.to_string())
    if counts.empty:
        return
    print(f"\nMinimum class size: {counts.min()}")
    print(f"Maximum class size: {counts.max()}")
    print(f"Average samples per class: {counts.mean():.1f}")
    recommended = max(30, int(counts.max() * 0.5))
    weak = counts[counts < recommended]
    if not weak.empty:
        print(f"\n[WARNING] Underrepresented classes (recommended at least {recommended} samples):")
        print(weak.to_string())
    if data.isna().any().any():
        print("\n[WARNING] Missing values were found; do not train until they are fixed.")


if __name__ == "__main__":
    main()
