"""Safely promote the verified candidate model to production with a backup."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_extraction import FEATURE_COUNT

CANDIDATE = ROOT / "models" / "gesture_model_new.pkl"
PRODUCTION = ROOT / "models" / "gesture_model.pkl"
BACKUPS = ROOT / "models" / "backups"


def main():
    if not CANDIDATE.exists():
        print(f"[ERROR] Candidate model not found: {CANDIDATE}")
        return 1
    try:
        candidate = joblib.load(CANDIDATE)
        if getattr(candidate, "n_features_in_", FEATURE_COUNT) != FEATURE_COUNT:
            raise ValueError("Candidate model has an incompatible feature count.")
    except Exception as error:
        print(f"[ERROR] Candidate model validation failed: {error}")
        return 1
    BACKUPS.mkdir(parents=True, exist_ok=True)
    if PRODUCTION.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUPS / f"gesture_model_{timestamp}.pkl"
        shutil.copy2(PRODUCTION, backup)
        print(f"[BACKUP] {backup}")
    shutil.copy2(CANDIDATE, PRODUCTION)
    print(f"[PROMOTED] {CANDIDATE.name} -> {PRODUCTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
