"""Interactive, append-safe collector for a personal VisionControl dataset."""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_extraction import FEATURE_COUNT, features_from_hand_landmarks


# Collection configuration
TARGET_SAMPLES = 200
SAMPLE_INTERVAL_SECONDS = 0.12
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DATASET_PATH = ROOT / "data" / "gesture_dataset.csv"
LABELS = {
    ord("1"): "THUMBS_UP",
    ord("2"): "THUMBS_DOWN",
    ord("3"): "FIST",
    ord("4"): "PEACE",
    ord("5"): "POINT",
    ord("6"): "OPEN_PALM",
    ord("7"): "ROCK",
}
GESTURES = list(LABELS.values())
FEATURE_COLUMNS = [f"feature_{index:02d}" for index in range(FEATURE_COUNT)]
EXPECTED_HEADER = FEATURE_COLUMNS + ["label"]


def feature_key(features):
    """Stable key used to prevent an exact normalized sample being stored twice."""
    values = np.asarray(features, dtype=np.float32).reshape(FEATURE_COUNT)
    return tuple(np.round(values, 7))


def inspect_or_create_dataset():
    """Validate existing data before appending; never overwrite or delete rows."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    existing_features = set()

    if not DATASET_PATH.exists():
        with DATASET_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(EXPECTED_HEADER)
        return counts, existing_features

    with DATASET_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(
                "Existing dataset columns are incompatible. No data was changed. "
                f"Expected 63 feature columns plus label in {DATASET_PATH}."
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[column]) for column in FEATURE_COLUMNS]
                label = str(row["label"]).strip()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row {row_number}; no data was changed: {error}") from error
            if not label or not np.isfinite(values).all():
                raise ValueError(f"Invalid row {row_number}; no data was changed.")
            counts[label] += 1
            existing_features.add(feature_key(values))

    return counts, existing_features


def append_sample(features, label):
    values = np.asarray(features, dtype=np.float32).reshape(FEATURE_COUNT)
    with DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(values.tolist() + [label])


def draw_text(frame, text, point, color=(245, 245, 245), size=0.48, thickness=1):
    cv2.putText(frame, str(text), point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def draw_dashboard(frame, label, counts, collection_state, hand_detected, message):
    height, width = frame.shape[:2]
    collected = counts[label]
    progress = min(collected / TARGET_SAMPLES, 1.0)
    state_color = (80, 210, 130) if collection_state == "COLLECTING" else (0, 220, 255)
    hand_color = (80, 210, 130) if hand_detected else (80, 80, 230)

    cv2.rectangle(frame, (0, 0), (width, 190), (18, 18, 24), -1)
    draw_text(frame, "VISIONCONTROL", (20, 30), (80, 210, 130), 0.75, 2)
    draw_text(frame, "AI GESTURE TRAINING STUDIO", (20, 55), (160, 165, 175), 0.42)
    draw_text(frame, f"CURRENT GESTURE: {label}", (20, 85), size=0.55, thickness=2)
    draw_text(frame, f"TARGET: {TARGET_SAMPLES} samples", (20, 112))
    draw_text(frame, f"COLLECTED: {collected} / {TARGET_SAMPLES}", (20, 137))
    draw_text(frame, f"PROGRESS: {progress * 100:.1f}%", (20, 162))
    draw_text(frame, f"COLLECTION: {collection_state}", (350, 85), state_color, 0.5, 2)
    draw_text(frame, f"HAND: {'DETECTED' if hand_detected else 'NOT DETECTED'}", (350, 112), hand_color, 0.5)
    draw_text(frame, message, (20, 184), (180, 185, 195), 0.4)

    cv2.rectangle(frame, (0, height - 145), (width, height), (18, 18, 24), -1)
    draw_text(frame, "DATASET SUMMARY", (20, height - 118), (160, 165, 175), 0.48, 2)
    for index, gesture in enumerate(GESTURES):
        column = 20 if index < 4 else 330
        row = height - 91 + (index % 4) * 20
        draw_text(frame, f"{gesture:<14} {counts[gesture]}", (column, row), size=0.42)
    total = sum(counts.values())
    draw_text(frame, f"TOTAL {total}", (500, height - 35), (80, 210, 130), 0.48, 2)
    draw_text(frame, "S start/pause | R reset state | H help | Q quit", (20, height - 11), (160, 165, 175), 0.38)


def class_balance_warning(counts):
    selected_counts = [counts[label] for label in GESTURES if counts[label] > 0]
    if len(selected_counts) < 2:
        return "Collect every gesture for a balanced dataset."
    if min(selected_counts) < max(selected_counts) * 0.5:
        return "Class imbalance warning: collect more samples for lower-count gestures."
    return "Dataset class balance looks reasonable."


def print_help():
    print("\nTRAINING STUDIO CONTROLS")
    print("1 THUMBS_UP | 2 THUMBS_DOWN | 3 FIST | 4 PEACE")
    print("5 POINT | 6 OPEN_PALM | 7 ROCK")
    print("S start/pause | R reset current collection state | Q quit")
    print("R does not delete any saved dataset rows.\n")


def main():
    try:
        counts, existing_features = inspect_or_create_dataset()
    except (OSError, ValueError) as error:
        print(f"[DATASET ERROR] {error}")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[CAMERA ERROR] Camera could not be opened.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    selected_label = "POINT"
    collection_state = "READY"
    last_saved_at = 0.0
    message = "Select a gesture and press S to start."
    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils

    try:
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.65,
                            min_tracking_confidence=0.65) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("[CAMERA ERROR] Could not read a frame.")
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_detected, features = False, None
                if result.multi_hand_landmarks:
                    hand_detected = True
                    hand = result.multi_hand_landmarks[0]
                    drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    try:
                        features = features_from_hand_landmarks(hand)
                        if features.shape != (1, FEATURE_COUNT):
                            raise ValueError(f"Expected 63 features, got {features.shape}.")
                    except ValueError as error:
                        features = None
                        message = f"Invalid feature vector: {error}"

                now = time.monotonic()
                if collection_state == "COLLECTING":
                    if counts[selected_label] >= TARGET_SAMPLES:
                        collection_state = "TARGET REACHED"
                        message = "TARGET REACHED. Press another number key."
                    elif not hand_detected:
                        message = "Warning: no hand detected."
                    elif features is None:
                        message = "Warning: invalid feature vector."
                    elif now - last_saved_at >= SAMPLE_INTERVAL_SECONDS:
                        key = feature_key(features)
                        if key in existing_features:
                            message = "Warning: exact duplicate sample skipped."
                            last_saved_at = now
                        else:
                            append_sample(features, selected_label)
                            existing_features.add(key)
                            counts[selected_label] += 1
                            last_saved_at = now
                            message = "Sample saved. " + class_balance_warning(counts)
                            if counts[selected_label] >= TARGET_SAMPLES:
                                collection_state = "TARGET REACHED"
                                message = "TARGET REACHED. Press another number key."

                draw_dashboard(frame, selected_label, counts, collection_state, hand_detected, message)
                cv2.imshow("VisionControl - Training Studio", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in LABELS:
                    selected_label = LABELS[key]
                    collection_state = "READY" if counts[selected_label] < TARGET_SAMPLES else "TARGET REACHED"
                    message = "TARGET REACHED. Choose another gesture." if collection_state == "TARGET REACHED" else f"Selected {selected_label}; press S to collect."
                elif key == ord("s"):
                    if counts[selected_label] >= TARGET_SAMPLES:
                        collection_state = "TARGET REACHED"
                        message = "TARGET REACHED. Choose another gesture."
                    elif collection_state == "COLLECTING":
                        collection_state, message = "PAUSED", "Collection paused."
                    else:
                        collection_state, message = "COLLECTING", "Collection started."
                elif key == ord("r"):
                    collection_state, last_saved_at = "READY", 0.0
                    message = "Collection state reset. Saved dataset rows were preserved."
                elif key == ord("h"):
                    print_help()
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Dataset preserved at {DATASET_PATH}; total samples: {sum(counts.values())}")


if __name__ == "__main__":
    main()
