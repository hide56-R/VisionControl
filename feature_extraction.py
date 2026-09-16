"""Shared MediaPipe hand-landmark features for training and live recognition."""

import numpy as np


LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
FEATURE_COUNT = LANDMARK_COUNT * COORDINATES_PER_LANDMARK


def normalize_landmarks(landmarks):
    """Return the project's wrist-relative, scale-normalized 63-value feature row."""
    array = np.asarray(landmarks, dtype=np.float32)

    if array.shape != (LANDMARK_COUNT, COORDINATES_PER_LANDMARK):
        raise ValueError(
            "Expected 21 landmarks with x, y, z coordinates; "
            f"received shape {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError("Landmark coordinates must be finite numbers.")

    wrist = array[0].copy()
    array = array - wrist

    maximum = np.max(np.abs(array))
    if maximum > 0:
        array = array / maximum

    return array.reshape(1, FEATURE_COUNT)


def features_from_hand_landmarks(hand_landmarks):
    """Extract normalized features from MediaPipe's NormalizedLandmarkList."""
    landmarks = [
        [landmark.x, landmark.y, landmark.z]
        for landmark in hand_landmarks.landmark
    ]
    return normalize_landmarks(landmarks)


def features_from_flat_row(values):
    """Extract normalized features from one CSV row containing 63 values."""
    array = np.asarray(values, dtype=np.float32)

    if array.size != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} landmark values; received {array.size}."
        )

    return normalize_landmarks(
        array.reshape(LANDMARK_COUNT, COORDINATES_PER_LANDMARK)
    )
