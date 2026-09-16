"""MediaPipe detection and Random Forest inference for VisionControl."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from feature_extraction import FEATURE_COUNT, features_from_hand_landmarks


class GestureEngineError(RuntimeError):
    """Raised for model, MediaPipe, or landmark-processing failures."""


@dataclass
class GestureResult:
    """One frame's safe classification result for the controller."""

    gesture: str = "NO HAND"
    confidence: float = 0.0
    hand_landmarks: object = None


class GestureEngine:
    """Own the complete camera-frame-to-gesture ML pipeline.

    Input frame -> MediaPipe landmarks -> shared 63 features -> Random Forest ->
    confidence filtering. This is intentionally the only live inference path.
    """

    def __init__(self, settings, model_path="models/gesture_model.pkl"):
        self.settings = settings
        self.model_path = Path(model_path)
        self.model = None
        self.hands = None
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils

    @property
    def is_ready(self):
        """Whether both model and MediaPipe tracker are usable."""
        return self.model is not None and self.hands is not None

    def start(self):
        """Load and validate the production model, then initialise MediaPipe."""
        if not self.model_path.exists():
            raise GestureEngineError(f"Gesture model not found: {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            expected_features = getattr(self.model, "n_features_in_", None)
            if expected_features != FEATURE_COUNT:
                raise GestureEngineError(
                    f"Model expects {expected_features} features; VisionControl requires {FEATURE_COUNT}."
                )
            if not hasattr(self.model, "predict_proba") or not hasattr(self.model, "classes_"):
                raise GestureEngineError("Model must provide predict_proba and classes_.")
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.settings.max_num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )
        except GestureEngineError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise GestureEngineError(f"Gesture engine could not start: {error}") from error

    def process(self, frame):
        """Detect one hand, classify it, and convert low confidence to UNKNOWN."""
        if not self.is_ready:
            raise GestureEngineError("Gesture engine has not been started.")
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
        except Exception as error:
            raise GestureEngineError(f"MediaPipe hand detection failed: {error}") from error

        if not results.multi_hand_landmarks:
            return GestureResult()

        hand = results.multi_hand_landmarks[0]
        self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
        if not self.settings.gesture_recognition_enabled:
            return GestureResult("DISABLED", 0.0, hand)

        try:
            features = features_from_hand_landmarks(hand)
            if features.shape != (1, FEATURE_COUNT):
                raise ValueError(f"Invalid feature shape: {features.shape}")
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
            gesture = str(self.model.classes_[int(np.argmax(probabilities))]).upper().strip()
        except Exception as error:
            raise GestureEngineError(f"Gesture prediction failed: {error}") from error

        if confidence < self.settings.confidence_threshold:
            gesture = "UNKNOWN"
        return GestureResult(gesture, confidence, hand)

    def close(self):
        """Close MediaPipe resources; safe during partial startup failures."""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
