"""Minimal camera-first OpenCV overlay for VisionControl."""

import cv2


class Dashboard:
    """Render essential recognition state without obscuring the camera feed."""

    WINDOW_TITLE = "VisionControl - AI Gesture Control"
    WHITE = (245, 245, 245)
    MUTED = (165, 170, 180)
    GREEN = (80, 210, 130)
    BLUE = (220, 150, 70)
    YELLOW = (0, 220, 255)
    DARK = (18, 18, 24)

    def text(self, frame, value, point, color=WHITE, size=0.5, thickness=1):
        """Draw anti-aliased dashboard text."""
        cv2.putText(
            frame,
            str(value),
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _overlay(frame, top_left, bottom_right, color, opacity=0.80):
        """Draw a translucent rectangle while preserving the scene behind it."""
        layer = frame.copy()
        cv2.rectangle(layer, top_left, bottom_right, color, -1)
        cv2.addWeighted(layer, opacity, frame, 1 - opacity, 0, frame)

    def render(self, frame, state):
        """Render a header, a small status card, and a short controls line."""
        height, width = frame.shape[:2]

        # Thin header leaves the camera view as the visual focus.
        self._overlay(frame, (0, 0), (width, 58), self.DARK, 0.88)
        self.text(frame, "VISIONCONTROL", (18, 27), self.WHITE, 0.72, 2)
        self.text(frame, "AI GESTURE CONTROL", (18, 47), self.MUTED, 0.38)
        self.text(frame, f"FPS {state['fps']}", (width - 86, 28), self.GREEN, 0.45)

        # Compact card: only current control information, no long history panels.
        card_width = min(335, width - 30)
        card_top = 72
        card_bottom = min(card_top + 106, height - 30)
        self._overlay(frame, (15, card_top), (15 + card_width, card_bottom), self.DARK)
        gesture = state["gesture"].replace("_", " ")
        mode_color = self.BLUE if state["mode"] == "MOUSE" else self.GREEN
        recognition = "ON" if state["recognition_enabled"] else "OFF"
        mouse = "ON" if state["mouse_enabled"] else "OFF"
        self.text(frame, f"GESTURE  {gesture}", (28, card_top + 27), self.WHITE, 0.56, 2)
        self.text(frame, f"CONFIDENCE  {state['confidence']:.1f}%", (28, card_top + 51), self.YELLOW, 0.43)
        self.text(frame, f"ACTION  {state['last_action']}", (28, card_top + 75), self.GREEN, 0.43)
        self.text(frame, f"{state['mode']}  |  REC {recognition}  |  MOUSE {mouse}", (28, card_top + 97), mode_color, 0.40)

        self._overlay(frame, (0, height - 27), (width, height), self.DARK, 0.85)
        self.text(
            frame,
            "M Mode   X Mouse   G Recognition   R Reset   H Help   Q Quit",
            (16, height - 9),
            self.MUTED,
            0.37,
        )

    def show(self, frame):
        """Display the rendered camera frame."""
        cv2.imshow(self.WINDOW_TITLE, frame)

    @staticmethod
    def close():
        """Close OpenCV windows during normal shutdown or startup failure."""
        cv2.destroyAllWindows()
