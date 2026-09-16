"""Coordinate VisionControl's camera, gesture engine, actions, and dashboard."""

import time

import cv2

import actions
import mouse_control
from app.camera import Camera, CameraError
from app.gesture_engine import GestureEngine, GestureEngineError
from app.settings import Settings
from app.ui import Dashboard


class VisionControlApp:
    """Application orchestration; it never performs ML feature extraction itself."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.settings.validate()
        self.camera = Camera(self.settings.camera_width, self.settings.camera_height)
        self.engine = GestureEngine(self.settings)
        self.ui = Dashboard()
        self.running = False
        self.last_mouse_gesture = None
        self.last_mouse_time = 0.0
        self.fps_start = time.time()
        self.fps_count = 0
        self.fps = 0

        # Existing action/mouse modules retain their mappings; settings configure timing only.
        actions.COOLDOWN = self.settings.action_cooldown_seconds
        mouse_control.SMOOTHING = self.settings.mouse_smoothing
        mouse_control.FRAME_WIDTH = self.settings.camera_width
        mouse_control.FRAME_HEIGHT = self.settings.camera_height

    def start(self):
        """Start dependencies in order and clean up if camera startup fails."""
        try:
            self.engine.start()
            self.camera.open()
            self.running = True
        except Exception:
            self.close()
            raise

    def _record_mouse_action(self, name):
        actions.last_action = name
        actions.action_status = "EXECUTED"
        actions.action_history.appendleft(f"{time.strftime('%H:%M:%S')} - {name}")

    def _handle_mouse_mode(self, result, frame):
        """Safely route mouse gestures only while the explicit toggle is ON."""
        if not self.settings.mouse_control_enabled:
            return
        gesture = result.gesture
        if gesture in {"UNKNOWN", "NO HAND", "DISABLED"}:
            self.last_mouse_gesture = None
            return

        try:
            if gesture == "POINT":
                tip = result.hand_landmarks.landmark[8]
                height, width = frame.shape[:2]
                x = ((tip.x - 0.5) * self.settings.mouse_sensitivity + 0.5) * width
                y = ((tip.y - 0.5) * self.settings.mouse_sensitivity + 0.5) * height
                mouse_control.move_mouse(int(x), int(y))
                self.last_mouse_gesture = None
                return

            if gesture == self.last_mouse_gesture:
                return
            if time.time() - self.last_mouse_time < self.settings.action_cooldown_seconds:
                return
            command = {
                "FIST": (mouse_control.mouse_click, "MOUSE LEFT CLICK"),
                "ROCK": (mouse_control.mouse_right_click, "MOUSE RIGHT CLICK"),
                "PEACE": (mouse_control.mouse_double_click, "MOUSE DOUBLE CLICK"),
            }.get(gesture)
            if command is None:
                return
            command[0]()
            self._record_mouse_action(command[1])
            self.last_mouse_gesture = gesture
            self.last_mouse_time = time.time()
        except Exception as error:
            actions.last_action = "MOUSE ERROR"
            actions.action_status = str(error)[:32]

    def dispatch(self, result, frame):
        """Route an already-filtered gesture to exactly one control mode."""
        if self.settings.active_control_mode == "MOUSE":
            # Also passes invalid states so mouse debounce can reset after hand loss.
            self._handle_mouse_mode(result, frame)
            return
        if result.gesture == "NO HAND":
            # A new hand appearance may legitimately repeat the previous gesture.
            # The action cooldown still prevents rapid re-triggering.
            actions.last_gesture = None
        if result.gesture in {"NO HAND", "UNKNOWN", "DISABLED"}:
            return
        try:
            actions.perform_gesture_action(result.gesture)
        except Exception as error:
            actions.last_action = "ACTION ERROR"
            actions.action_status = str(error)[:32]

    def _update_fps(self):
        self.fps_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps = int(self.fps_count / elapsed)
            self.fps_count = 0
            self.fps_start = time.time()

    def state(self, result):
        """Return display-only state for the dashboard."""
        return {
            "camera_status": "CONNECTED" if self.camera.capture is not None else "DISCONNECTED",
            "model_status": "LOADED" if self.engine.model is not None else "ERROR",
            "engine_status": "READY" if self.engine.is_ready else "ERROR",
            "mode": self.settings.active_control_mode,
            "gesture": result.gesture,
            "confidence": result.confidence * 100,
            "last_action": actions.last_action,
            "history": list(actions.action_history),
            "mouse_enabled": self.settings.mouse_control_enabled,
            "recognition_enabled": self.settings.gesture_recognition_enabled,
            "fps": self.fps,
        }

    def key(self, key):
        """Handle all keyboard controls in one place."""
        if key == ord("q"):
            self.running = False
        elif key == ord("m"):
            self.settings.active_control_mode = "MOUSE" if self.settings.active_control_mode == "MEDIA" else "MEDIA"
            self.last_mouse_gesture = None
            actions.last_gesture = None
            actions.last_action = f"{self.settings.active_control_mode} MODE ENABLED"
        elif key == ord("x"):
            self.settings.mouse_control_enabled = not self.settings.mouse_control_enabled
            state = "ON" if self.settings.mouse_control_enabled else "OFF"
            actions.last_action, actions.action_status = f"MOUSE CONTROL {state}", "READY"
            self.last_mouse_gesture = None
        elif key == ord("g"):
            self.settings.gesture_recognition_enabled = not self.settings.gesture_recognition_enabled
            state = "ON" if self.settings.gesture_recognition_enabled else "OFF"
            actions.last_action, actions.action_status = f"RECOGNITION {state}", "READY"
        elif key == ord("r"):
            actions.reset_action_state()
            self.last_mouse_gesture = None
            self.last_mouse_time = 0.0
        elif key == ord("h"):
            print("M: mode | X: mouse ON/OFF | G: recognition ON/OFF | R: reset | Q: quit")

    def run(self):
        """Run the frame loop until quit or a friendly runtime error occurs."""
        try:
            while self.running:
                frame = cv2.flip(self.camera.read(), 1)
                result = self.engine.process(frame)
                self.dispatch(result, frame)
                self._update_fps()
                self.ui.render(frame, self.state(result))
                self.ui.show(frame)
                self.key(cv2.waitKey(1) & 0xFF)
        except (CameraError, GestureEngineError) as error:
            print(f"[VISIONCONTROL ERROR] {error}")
        finally:
            self.close()

    def close(self):
        """Release camera, MediaPipe, and OpenCV window resources safely."""
        self.running = False
        self.camera.release()
        self.engine.close()
        self.ui.close()
