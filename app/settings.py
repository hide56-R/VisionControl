from dataclasses import dataclass


@dataclass
class Settings:
    """Central runtime configuration."""
    confidence_threshold: float = 0.70
    action_cooldown_seconds: float = 1.5
    mouse_sensitivity: float = 1.0
    mouse_smoothing: float = 0.25
    active_control_mode: str = "MEDIA"
    camera_width: int = 640
    camera_height: int = 480
    max_num_hands: int = 1
    gesture_recognition_enabled: bool = True
    mouse_control_enabled: bool = False

    def validate(self):
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if self.action_cooldown_seconds < 0 or self.mouse_sensitivity <= 0:
            raise ValueError("Cooldown must be non-negative and sensitivity positive.")
        if not 0 < self.mouse_smoothing <= 1:
            raise ValueError("mouse_smoothing must be in (0, 1].")
        if self.active_control_mode not in {"MEDIA", "MOUSE"}:
            raise ValueError("active_control_mode must be MEDIA or MOUSE.")
        if self.camera_width <= 0 or self.camera_height <= 0 or self.max_num_hands < 1:
            raise ValueError("Camera dimensions and max_num_hands must be positive.")
