"""OpenCV camera lifecycle used by the VisionControl application."""

import cv2


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or a frame cannot be read."""


class Camera:
    """Own exactly one OpenCV capture device and release it safely."""

    def __init__(self, width, height, index=0):
        self.width = width
        self.height = height
        self.index = index
        self.capture = None

    def open(self):
        """Open the configured device and request the configured resolution."""
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.release()
            raise CameraError("Camera could not be opened. Check its connection and close other camera apps.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Return one frame, or raise a readable error instead of invalid data."""
        if self.capture is None:
            raise CameraError("Camera has not been opened.")
        success, frame = self.capture.read()
        if not success or frame is None:
            raise CameraError("Camera frame could not be read.")
        return frame

    def release(self):
        """Release the device; safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
