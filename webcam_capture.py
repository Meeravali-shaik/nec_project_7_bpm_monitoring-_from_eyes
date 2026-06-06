"""
webcam_capture.py
Simple OpenCV webcam wrapper with configurable resolution and FPS.
"""

import cv2


class WebcamCapture:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

    def read(self):
        """Return (success, frame)."""
        return self.cap.read()

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        self.cap.release()
