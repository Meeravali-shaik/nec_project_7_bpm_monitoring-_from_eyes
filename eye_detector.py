"""
eye_detector.py
Detects eyes and extracts 16 landmarks per eye using MediaPipe Face Mesh.
Includes EAR (Eye Aspect Ratio) calculation to detect open/closed eyes.

Visualization:
- Green box = Left Eye
- Orange box = Right Eye
"""

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Face Mesh eye landmark indices
LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

# EAR landmark positions within the 16-point eye arrays
EAR_P1 = 0
EAR_P2 = 8
EAR_P3 = 13
EAR_P4 = 5
EAR_P5 = 14
EAR_P6 = 6

EAR_OPEN_THRESHOLD = 0.20


def _ear(eye_pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio:
    EAR = (||p3-p4|| + ||p5-p6||) / (2 * ||p1-p2||)
    """

    try:
        p1 = eye_pts[EAR_P1]
        p2 = eye_pts[EAR_P2]
        p3 = eye_pts[EAR_P3]
        p4 = eye_pts[EAR_P4]
        p5 = eye_pts[EAR_P5]
        p6 = eye_pts[EAR_P6]

        vert1 = np.linalg.norm(p3 - p4)
        vert2 = np.linalg.norm(p5 - p6)
        horiz = np.linalg.norm(p1 - p2)

        if horiz < 1e-6:
            return 0.0

        return (vert1 + vert2) / (2.0 * horiz)

    except Exception:
        return 0.0


class EyeDetector:

    def __init__(
        self,
        max_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        ear_threshold=EAR_OPEN_THRESHOLD
    ):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.ear_threshold = ear_threshold

        self.left_ear = 0.0
        self.right_ear = 0.0

    def detect(self, frame):
        """
        Returns:
            all_pts        : all face landmarks
            left_eye_pts   : 16 eye landmarks
            right_eye_pts  : 16 eye landmarks

        If both eyes are closed:
            return (all_pts, None, None)
        """

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        self.left_ear = 0.0
        self.right_ear = 0.0

        if not results.multi_face_landmarks:
            return None, None, None

        face_lm = results.multi_face_landmarks[0]

        all_pts = np.array(
            [(lm.x * w, lm.y * h) for lm in face_lm.landmark],
            dtype=np.float32
        )

        left_eye_pts = all_pts[LEFT_EYE_INDICES]
        right_eye_pts = all_pts[RIGHT_EYE_INDICES]

        self.left_ear = _ear(left_eye_pts)
        self.right_ear = _ear(right_eye_pts)

        left_open = self.left_ear >= self.ear_threshold
        right_open = self.right_ear >= self.ear_threshold

        if not left_open and not right_open:
            return all_pts, None, None

        return all_pts, left_eye_pts, right_eye_pts

    def eyes_open(self):
        """
        True if at least one eye is open.
        """
        return (
            self.left_ear >= self.ear_threshold or
            self.right_ear >= self.ear_threshold
        )

    def draw_landmarks(self, frame, left_eye_pts, right_eye_pts):
        """
        Draw eye bounding boxes instead of landmark points.
        """

        padding = 6

        # LEFT EYE
        if left_eye_pts is not None:

            left_pts = np.array(left_eye_pts, dtype=np.int32)

            x, y, w, h = cv2.boundingRect(left_pts)

            cv2.rectangle(
                frame,
                (x - padding, y - padding),
                (x + w + padding, y + h + padding),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "LEFT",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # RIGHT EYE
        if right_eye_pts is not None:

            right_pts = np.array(right_eye_pts, dtype=np.int32)

            x, y, w, h = cv2.boundingRect(right_pts)

            cv2.rectangle(
                frame,
                (x - padding, y - padding),
                (x + w + padding, y + h + padding),
                (0, 200, 255),
                2
            )

            cv2.putText(
                frame,
                "RIGHT",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1
            )

        return frame

    def close(self):
        self.face_mesh.close()