"""Video capture, object detection, pose/hand, hand-object interaction.

All heavy libraries (OpenCV, ultralytics/YOLO, MediaPipe) are imported
*lazily* inside the functions/classes that need them. Importing this module
on a machine without them must not crash — every component reports
``available == False`` with a clear reason, and the system falls back to
colour-based container detection (HSV, OpenCV only) or to the simulated
demo feed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# -- lazy availability flags (no imports at module load) --------------------
try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def require_cv2() -> None:
    if not HAS_CV2:
        raise ImportError(
            "opencv-python is not installed. Install the full requirements "
            "(pip install -r requirements.txt) or run in demo mode "
            "(python demo.py --scenario correct), which needs no camera."
        )


@dataclass
class Detection:
    """One detected object: normalised centre, size, label, confidence."""

    label: str
    x: float
    y: float
    size: float
    confidence: float = 1.0

    def centre(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class HandState:
    visible: bool = False
    x: float = 0.5
    y: float = 0.5
    landmarks: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Colour container detection (no-model fallback, OpenCV only)
# ---------------------------------------------------------------------------

def detect_coloured_containers(frame) -> list[Detection]:
    """Detect red / yellow containers by HSV colour. Returns [] if OpenCV
    is absent. This is the fallback when no YOLO model is available: the
    experiment props are deliberately coloured, so colour is a real signal,
    not a placeholder."""
    require_cv2()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    out: list[Detection] = []
    # (label, lower HSV, upper HSV) — red wraps around hue 0, so two ranges.
    ranges = [
        ("red", (0, 70, 50), (10, 255, 255)),
        ("red", (170, 70, 50), (180, 255, 255)),
        ("yellow", (20, 100, 100), (35, 255, 255)),
    ]
    for label, lo, hi in ranges:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore specks
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = (m["m10"] / m["m00"]) / width
            cy = (m["m01"] / m["m00"]) / height
            size = math.sqrt(area) / max(width, height)
            out.append(Detection(label=label, x=cx, y=cy, size=size, confidence=0.7))
    return out


# ---------------------------------------------------------------------------
# YOLO object detection (lazy)
# ---------------------------------------------------------------------------

class YoloDetector:
    """Ultralytics YOLO wrapper. Loads lazily; ``available`` is False with
    a reason when ultralytics or the weights file is missing."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        self.weights = weights
        self.model: Any = None
        self.available = False
        self.reason = ""
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(weights)
            self.available = True
        except Exception as exc:  # missing lib, missing weights, no network
            self.reason = str(exc)

    def detect(self, frame) -> list[Detection]:
        if not self.available or self.model is None:
            return []
        results = self.model(frame, verbose=False)
        out: list[Detection] = []
        height, width = frame.shape[:2]
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names.get(cls, str(cls))
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                out.append(Detection(
                    label=name, x=((x1 + x2) / 2) / width,
                    y=((y1 + y2) / 2) / height,
                    size=max(x2 - x1, y2 - y1) / max(width, height),
                    confidence=float(box.conf[0]),
                ))
        return out


# ---------------------------------------------------------------------------
# MediaPipe pose / hands (lazy)
# ---------------------------------------------------------------------------

class PoseHandTracker:
    """MediaPipe Pose + Hands wrapper, lazy-loaded."""

    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        self._pose = None
        self._hands = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, min_detection_confidence=0.5)
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.5)
            self.available = True
        except Exception as exc:
            self.reason = str(exc)

    def track(self, frame_rgb) -> tuple[list[tuple[float, float]], HandState]:
        """Return (pose_landmarks, hand_state). Empty results when unavailable."""
        if not self.available:
            return [], HandState()
        pose_landmarks: list[tuple[float, float]] = []
        hand = HandState()
        pose_res = self._pose.process(frame_rgb)
        if pose_res.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y) for lm in pose_res.pose_landmarks.landmark]
        hands_res = self._hands.process(frame_rgb)
        if hands_res.multi_hand_landmarks:
            lm = hands_res.multi_hand_landmarks[0].landmark
            pts = [(p.x, p.y) for p in lm]
            hand = HandState(
                visible=True, x=sum(p[0] for p in pts) / len(pts),
                y=sum(p[1] for p in pts) / len(pts), landmarks=pts)
        return pose_landmarks, hand


# ---------------------------------------------------------------------------
# Hand-object interaction
# ---------------------------------------------------------------------------

def hand_object_distance(hand: HandState, det: Detection) -> float:
    return math.hypot(hand.x - det.x, hand.y - det.y)


def is_holding(hand: HandState, det: Detection, threshold: float = 0.10) -> bool:
    return hand.visible and hand_object_distance(hand, det) < threshold


def is_near(hand: HandState, det: Detection, threshold: float = 0.18) -> bool:
    return hand.visible and hand_object_distance(hand, det) < threshold


class InteractionTracker:
    """Tracks holding over time: an object counts as held only after the
    hand stays close to it for ``hold_frames`` consecutive frames."""

    def __init__(self, hold_frames: int = 5) -> None:
        self.hold_frames = hold_frames
        self._counts: dict[str, int] = {}

    def update(self, hand: HandState, detections: list[Detection]) -> dict[str, bool]:
        held: dict[str, bool] = {}
        for det in detections:
            if is_holding(hand, det):
                self._counts[det.label] = self._counts.get(det.label, 0) + 1
            else:
                self._counts[det.label] = 0
            held[det.label] = self._counts.get(det.label, 0) >= self.hold_frames
        return held


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def camera_available(index: int = 0) -> bool:
    """True if a camera can be opened. False (not an exception) when OpenCV
    is missing or no camera is present — demo mode uses this check."""
    if not HAS_CV2:
        return False
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    cap.release()
    return bool(ok)


class VideoCapture:
    """Thin OpenCV capture wrapper with a clean failure message."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        require_cv2()
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"No camera available at index {index}. Run in demo mode "
                "(python demo.py --scenario correct) or check the webcam."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):  # -> tuple[bool, frame]
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()
