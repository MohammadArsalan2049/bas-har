"""Activity labels and recognisers.

Two paths, kept separate on purpose:

1. Rule-based fallback (``RuleBasedActivityRecognizer``) — deterministic,
   works from hand/object features with no ML model at all. This is what
   demo mode uses, and what the system falls back to when no trained model
   is available. It is *not* a stub: it reasons about hand position,
   hand-object proximity, holding and motion.

2. Learned temporal model (``src.bas_har.models.TemporalHAR``, PyTorch) —
   an LSTM over a sequence of feature vectors. Used only when a trained
   checkpoint exists; see ``scripts/train_har.py``. Accuracy for that model
   is "to be measured on your recorded dataset" — no number is claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ACTIVITY_LABELS = [
    "OPEN_BOX",
    "PICK_RED",
    "PICK_YELLOW",
    "OPEN_RED",
    "TRANSFER",
    "MIX",
    "CLOSE_CONTAINERS",
    "RETURN_CONTAINERS",
    "IDLE",
    "UNKNOWN",
]

# Normalised image coordinates (0..1). A hand is "near" an object when the
# distance between hand centre and object centre is below this.
NEAR_THRESHOLD = 0.18
HOLD_THRESHOLD = 0.10
# Hand speed (normalised units / second) above which motion counts as mixing.
MIX_SPEED = 0.35


@dataclass
class FrameFeatures:
    """One frame of hand + object features, in normalised coordinates.

    All fields are plain floats / bools so this works with or without numpy.
    ``objects`` maps an object name ("box", "red", "yellow") to its centre
    (x, y) and a size estimate, e.g. ``{"red": (0.5, 0.5, 0.12)}``.
    """

    hand_x: float = 0.5
    hand_y: float = 0.5
    hand_speed: float = 0.0
    hand_visible: bool = False
    # hand movement direction over the last frames, for transfer detection
    hand_dx: float = 0.0
    hand_dy: float = 0.0
    objects: dict = field(default_factory=dict)  # name -> (x, y, size)

    def distance_to(self, name: str) -> Optional[float]:
        obj = self.objects.get(name)
        if obj is None or not self.hand_visible:
            return None
        return ((self.hand_x - obj[0]) ** 2 + (self.hand_y - obj[1]) ** 2) ** 0.5

    def near(self, name: str, threshold: float = NEAR_THRESHOLD) -> bool:
        d = self.distance_to(name)
        return d is not None and d < threshold

    def holding(self, name: str) -> bool:
        return self.near(name, HOLD_THRESHOLD)

    def to_vector(self) -> list[float]:
        """Fixed-length feature vector for the temporal model (length 12)."""
        vec = [self.hand_x, self.hand_y, self.hand_speed,
               self.hand_dx, self.hand_dy, 1.0 if self.hand_visible else 0.0]
        for name in ("box", "red", "yellow"):
            obj = self.objects.get(name)
            if obj is None:
                vec += [0.0, 0.0]
            else:
                vec += [obj[0], obj[1]]
        return vec


def extract_features(
    hand_landmarks: Optional[list[tuple[float, float]]] = None,
    pose_landmarks: Optional[list[tuple[float, float]]] = None,
    objects: Optional[dict] = None,
    prev_hand: Optional[tuple[float, float]] = None,
    dt: float = 1.0 / 30.0,
) -> FrameFeatures:
    """Build ``FrameFeatures`` from raw landmarks / detections.

    ``hand_landmarks`` are normalised (x, y) wrist/fingertip points; the
    hand centre is their mean. ``objects`` is the detector output
    (``{"red": (x, y, size), ...}``). Works with plain lists — no numpy,
    no MediaPipe types required, so callers can pass whatever their
    detector produced.
    """
    feats = FrameFeatures(objects=dict(objects or {}))
    if hand_landmarks:
        xs = [p[0] for p in hand_landmarks]
        ys = [p[1] for p in hand_landmarks]
        feats.hand_x = sum(xs) / len(xs)
        feats.hand_y = sum(ys) / len(ys)
        feats.hand_visible = True
        if prev_hand is not None and dt > 0:
            dx = feats.hand_x - prev_hand[0]
            dy = feats.hand_y - prev_hand[1]
            feats.hand_dx, feats.hand_dy = dx, dy
            feats.hand_speed = (dx * dx + dy * dy) ** 0.5 / dt
    elif pose_landmarks:
        # Fall back to the wrist landmark (index 15/16 in MediaPipe Pose)
        # if no hand landmarks are available.
        wrist = pose_landmarks[15] if len(pose_landmarks) > 15 else None
        if wrist is not None:
            feats.hand_x, feats.hand_y = wrist
            feats.hand_visible = True
    return feats


class RuleBasedActivityRecognizer:
    """Deterministic recogniser from hand/object features.

    Rules, in priority order (a frame is classified by the first rule that
    fires — this ordering is the whole point, and is what the tests check):
      1. Fast repeated hand motion while holding a container -> MIX
      2. Hand holding red, moving towards yellow -> TRANSFER
      3. Hand holding red, near box edge / moving away from box -> PICK_RED
      4. Hand holding yellow -> PICK_YELLOW
      5. Hand near red, twisting (speed moderate, not translating) -> OPEN_RED
      6. Hand near box -> OPEN_BOX / CLOSE_CONTAINERS (distinguished by
         whether the box was already opened — caller passes context)
      7. Hand moving back towards the box with a container -> RETURN_CONTAINERS
      8. Otherwise -> IDLE

    Confidence is deliberately modest (0.55–0.85): the rule path is a
    fallback, and the FSM's 0.70 threshold will hold its weakest calls.
    """

    def __init__(self) -> None:
        self.box_opened: bool = False
        self._prev_holding: Optional[str] = None

    def reset(self) -> None:
        self.box_opened = False
        self._prev_holding = None

    def recognize(self, feats: FrameFeatures) -> tuple[str, float]:
        if not feats.hand_visible:
            return "IDLE", 0.90

        holding_red = feats.holding("red")
        holding_yellow = feats.holding("yellow")
        near_box = feats.near("box")
        near_red = feats.near("red")
        near_yellow = feats.near("yellow")

        # 1. Mixing: fast hand motion while holding something.
        if (holding_red or holding_yellow) and feats.hand_speed > MIX_SPEED:
            return "MIX", 0.80

        # 2. Transfer: holding red, moving towards yellow.
        if holding_red and near_yellow:
            return "TRANSFER", 0.82

        # 7/3. Holding a container and moving back to the box = returning it.
        if (holding_red or holding_yellow) and near_box and self.box_opened:
            # If the box was opened and we are bringing a container back,
            # treat as RETURN once both containers were handled; otherwise
            # a pick is more likely. The caller (FSM state) disambiguates
            # in practice — here we prefer RETURN when hand is slow.
            if feats.hand_speed < 0.10:
                return "RETURN_CONTAINERS", 0.75

        # 3/4. Picking up a container.
        if holding_red and not self.box_opened:
            return "PICK_RED", 0.78
        if holding_red:
            return "PICK_RED", 0.72
        if holding_yellow:
            return "PICK_YELLOW", 0.78

        # 5. Opening the red container: hand near red, moderate motion.
        if near_red and 0.05 < feats.hand_speed < MIX_SPEED:
            return "OPEN_RED", 0.74

        # 6. Near the box: opening it (first time) or closing containers.
        if near_box:
            if not self.box_opened:
                self.box_opened = True
                return "OPEN_BOX", 0.80
            return "CLOSE_CONTAINERS", 0.68  # below FSM threshold: held

        # Hand near yellow without holding it: reaching for it.
        if near_yellow:
            return "PICK_YELLOW", 0.62  # below threshold: held, not counted

        if near_red:
            return "OPEN_RED", 0.62

        return "IDLE", 0.85

    def recognize_sequence(self, frames: list[FrameFeatures]) -> tuple[str, float]:
        """Classify a short clip by majority vote over per-frame rules."""
        if not frames:
            return "IDLE", 0.0
        votes: dict[str, list[float]] = {}
        for f in frames:
            label, conf = self.recognize(f)
            votes.setdefault(label, []).append(conf)
        best = max(votes.items(), key=lambda kv: len(kv[1]))
        label, confs = best
        return label, sum(confs) / len(confs)
