"""Experiment protocol and finite state machine (FSM).

Pure Python — no camera, ML or network imports in this module, so it is
unit-testable on any machine.

The experiment (Bharatiya Antariksh Station, BAS) is an 8-step procedure
with a main box, a red container and a yellow container:

    1 OPEN_BOX           Open the main box
    2 PICK_RED           Take out the red container
    3 PICK_YELLOW        Take out the yellow container
    4 OPEN_RED           Open the red container
    5 TRANSFER           Transfer its contents into the yellow container
    6 MIX                Mix the contents
    7 CLOSE_CONTAINERS   Close the containers
    8 RETURN_CONTAINERS  Put both containers back

Classification of a *confirmed* recognised activity:
  - CORRECT      : the expected next step
  - SKIPPED      : a later step, so intermediate step(s) were missed
  - OUT_OF_ORDER : an already-completed step, or a step out of sequence
  - UNKNOWN      : not a protocol activity at all
  - INCOMPLETE   : an activity that started but did not finish (see
                   ``mark_incomplete``)

Two safeguards stop the system crying wolf on every flicker of the
recogniser:
  1. Confidence threshold (default 0.70) — low-confidence detections are
     *held*, never flagged.
  2. Temporal confirmation (``TemporalConfirmer``) — a raw frame-level
     detection must persist for ~0.8 s before it is treated as a real
     activity at all.
  3. Grace period (default 2.0 s) — a suspect wrong step only becomes an
     *active* alert after the grace period, so a brief mis-read that the
     recogniser corrects on its own does not raise an alarm.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Protocol data
# ---------------------------------------------------------------------------

STEP_NAMES = [
    "OPEN_BOX",
    "PICK_RED",
    "PICK_YELLOW",
    "OPEN_RED",
    "TRANSFER",
    "MIX",
    "CLOSE_CONTAINERS",
    "RETURN_CONTAINERS",
]

STEP_DESCRIPTIONS = {
    "OPEN_BOX": "Open the main box",
    "PICK_RED": "Take out the red container",
    "PICK_YELLOW": "Take out the yellow container",
    "OPEN_RED": "Open the red container",
    "TRANSFER": "Transfer the red container contents into the yellow container",
    "MIX": "Mix the contents",
    "CLOSE_CONTAINERS": "Close the containers",
    "RETURN_CONTAINERS": "Put both containers back",
}

NEXT_STEP_HINTS = {
    "OPEN_BOX": "Open the main box.",
    "PICK_RED": "Take out the red container.",
    "PICK_YELLOW": "Take out the yellow container.",
    "OPEN_RED": "Open the red container.",
    "TRANSFER": "Transfer the contents into the yellow container.",
    "MIX": "Mix the contents.",
    "CLOSE_CONTAINERS": "Close the containers.",
    "RETURN_CONTAINERS": "Put both containers back.",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_CONFIRM_SECONDS = 0.8
DEFAULT_GRACE_SECONDS = 2.0


@dataclass
class StepResult:
    """Result of feeding one confirmed activity to the FSM."""

    activity: str
    status: str  # CORRECT | SKIPPED | OUT_OF_ORDER | UNKNOWN | INCOMPLETE | HELD
    confidence: float
    timestamp: float
    message: str
    expected: Optional[str] = None
    missed_steps: list[str] = field(default_factory=list)
    next_step: Optional[str] = None


@dataclass
class Alert:
    """A protocol violation, raised after the grace period."""

    kind: str  # SKIPPED | OUT_OF_ORDER | UNKNOWN | INCOMPLETE
    activity: str
    timestamp: float
    message: str
    recovery: str
    missed_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Temporal confirmation (raw recogniser output -> confirmed activity)
# ---------------------------------------------------------------------------

class TemporalConfirmer:
    """Accept a raw detection only after it persists.

    Feed raw (activity, confidence, timestamp) observations, typically one
    per video frame. Returns the activity label once it has been observed
    continuously for ``confirm_seconds`` at or above ``min_confidence``,
    otherwise ``None``. A different label, or a low-confidence frame, resets
    the timer — this is what stops a single mis-read frame from becoming
    a "step completed".
    """

    def __init__(
        self,
        confirm_seconds: float = DEFAULT_CONFIRM_SECONDS,
        min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confirm_seconds = confirm_seconds
        self.min_confidence = min_confidence
        self._candidate: Optional[str] = None
        self._candidate_since: float = 0.0
        self._confirmed: set[str] = set()

    def feed(self, activity: str, confidence: float, timestamp: float) -> Optional[str]:
        if confidence < self.min_confidence:
            self._candidate = None
            return None
        if activity != self._candidate:
            self._candidate = activity
            self._candidate_since = timestamp
            return None
        if timestamp - self._candidate_since >= self.confirm_seconds:
            if activity not in self._confirmed:
                self._confirmed.add(activity)
                return activity
            return None
        return None

    def reset(self) -> None:
        self._candidate = None
        self._confirmed = set()


# ---------------------------------------------------------------------------
# Finite state machine
# ---------------------------------------------------------------------------

class ProtocolEngine:
    """Validates recognised activities against the 8-step protocol."""

    def __init__(
        self,
        protocol_path: Optional[str | Path] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
    ) -> None:
        self.steps: list[str] = list(STEP_NAMES)
        self.descriptions: dict[str, str] = dict(STEP_DESCRIPTIONS)
        if protocol_path is not None:
            self.load_protocol(protocol_path)
        self.confidence_threshold = confidence_threshold
        self.grace_seconds = grace_seconds
        self.current_index: int = 0
        self.completed: list[str] = []
        self.skipped: list[str] = []
        self.results: list[StepResult] = []
        self._pending_alerts: list[Alert] = []  # raised, inside grace period
        self.alerts: list[Alert] = []           # active, grace period passed
        self.start_time: float = time.time()

    # -- protocol loading ---------------------------------------------------
    def load_protocol(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = [s["activity"] for s in data["steps"]]
        self.steps = steps
        self.descriptions = {s["activity"]: s["description"] for s in data["steps"]}

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "ProtocolEngine":
        return cls(protocol_path=path, **kwargs)

    # -- state --------------------------------------------------------------
    @property
    def expected_step(self) -> Optional[str]:
        if self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.steps)

    @property
    def progress(self) -> dict[str, Any]:
        return {
            "completed": list(self.completed),
            "pending": self.steps[self.current_index:],
            "skipped": list(self.skipped),
            "current_step": self.expected_step,
            "step_number": min(self.current_index + 1, len(self.steps)),
            "total_steps": len(self.steps),
        }

    def next_step_suggestion(self) -> str:
        step = self.expected_step
        if step is None:
            return "Experiment complete. All steps finished."
        return f"Next, {NEXT_STEP_HINTS.get(step, step)}"

    # -- alerts / grace -----------------------------------------------------
    def _raise_alert(self, alert: Alert) -> None:
        self._pending_alerts.append(alert)

    def active_alerts(self, now: Optional[float] = None) -> list[Alert]:
        """Alerts whose grace period has passed. Call with a later timestamp
        (or after time passes) to promote pending alerts to active ones."""
        now = time.time() if now is None else now
        still_pending: list[Alert] = []
        for alert in self._pending_alerts:
            if now - alert.timestamp >= self.grace_seconds:
                self.alerts.append(alert)
            else:
                still_pending.append(alert)
        self._pending_alerts = still_pending
        return list(self.alerts)

    # -- core FSM -----------------------------------------------------------
    def ingest(
        self, activity: str, confidence: float, timestamp: Optional[float] = None
    ) -> StepResult:
        """Feed one *confirmed* activity (temporal confirmation already done).

        Low-confidence activities are HELD: recorded, but never flagged and
        never advancing the experiment.
        """
        ts = time.time() if timestamp is None else timestamp
        expected = self.expected_step

        if confidence < self.confidence_threshold:
            result = StepResult(
                activity=activity, status="HELD", confidence=confidence,
                timestamp=ts,
                message=f"Held: '{activity}' confidence {confidence:.2f} "
                        f"is below threshold {self.confidence_threshold:.2f}; "
                        "not counted, not flagged.",
                expected=expected, next_step=expected,
            )
            self.results.append(result)
            return result

        if activity not in self.steps:
            result = StepResult(
                activity=activity, status="UNKNOWN", confidence=confidence,
                timestamp=ts,
                message=f"Unknown activity '{activity}' — not part of the protocol.",
                expected=expected, next_step=expected,
            )
            self.results.append(result)
            self._raise_alert(Alert(
                kind="UNKNOWN", activity=activity, timestamp=ts,
                message=result.message,
                recovery="Ignore this action and continue with the next protocol step: "
                         f"{self.next_step_suggestion()}",
            ))
            return result

        if self.is_complete:
            result = StepResult(
                activity=activity, status="OUT_OF_ORDER", confidence=confidence,
                timestamp=ts,
                message=f"Experiment already complete; '{activity}' is extra.",
                expected=None, next_step=None,
            )
            self.results.append(result)
            return result

        if activity == expected:
            self.completed.append(activity)
            self.current_index += 1
            nxt = self.expected_step
            result = StepResult(
                activity=activity, status="CORRECT", confidence=confidence,
                timestamp=ts,
                message=f"Step completed: {activity} — {self.descriptions[activity]}.",
                expected=expected, next_step=nxt,
            )
            self.results.append(result)
            return result

        idx = self.steps.index(activity)
        if idx < self.current_index:
            # Already done — repeating an old step out of order.
            recovery = (f"'{activity}' was already completed. "
                        f"Return to the current step: {self.next_step_suggestion()}")
            result = StepResult(
                activity=activity, status="OUT_OF_ORDER", confidence=confidence,
                timestamp=ts,
                message=f"Out of order: '{activity}' was already completed; "
                        f"expected '{expected}'.",
                expected=expected, next_step=expected,
            )
            self.results.append(result)
            self._raise_alert(Alert(
                kind="OUT_OF_ORDER", activity=activity, timestamp=ts,
                message=result.message, recovery=recovery,
            ))
            return result

        # idx > current_index: a later step — intermediate step(s) skipped.
        missed = self.steps[self.current_index:idx]
        self.skipped.extend(missed)
        self.completed.append(activity)
        self.current_index = idx + 1
        nxt = self.expected_step
        missed_text = ", ".join(missed)
        recovery = (
            f"Skipped step(s): {missed_text}. If possible, go back and do "
            f"{missed[0]} now; otherwise confirm the skip with mission control "
            f"and continue: {self.next_step_suggestion()}"
        )
        result = StepResult(
            activity=activity, status="SKIPPED", confidence=confidence,
            timestamp=ts,
            message=f"Skipped step(s) {missed_text} — '{activity}' happened "
                    f"before them.",
            expected=expected, missed_steps=missed, next_step=nxt,
        )
        self.results.append(result)
        self._raise_alert(Alert(
            kind="SKIPPED", activity=activity, timestamp=ts,
            message=result.message, recovery=recovery, missed_steps=missed,
        ))
        return result

    def mark_incomplete(self, activity: str, timestamp: Optional[float] = None) -> StepResult:
        """Record an activity that started but did not finish."""
        ts = time.time() if timestamp is None else timestamp
        result = StepResult(
            activity=activity, status="INCOMPLETE", confidence=0.0, timestamp=ts,
            message=f"Incomplete: '{activity}' started but did not finish.",
            expected=self.expected_step, next_step=self.expected_step,
        )
        self.results.append(result)
        self._raise_alert(Alert(
            kind="INCOMPLETE", activity=activity, timestamp=ts,
            message=result.message,
            recovery=f"Repeat the step from the start: "
                     f"{self.next_step_suggestion()}",
        ))
        return result

    def recovery_guidance(self) -> str:
        """What to do after the most recent error, if any."""
        if not self.alerts and not self._pending_alerts:
            return self.next_step_suggestion()
        alert = (self.alerts or self._pending_alerts)[-1]
        return alert.recovery

    def reset(self) -> None:
        self.current_index = 0
        self.completed = []
        self.skipped = []
        self.results = []
        self._pending_alerts = []
        self.alerts = []
        self.start_time = time.time()
