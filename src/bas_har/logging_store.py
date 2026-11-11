"""Experiment logging: SQLite + JSON, with replay.

Event schema (one event per recognised activity):

    {
      "experiment_id": "EXP001",
      "start_time": "10:30:00",
      "events": [
        {"step": 1, "activity": "OPEN_BOX", "timestamp": "10:30:05",
         "status": "COMPLETED", "confidence": 0.96}
      ]
    }

``status`` is one of COMPLETED / SKIPPED / OUT_OF_ORDER / UNKNOWN /
INCOMPLETE / HELD, matching the FSM in ``protocol.py``.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentLogger:
    """Writes events to SQLite (``logs/experiments.db`` by default) and can
    export the same log as JSON or CSV."""

    def __init__(self, db_path: str | Path = "logs/experiments.db",
                 experiment_id: str = "EXP001") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.start_time = _now_iso()
        self._events: list[dict[str, Any]] = []
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT, step INTEGER, activity TEXT,
                timestamp TEXT, status TEXT, confidence REAL,
                message TEXT)"""
        )
        self._conn.commit()

    def log(self, step: Optional[int], activity: str, status: str,
            confidence: float, timestamp: Optional[str] = None,
            message: str = "") -> dict[str, Any]:
        event = {
            "step": step, "activity": activity,
            "timestamp": timestamp or _now_iso(),
            "status": status, "confidence": round(float(confidence), 3),
            "message": message,
        }
        self._events.append(event)
        self._conn.execute(
            "INSERT INTO events (experiment_id, step, activity, timestamp,"
            " status, confidence, message) VALUES (?,?,?,?,?,?,?)",
            (self.experiment_id, step, activity, event["timestamp"],
             status, event["confidence"], message),
        )
        self._conn.commit()
        return event

    def log_result(self, result: Any, step: Optional[int] = None) -> dict[str, Any]:
        """Log a ``StepResult`` from the FSM directly. If no step number
        is given, derive it from the protocol order (1-based)."""
        if step is None:
            try:
                from .protocol import STEP_NAMES

                step = STEP_NAMES.index(result.activity) + 1
            except ValueError:
                step = None
        return self.log(
            step=step, activity=result.activity, status=result.status,
            confidence=result.confidence,
            timestamp=datetime.fromtimestamp(
                result.timestamp, tz=timezone.utc).isoformat(timespec="seconds"),
            message=result.message,
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "events": self._events,
        }

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["step", "activity", "timestamp", "status",
                               "confidence", "message"])
            w.writeheader()
            w.writerows(self._events)
        return path

    def close(self) -> None:
        self._conn.close()


def load_log(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay_log(path: str | Path, engine: Any) -> list[dict[str, Any]]:
    """Replay a saved JSON log through a fresh FSM and return the verdict
    for each event. Used by the dashboard replay view and by ``/experiment/replay``."""
    from .protocol import ProtocolEngine  # local import: keeps this module light

    log = load_log(path)
    if engine is None:
        engine = ProtocolEngine()
    engine.reset()
    outcomes: list[dict[str, Any]] = []
    base = time.time()
    for i, event in enumerate(log.get("events", [])):
        result = engine.ingest(
            event["activity"], float(event.get("confidence", 1.0)),
            timestamp=base + i * 2.0,
        )
        # Promote any alerts past their grace period so replay shows them.
        engine.active_alerts(now=base + i * 2.0 + 3.0)
        outcomes.append({
            "activity": event["activity"], "status": result.status,
            "message": result.message, "next_step": result.next_step,
        })
    return outcomes
