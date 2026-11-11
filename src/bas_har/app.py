"""FastAPI backend for BAS-HAR using AI.

Endpoints:
  GET  /health
  POST /experiment/start
  POST /experiment/event      ingest one recognised activity
  GET  /experiment/status     current step, next step, alerts, confidence
  GET  /experiment/log        full JSON log
  POST /experiment/replay     replay a saved log through the FSM
  GET  /stream                streaming / recording info

State is held in memory (the FSM) plus SQLite (the log). Importing this
module without FastAPI installed must not crash: ``create_app()`` raises
a clear error in that case, and ``python -m src.bas_har.app`` prints it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .logging_store import ExperimentLogger, load_log, replay_log
from .protocol import ProtocolEngine
from .streaming import stream_info
from .voice import VoiceAssistant

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    HAS_FASTAPI = False


def require_fastapi() -> None:
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Install the demo requirements "
            "(pip install -r requirements-demo.txt) and retry."
        )


class EventIn(BaseModel):  # type: ignore[misc]
    activity: str
    confidence: float = 1.0


class ExperimentState:
    """In-memory experiment state shared by the endpoints."""

    def __init__(self, experiment_id: str = "EXP001") -> None:
        self.engine = ProtocolEngine()
        self.logger = ExperimentLogger(experiment_id=experiment_id)
        self.voice = VoiceAssistant(enabled=False)  # silent by default in API mode
        self.experiment_id = experiment_id
        self.started = False

    def start(self, experiment_id: str = "EXP001") -> dict[str, Any]:
        self.engine.reset()
        self.experiment_id = experiment_id
        self.logger = ExperimentLogger(experiment_id=experiment_id)
        self.started = True
        self.voice.say_key("experiment_start")
        return {"experiment_id": experiment_id, "status": "started",
                "next_step": self.engine.next_step_suggestion()}

    def ingest(self, activity: str, confidence: float) -> dict[str, Any]:
        result = self.engine.ingest(activity, confidence)
        step_no = self.engine.current_index if result.status == "CORRECT" else None
        self.logger.log_result(result, step=step_no)
        if result.status == "CORRECT":
            if self.engine.is_complete:
                self.voice.say_key("experiment_complete")
            else:
                self.voice.say_key(
                    "step_completed",
                    next_step=self.engine.next_step_suggestion())
        elif result.status == "HELD":
            self.voice.say_key("held")
        elif result.status == "SKIPPED":
            self.voice.say_key("skipped", missed=", ".join(result.missed_steps),
                               recovery=self.engine.recovery_guidance())
        elif result.status == "OUT_OF_ORDER":
            self.voice.say_key("wrong_order", recovery=self.engine.recovery_guidance())
        elif result.status == "UNKNOWN":
            self.voice.say_key("unknown", recovery=self.engine.recovery_guidance())
        return {
            "activity": result.activity, "status": result.status,
            "confidence": result.confidence, "message": result.message,
            "expected": result.expected, "missed_steps": result.missed_steps,
            "next_step": result.next_step,
        }

    def status(self) -> dict[str, Any]:
        alerts = self.engine.active_alerts(now=time.time())
        return {
            "experiment_id": self.experiment_id,
            "started": self.started,
            "is_complete": self.engine.is_complete,
            "progress": self.engine.progress,
            "next_step": self.engine.next_step_suggestion(),
            "recovery_guidance": self.engine.recovery_guidance(),
            "alerts": [
                {"kind": a.kind, "activity": a.activity, "message": a.message,
                 "recovery": a.recovery, "missed_steps": a.missed_steps}
                for a in alerts
            ],
            "last_results": [
                {"activity": r.activity, "status": r.status,
                 "confidence": r.confidence, "message": r.message}
                for r in self.engine.results[-10:]
            ],
            "voice_messages": list(self.voice.spoken[-5:]),
        }


STATE = ExperimentState()


def create_app() -> Any:
    require_fastapi()
    app = FastAPI(title="BAS-HAR using AI", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "offline": True, "experiment_id": STATE.experiment_id}

    @app.post("/experiment/start")
    def start(experiment_id: str = "EXP001") -> dict[str, Any]:
        return STATE.start(experiment_id)

    @app.post("/experiment/event")
    def event(payload: EventIn) -> dict[str, Any]:
        return STATE.ingest(payload.activity, payload.confidence)

    @app.get("/experiment/status")
    def status() -> dict[str, Any]:
        return STATE.status()

    @app.get("/experiment/log")
    def log() -> dict[str, Any]:
        return STATE.logger.to_dict()

    @app.post("/experiment/replay")
    def replay(path: str) -> dict[str, Any]:
        engine = ProtocolEngine()
        outcomes = replay_log(path, engine)
        return {"log": path, "outcomes": outcomes,
                "alerts": [a.message for a in engine.alerts]}

    @app.get("/stream")
    def stream() -> dict[str, Any]:
        return stream_info()

    return app


app = create_app() if HAS_FASTAPI else None

if __name__ == "__main__":
    require_fastapi()
    import uvicorn

    print("BAS-HAR using AI — API on http://127.0.0.1:8000  (offline, local only)")
    print("Docs: http://127.0.0.1:8000/docs")
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
