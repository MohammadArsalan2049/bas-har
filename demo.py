#!/usr/bin/env python3
"""Primary demo path — no camera, no GPU, no heavy ML libraries.

    python demo.py --scenario correct
    python demo.py --scenario skipped
    python demo.py --scenario wrong-order

Runs the full pipeline with simulated recognised activities: the
rule-based recogniser's confirmed outputs drive the FSM, voice messages
are printed (and spoken if offline TTS is available), and a timestamped
JSON log is written to logs/.

The point of the three scenarios is the *difference* between them:
correct runs clean, skipped flags the missed steps, wrong-order flags
the out-of-sequence step — same code, same protocol, different behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bas_har.logging_store import ExperimentLogger  # noqa: E402
from bas_har.protocol import ProtocolEngine  # noqa: E402
from bas_har.voice import VoiceAssistant  # noqa: E402

SCENARIOS: dict[str, list[tuple[str, float]]] = {
    "correct": [
        ("OPEN_BOX", 0.96), ("PICK_RED", 0.93), ("PICK_YELLOW", 0.94),
        ("OPEN_RED", 0.91), ("TRANSFER", 0.90), ("MIX", 0.95),
        ("CLOSE_CONTAINERS", 0.92), ("RETURN_CONTAINERS", 0.94),
    ],
    "skipped": [
        # MIX before OPEN_RED / TRANSFER: PICK_YELLOW, OPEN_RED and
        # TRANSFER are skipped.
        ("OPEN_BOX", 0.96), ("PICK_RED", 0.93), ("MIX", 0.88),
        ("CLOSE_CONTAINERS", 0.92), ("RETURN_CONTAINERS", 0.94),
    ],
    "wrong-order": [
        # Yellow container picked before red.
        ("OPEN_BOX", 0.96), ("PICK_YELLOW", 0.93), ("PICK_RED", 0.91),
        ("OPEN_RED", 0.90), ("TRANSFER", 0.89), ("MIX", 0.95),
        ("CLOSE_CONTAINERS", 0.92), ("RETURN_CONTAINERS", 0.94),
    ],
}


def run_scenario(name: str, experiment_id: str = "EXP001") -> dict:
    engine = ProtocolEngine()
    voice = VoiceAssistant(enabled=False)  # print only; TTS if you enable it
    logger = ExperimentLogger(experiment_id=experiment_id)
    print(f"\n=== BAS-HAR using AI — demo scenario: {name} ===")
    print(f"Experiment {experiment_id} | protocol: 8 steps, red/yellow containers")
    voice.say_key("experiment_start")
    now = time.time()
    for i, (activity, confidence) in enumerate(SCENARIOS[name]):
        ts = now + i * 2.0
        result = engine.ingest(activity, confidence, timestamp=ts)
        # Promote alerts past their grace period so the demo shows them.
        engine.active_alerts(now=ts + 3.0)
        logger.log_result(result)
        icon = {"CORRECT": "OK ", "SKIPPED": "!! ", "OUT_OF_ORDER": "!! ",
                "UNKNOWN": "?? ", "HELD": "-- ", "INCOMPLETE": "!! "}.get(result.status, "   ")
        print(f"[{icon}] {activity:18s} conf={confidence:.2f}  {result.status:12s}  {result.message}")
        if result.status == "CORRECT":
            if engine.is_complete:
                voice.say_key("experiment_complete")
            else:
                voice.say_key("step_completed", next_step=engine.next_step_suggestion())
        elif result.status == "SKIPPED":
            voice.say_key("skipped", missed=", ".join(result.missed_steps),
                          recovery=engine.recovery_guidance())
        elif result.status == "OUT_OF_ORDER":
            voice.say_key("wrong_order", recovery=engine.recovery_guidance())

    print("\n--- Summary ---")
    print(f"Completed: {engine.completed}")
    print(f"Skipped:   {engine.skipped or 'none'}")
    print(f"Next:      {engine.next_step_suggestion()}")
    print(f"Alerts:    {[a.message for a in engine.alerts] or 'none'}")
    print(f"Complete:  {engine.is_complete}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"{stamp}_{name}.json"
    logger.export_json(log_path)
    logger.export_csv(log_path.with_suffix(".csv"))
    logger.close()
    print(f"Log written: {log_path}  (CSV alongside it)")
    return {
        "scenario": name, "completed": engine.completed,
        "skipped": engine.skipped, "alerts": [a.message for a in engine.alerts],
        "complete": engine.is_complete, "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BAS-HAR using AI — demo")
    parser.add_argument("--scenario", choices=list(SCENARIOS),
                        default="correct", help="demo scenario to run")
    parser.add_argument("--experiment-id", default="EXP001")
    args = parser.parse_args()
    summary = run_scenario(args.scenario, args.experiment_id)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
