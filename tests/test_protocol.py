"""Tests for the experiment FSM (protocol.py). No camera/ML needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bas_har.protocol import (  # noqa: E402
    ProtocolEngine, TemporalConfirmer,
)


def test_correct_sequence_passes():
    engine = ProtocolEngine()
    for step in ["OPEN_BOX", "PICK_RED", "PICK_YELLOW", "OPEN_RED",
                 "TRANSFER", "MIX", "CLOSE_CONTAINERS", "RETURN_CONTAINERS"]:
        r = engine.ingest(step, 0.95, timestamp=1000.0)
        assert r.status == "CORRECT", r
    assert engine.is_complete
    assert engine.active_alerts(now=2000.0) == []
    assert engine.next_step_suggestion().startswith("Experiment complete")


def test_skipped_step_flagged():
    engine = ProtocolEngine()
    engine.ingest("OPEN_BOX", 0.96, timestamp=1000.0)
    engine.ingest("PICK_RED", 0.93, timestamp=1002.0)
    r = engine.ingest("MIX", 0.88, timestamp=1004.0)
    assert r.status == "SKIPPED"
    assert r.missed_steps == ["PICK_YELLOW", "OPEN_RED", "TRANSFER"]
    assert engine.skipped == ["PICK_YELLOW", "OPEN_RED", "TRANSFER"]
    # Grace period: not active immediately, active after 2 s.
    assert engine.active_alerts(now=1004.5) == []
    alerts = engine.active_alerts(now=1007.0)
    assert len(alerts) == 1 and alerts[0].kind == "SKIPPED"
    assert "PICK_YELLOW" in alerts[0].recovery


def test_wrong_order_flagged():
    engine = ProtocolEngine()
    engine.ingest("OPEN_BOX", 0.96, timestamp=1000.0)
    r = engine.ingest("PICK_YELLOW", 0.93, timestamp=1002.0)
    assert r.status == "SKIPPED"  # PICK_RED missed
    assert r.missed_steps == ["PICK_RED"]
    # Repeating an already-completed step is out of order.
    engine2 = ProtocolEngine()
    engine2.ingest("OPEN_BOX", 0.96, timestamp=1000.0)
    r2 = engine2.ingest("OPEN_BOX", 0.96, timestamp=1002.0)
    assert r2.status == "OUT_OF_ORDER"


def test_low_confidence_held_not_flagged():
    engine = ProtocolEngine()
    r = engine.ingest("OPEN_BOX", 0.40, timestamp=1000.0)
    assert r.status == "HELD"
    assert engine.completed == []
    assert engine.expected_step == "OPEN_BOX"
    assert engine.active_alerts(now=2000.0) == []
    # The same step at good confidence then passes.
    r2 = engine.ingest("OPEN_BOX", 0.95, timestamp=1002.0)
    assert r2.status == "CORRECT"


def test_unknown_activity_flagged():
    engine = ProtocolEngine()
    r = engine.ingest("DANCING", 0.95, timestamp=1000.0)
    assert r.status == "UNKNOWN"
    assert engine.completed == []


def test_temporal_confirmer_holds_transient_detection():
    conf = TemporalConfirmer(confirm_seconds=0.8, min_confidence=0.70)
    assert conf.feed("OPEN_BOX", 0.95, 1000.0) is None  # first sighting
    assert conf.feed("OPEN_BOX", 0.95, 1000.5) is None  # only 0.5 s in
    assert conf.feed("OPEN_BOX", 0.95, 1000.9) == "OPEN_BOX"  # 0.9 s: confirmed
    # Low confidence resets it.
    conf2 = TemporalConfirmer()
    assert conf2.feed("OPEN_BOX", 0.40, 1000.0) is None
    assert conf2.feed("OPEN_BOX", 0.95, 1000.0) is None


def test_recovery_after_error():
    engine = ProtocolEngine()
    engine.ingest("OPEN_BOX", 0.96, timestamp=1000.0)
    engine.ingest("MIX", 0.88, timestamp=1002.0)  # skips 3 steps
    engine.active_alerts(now=1010.0)
    guidance = engine.recovery_guidance()
    assert "PICK_YELLOW" in guidance
    # The experiment can continue from the next step after the error.
    assert engine.expected_step == "CLOSE_CONTAINERS"
    r = engine.ingest("CLOSE_CONTAINERS", 0.92, timestamp=1012.0)
    assert r.status == "CORRECT"
    r = engine.ingest("RETURN_CONTAINERS", 0.94, timestamp=1014.0)
    assert r.status == "CORRECT"
    assert engine.is_complete


def test_protocol_loads_from_json():
    engine = ProtocolEngine.from_json("data/protocols/bas_experiment_01.json")
    assert engine.steps[0] == "OPEN_BOX"
    assert len(engine.steps) == 8
    assert engine.confidence_threshold == 0.70
