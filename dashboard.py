#!/usr/bin/env python3
"""Streamlit dashboard for BAS-HAR using AI.

    streamlit run dashboard.py                 # standalone demo mode
    streamlit run dashboard.py -- --api http://127.0.0.1:8000   # live API mode

Standalone demo mode reads a simulated log (or runs the demo scenarios
in-process) and needs no backend, camera or model. API mode polls the
FastAPI backend's /experiment/status and /experiment/log endpoints
(standard library only — no extra client dependency).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def fetch_status(api_base: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{api_base}/experiment/status", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main() -> None:
    import streamlit as st  # lazy: dashboard only

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api", default="")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="BAS-HAR using AI", layout="wide")
    st.title("BAS-HAR using AI — On-board Experiment Monitor")
    st.caption("Offline-first · Bharatiya Antariksh Station (BAS) · no cloud, no LLM")

    status = fetch_status(args.api) if args.api else None
    mode = "Live API" if status else "Standalone demo"
    st.sidebar.header("System status")
    st.sidebar.write(f"Mode: **{mode}**")
    st.sidebar.write("Offline operation: **yes**")
    st.sidebar.write("Model health: **rule-based fallback active** "
                     "(trained model: not yet trained — see scripts/train_har.py)")
    st.sidebar.write("Camera: **simulated** in demo mode")

    if status:
        progress = status["progress"]
        completed = progress["completed"]
        pending = progress["pending"]
        alerts = status.get("alerts", [])
        next_step = status.get("next_step", "")
        last = status.get("last_results", [])
        voice = status.get("voice_messages", [])
    else:
        # Standalone demo: run the correct scenario in-process.
        from demo import run_scenario  # noqa: E402

        summary = run_scenario("correct")
        from bas_har.protocol import STEP_NAMES  # noqa: E402

        completed = summary["completed"]
        pending = [s for s in STEP_NAMES if s not in completed]
        alerts = [{"kind": "—", "message": m} for m in summary["alerts"]]
        next_step = "Experiment complete." if summary["complete"] else "See log."
        last = [{"activity": a, "status": "CORRECT", "confidence": 0.93}
                for a in completed]
        voice = ["Step completed. Next, … (demo mode)"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Experiment progress")
        total = len(completed) + len(pending)
        st.progress(len(completed) / max(total, 1))
        st.write(f"Completed ({len(completed)}): " + (", ".join(completed) or "—"))
        st.write(f"Pending ({len(pending)}): " + (", ".join(pending) or "—"))
        st.subheader("Next instruction")
        st.info(next_step)
    with col2:
        st.subheader("Alert panel")
        if alerts:
            for a in alerts:
                st.error(f"{a.get('kind', '')}: {a.get('message', '')}")
        else:
            st.success("No alerts.")
        st.subheader("Voice messages")
        for m in voice:
            st.write(f"🔊 {m}")

    st.subheader("Activity timeline")
    if last:
        st.table([{"activity": r["activity"], "status": r["status"],
                   "confidence": r.get("confidence", "")} for r in last])
    else:
        st.write("No activities yet.")

    st.subheader("Experiment logs")
    logs = sorted(Path("logs").glob("*.json"))
    if logs:
        choice = st.selectbox("Log file", [str(p) for p in logs])
        st.download_button("Download log (JSON)", Path(choice).read_bytes(),
                           file_name=Path(choice).name)
        st.json(json.loads(Path(choice).read_text()))
    else:
        st.write("No logs yet — run `python demo.py --scenario correct` first.")

    st.subheader("Recording / streaming")
    st.write("Start/stop recording and streaming are controlled from the "
             "FastAPI backend (`/stream`) or from `src/bas_har/streaming.py`. "
             "Recording runs independently of inference.")


if __name__ == "__main__":
    main()
