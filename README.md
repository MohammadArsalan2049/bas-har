# BAS-HAR using AI

**AI Human Activity Recognition for On-board BAS Experiments**

An independent project (not a competition entry). An AI assistant that
watches an astronaut performing an experiment on the Bharatiya Antariksh
Station (BAS), verifies whether they are following the correct procedure,
suggests the next step, warns about skipped or out-of-order steps, speaks
the warning out loud, and saves a timestamped log — all **offline**, on a
standalone system, with no internet connection and no cloud service.

Think of it as: CCTV camera + AI vision + experiment supervisor + voice
assistant + automatic experiment recorder, in one offline system.

## The problem, in simple language

Astronauts on BAS conduct experiments: handling instruments, moving
containers and samples, mixing or transferring liquids, operating lab
equipment, following a predefined sequence. But communication with Earth
is delayed, mission control cannot watch every movement, and an astronaut
can accidentally skip a step, do steps in the wrong order, or forget what
comes next. This system watches the experiment and checks the procedure
as it happens.

The reference experiment in this project: a coloured-container transfer.

| Step | Activity |
| --- | --- |
| 1 | Open the main box |
| 2 | Take out the red container |
| 3 | Take out the yellow container |
| 4 | Open the red container |
| 5 | Transfer its contents into the yellow container |
| 6 | Mix the contents |
| 7 | Close the containers |
| 8 | Put both containers back |

## Architecture

```
 webcam / simulated feed
        |
        v
 +------------------+     +-------------------+
 | vision.py        |     | activity.py       |
 | YOLO objects     | --> | hand+object feats |
 | MediaPipe pose/  |     | rule-based recog. |
 | hands, HSV colour|     | LSTM model (opt.) |
 +------------------+     +-------------------+
        |                          |
        v                          v
 +------------------+     +-------------------+
 | streaming.py     |     | protocol.py (FSM) |
 | local recording  |     | correct / skipped |
 | RTSP via ffmpeg  |     | out-of-order /    |
 | (independent of  |     | unknown / held    |
 |  inference)      |     +-------------------+
 +------------------+              |
                                   v
                 +----------------+----------------+
                 | voice.py (offline TTS,          |
                 | predefined messages only)       |
                 | logging_store.py (SQLite+JSON)  |
                 +----------------+----------------+
                                   |
                                   v
                 +----------------+----------------+
                 | app.py (FastAPI)  | dashboard.py |
                 | /health /status    | (Streamlit)  |
                 | /event /log /replay| live status, |
                 +----------------+----------------+
```

Key design choices:

- **Offline-first.** No cloud LLM, no internet at runtime. Voice is
  offline TTS (pyttsx3 / espeak) with predefined messages only.
- **Two recogniser paths, kept separate.** A deterministic rule-based
  recogniser (hand position + hand-object proximity + holding + motion)
  that works with no model, and a small PyTorch LSTM over feature
  sequences when a trained checkpoint exists.
- **The FSM does not trust a single frame.** An activity must clear a
  confidence threshold (default 0.70), persist for ~0.8 s (temporal
  confirmation), and a suspect error only becomes an active alert after
  a 2 s grace period.
- **Streaming/recording is independent of inference**, so a network
  problem never stops on-board monitoring.

## How to run

### Demo mode (no camera, no GPU, no heavy libraries) — the primary demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-demo.txt
python demo.py --scenario correct      # clean run
python demo.py --scenario skipped      # skips steps -> warning + log
python demo.py --scenario wrong-order  # yellow before red -> warning
```

Each run prints the voice messages, drives the FSM, and writes a
timestamped JSON + CSV log to `logs/`.

### Tests

```bash
python -m pytest tests/ -q
```

### API backend

```bash
python -m src.bas_har.app
# http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/experiment/start
curl -X POST http://127.0.0.1:8000/experiment/event \
  -H 'Content-Type: application/json' -d '{"activity":"OPEN_BOX","confidence":0.96}'
curl http://127.0.0.1:8000/experiment/status
```

### Dashboard

```bash
streamlit run dashboard.py                 # standalone demo mode
streamlit run dashboard.py -- --api http://127.0.0.1:8000   # live API mode
```

### With a real webcam (full install)

```bash
pip install -r requirements.txt
python scripts/record_dataset.py --label OPEN_BOX --clips 5 --seconds 4
python scripts/train_har.py --synthetic   # smoke test only
```

## Project structure

```
bas-har/
  README.md
  requirements.txt          full install (OpenCV, YOLO, MediaPipe, torch CPU)
  requirements-demo.txt     demo install only (numpy, FastAPI, Streamlit)
  demo.py                   primary demo: correct / skipped / wrong-order
  dashboard.py              Streamlit dashboard (demo + live API modes)
  src/bas_har/
    protocol.py             8-step protocol + FSM, temporal confirmer
    activity.py             feature extraction + rule-based recogniser
    models.py               PyTorch LSTM temporal model (lazy torch)
    vision.py               OpenCV capture, YOLO, MediaPipe, HSV colour,
                            hand-object interaction (all lazy, graceful fallback)
    voice.py                offline TTS (pyttsx3/espeak), predefined messages
    logging_store.py        SQLite + JSON logs, export, replay through FSM
    streaming.py            local recording + RTSP/ffmpeg (independent of inference)
    app.py                  FastAPI backend
  scripts/
    record_dataset.py       record webcam clips per activity label
    annotate.py             CSV annotation helper
    train_har.py            train the LSTM (synthetic smoke test or real clips)
  data/protocols/bas_experiment_01.json   the 8-step protocol as JSON
  tests/test_protocol.py    FSM tests (correct / skipped / wrong-order / held / recovery)
```

## Team roles (4 members)

| Member | Area |
| --- | --- |
| 1 — Computer Vision | YOLO detection, MediaPipe pose/hands, hand-object interaction |
| 2 — HAR & Dataset | Record/annotate clips, train LSTM/TCN, evaluate activity recognition |
| 3 — Backend & Protocol Engine | FSM, step validation, voice alerts, logs, streaming/recording |
| 4 — Dashboard & Integration | Streamlit dashboard, API integration, offline deployment, testing |

All four members should understand the whole architecture; the AI and
backend members must agree on the format of detected activities.

## Metrics (measured, not invented)

- **FSM tests:** `python -m pytest tests/ -q` — 8 tests, all passing
  (correct sequence, skipped, wrong-order, low-confidence held,
  unknown, temporal confirmation, recovery, protocol JSON load).
- **Demo runs:** correct / skipped / wrong-order scenarios, with logs
  in `logs/` — the difference between runs is the evidence, not a claim.
- **Model accuracy: to be measured on your recorded dataset.**
  No accuracy number is claimed in this project. Train with
  `scripts/train_har.py` and report the val accuracy it prints.
- Metrics to track once you have real clips: activity accuracy,
  precision/recall, sequence accuracy, error-detection rate,
  false-alarm rate, inference latency, FPS, offline reliability.
  Prioritise sequence accuracy, false alarms and latency over a single
  object-detection score.

## Honest limits

- **An Earth webcam demo cannot prove microgravity readiness.**
  In microgravity astronauts float, objects float, and a person can work
  upside down or sideways relative to the camera, so normal assumptions
  about body orientation break. This project starts with 2D pose and
  orientation-robust rules; 3D human mesh recovery (HMR) is future work,
  not a first milestone.
- The rule-based recogniser is a fallback, not a trained model. Its
  confidence is deliberately modest, and the FSM holds its weakest calls.
- The LSTM in `models.py` is untrained until you record a dataset and
  run `scripts/train_har.py`. The `--synthetic` option only tests the
  training loop; its accuracy is meaningless on real video.
- Voice is predefined messages only — safer and easier to validate than
  a generative assistant, but it cannot answer free-form questions.
- YOLO/MediaPipe are optional: if they are absent, the system falls back
  to HSV colour detection for the red/yellow containers, or to demo mode.
