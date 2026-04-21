# Face Recognition — Live Pipeline

Real-time face recognition with FaceNet embeddings, IoU tracking,
FIQA quality gating, and a two-stage enrollment state machine.
Ships a standalone OpenCV window app and a Flask dashboard, plus a
deterministic synthetic harness for iterating on the gatekeeper's
state-machine logic.

## Setup

```bash
python -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Standalone live view (OpenCV window)
python scripts/live_webcam.py

# Dashboard + MJPEG stream at http://127.0.0.1:5000
python scripts/server.py

# Tests
pytest tests/ -v
```

Live enrollments persist to `data/enrollments/<Person N>/*.jpg` and
`data/faces.db`. Delete those directories/files to reset.

## Iterate on the model

The pipeline's behavior is controlled by a handful of threshold +
policy knobs in `src/gatekeeper.py`, `src/quality.py`, and
`src/tracker.py`. Every change follows the run convention:

1. Pick one failure mode + one hypothesis.
2. Copy the most recent `scripts/models/run_NNN.py` as `run_NNN+1.py`,
   edit the hypothesis and config diff, run it.
3. Land the artifacts alongside the `src/` change:
   `outputs/runs/run_NNN.json`, `outputs/plots/run_NNN_scenarios.png`,
   and a line in `outputs/runs/README.md`.

Full protocol and JSON schema: [`docs/prompts/optimize_recognition.md`](docs/prompts/optimize_recognition.md).
Project conventions and architecture: [`CLAUDE.md`](CLAUDE.md).

## Repo layout

```
src/                  core library (live pipeline + synthetic harness)
scripts/              live entry points (server.py, live_webcam.py),
                      run scripts (models/run_NNN.py), plot tool
web/                  dashboard + live page
tests/                pytest (unit + live_eval harness tests)
outputs/runs/         per-run JSON artifacts + rolling README changelog
outputs/plots/        per-run scenario bar charts
data/                 enrollments/ + SQLite stores (live-populated)
docs/prompts/         durable prompts (optimize_recognition.md)
```
