# CLAUDE.md
Guidance for Claude Code when working in this repo.

## Primary goal
**Improve the live face-recognition pipeline** on the real failure
modes: re-acquisition, crowded-scene crossings, pose / lighting
drift, fast-enroll noise, low-res surveillance.

## Project layout
```
src/             core library (all live-pipeline code + eval harnesses)
scripts/         live entry points + run scripts + plot tool
tests/           pytest (unit + synthetic + real-data harness tests)
web/             dashboard + live page (Flask static)
outputs/runs/    per-run JSON artifacts + rolling README changelog
outputs/plots/   per-run scenario bar charts
data/            enrollments/ + *.db + real_eval/ (scface/chokepoint)
docs/prompts/    durable prompts (optimize_recognition.md, improve_pose_and_distance.md)
docs/            real_data_eval.md (harness + dataset layout docs)
```
`data/enrollments/` is live-pipeline enrollment storage. `data/real_eval/`
holds the surveillance/pose datasets — gitignored, populated manually.

## Commands
```bash
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v                                       # unit + harness tests
python scripts/server.py                               # dashboard (127.0.0.1:5000)
python scripts/live_webcam.py                          # standalone live app
python -m scripts.models.run_NNN                       # run an experiment
python scripts/plot_run.py                             # (re)render run plots
```

## Architecture (live pipeline is the system)
- `src/similarity.py`  — cosine / Euclidean distance
- `src/quality.py`     — FIQA: sharpness, illumination, pose (tunable thresholds at top)
- `src/tracker.py`     — `FaceTracker` IoU greedy matching, stable `track_id`
- `src/gatekeeper.py`  — enrollment state machine, tracking lock, embedding bank. **The primary surface for optimization.** Knobs: `MATCH_THRESHOLD`, `REMATCH_THRESHOLD`, `ENROLL_THRESHOLD`, `CONSENSUS_FRAMES`, `MAX_EMBEDDINGS_PER_IDENTITY`.
- `src/database.py`    — SQLite identity store, async writer thread
- `src/memory.py`, `src/speech.py` — session events, optional transcription
- `src/alignment.py`   — `align_face` (2-pt/160, FaceNet) + `align_face_5point` (5-pt/112, ArcFace)
- `src/embedder.py`    — `get_embedder()` → ArcFace (default, buffalo_l ONNX) or FaceNet; `FACE_EMBEDDER` env toggles
- `src/live_eval.py`, `src/real_data_eval.py` — synthetic + real-data harnesses
- `scripts/server.py`  — Flask + camera + ML worker threads + REST + MJPEG
- `scripts/live_webcam.py` — same 7-stage pipeline, standalone OpenCV window
- `web/` — live page + identity-management dashboard (pauses pipeline on dashboard entry)

**Live pipeline (7 stages).** Detect (MTCNN) → Track (IoU) → Align
(2pt/5pt per embedder) → Quality gate → Embed (ArcFace 512-d,
ONNX, async) → Gatekeeper → Persist.

## Evaluation — two harnesses
- **Synthetic** (`src/live_eval.py`) — seconds, deterministic. Eight
  scenarios (S1–S8) feed scripted embedding streams into the real
  `Gatekeeper.process()`. Direction-of-travel signal on state-machine
  logic. S1 is the regression floor; S3 crossing the baseline failure.
- **Real-data** (`src/real_data_eval.py`) — ~10–15 min, full pipeline
  (MTCNN → align → `get_embedder()` → Gatekeeper) on SCface +
  ChokePoint. Absolute calibration — confirms a synthetic win holds
  on real faces. Layout + scenarios: `docs/real_data_eval.md`.

## Run convention (every pipeline change)
One **hypothesis per run**. Per run, land all of:
- `scripts/models/run_NNN.py` — thin wrapper calling `live_eval.run_all`
- `outputs/runs/run_NNN.json` — schema below
- `outputs/plots/run_NNN_scenarios.png` — via `scripts/plot_run.py`
- one line appended to `outputs/runs/README.md`

**JSON (additive).** `run_id`, `eval_type` (`"synth"` | `"real"`),
`created_at`, `git_sha`, `hypothesis`, `config_diff`
(`{KNOB: [before, after]}`), `knobs` (full snapshot), `scenarios`
(`[{id, trials, successes, success_rate, notes}]`), `aggregate`,
`decision` (`keep` / `revert` / `inconclusive`), `notes`. Real-data
runs additionally carry `datasets` listing the roots walked.

**Decision rule — keep only if** S1 floor stays ≥ 95 %, the targeted
scenario improves ≥ 20 % absolute, and no other scenario regresses
≥ 20 %. Otherwise revert the `src/` change; artifacts stay. Full
protocol: `docs/prompts/optimize_recognition.md`.

## Out of scope
No model training. No offline LFW pair eval — it's gone, don't
resurrect. No detector swap (MTCNN stays for now). Video-replay
harness is future work.
