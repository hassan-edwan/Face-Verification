# CLAUDE.md
Guidance for Claude Code when working in this repo.
## Project
Face verification pipeline: FaceNet embeddings + cosine similarity on
LFW. Offline evaluation (numbered runs) plus a real-time webcam +
Flask dashboard with two-stage enrollment, IoU tracking, FIQA.
## Commands
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v                                     # unit + integration
black .                                              # format
python scripts/plumbing/ingest_lfw.py                # data
python scripts/plumbing/make_pairs_v4.py             # current pair set
python -m scripts.models.run_005                     # most recent offline eval
python scripts/live_webcam.py                        # live app
python scripts/server.py                             # dashboard (127.0.0.1:5000)
```
## Architecture
- `src/similarity.py` — cosine / Euclidean distance
- `src/quality.py`    — FIQA: sharpness, illumination, pose
- `src/tracker.py`    — `FaceTracker` IoU greedy matching, stable `track_id`
- `src/gatekeeper.py` — enrollment state machine, tracking lock, embedding bank
- `src/database.py`   — SQLite identity store, async writer thread
- `scripts/server.py` — Flask + camera + ML threads + REST + MJPEG
- `web/`              — live page + identity-management dashboard

**Live (7 stages).** Detect (MTCNN) → Track (IoU) → Align → Quality gate
→ Embed (FaceNet 128-d, async) → Gatekeeper → Persist.
**Offline.** `configs/pairs_v{1..4}.csv` → `scripts/models/run_{001..005}.py`
→ `outputs/runs/run_NNN.json` → `scripts/plots/*`.
## Principles
Data-centric iteration · quality-gate early · tracking lock is permanent
per session · runs use module syntax (`python -m scripts.models.run_XXX`).
## Run convention (required for every pipeline change)
One **hypothesis per run**. Per run, land every artifact:
- `scripts/models/run_NNN.py` — reproducible entry
- `outputs/runs/run_NNN.json` — schema below
- `outputs/plots/run_NNN_<name>.png` — versioned, never overwrite
- one line in `outputs/runs/README.md` — rolling changelog
- one commit `run_NNN: <hypothesis>` containing artifacts + `src/` diff

**JSON (additive).** `run_id`, `created_at`, `git_sha`, `hypothesis`,
`config_diff` (`{KNOB: [before, after]}`),
`eval: {offline: {pairs_csv, best_threshold, val_metrics, test_metrics},
live: [{scenario, trials, successes, failures, notes}]}`,
`decision` (`keep` / `revert` / `inconclusive`), `notes`.

**Live scenarios.** S1 single-lock · S2 two-no-cross · S3 crossing
· S4 re-acquisition · S5 pose drift · S6 illumination. 5 trials each,
operator hands back counts. Offline F1 is the regression floor.

**Decision — keep only if** offline F1 drops ≤ 2 pts AND targeted live
scenario up ≥ 20 % AND no other scenario down ≥ 20 %. Otherwise revert
`src/` but keep artifacts. Full prompt: `docs/prompts/optimize_recognition.md`.
## Out of scope
No retraining / no embedder swaps. No enrichment or reverse-image
subsystem (removed — don't reintroduce). Don't mirror knob values here;
git blame over `src/` is authoritative.
