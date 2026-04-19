# Runs — Rolling Changelog

One line per numbered run. Newest at the bottom. Columns:

| Run | Date | Hypothesis | Offline F1 (test) | Decision |
|-----|------|------------|-------------------|----------|

Keys:
- **keep** — change merged; offline F1 held and live scenarios improved.
- **revert** — src diff reverted; artifacts preserved for reference.
- **inconclusive** — no clear win; see run's `notes` for next step.
- **historical** — pre-convention baseline retained for continuity.

See `docs/prompts/optimize_recognition.md` for the iteration protocol
and `outputs/runs/run_NNN.json` for the full artifact.

---

| Run     | Date       | Hypothesis                                                                 | Offline F1 | Decision   |
|---------|------------|----------------------------------------------------------------------------|------------|------------|
| run_001 | pre-convention | Baseline — no preprocessing, no data-centric improvements.              | 0.9702     | historical |
| run_002 | pre-convention | Uniform identity weighting on pair construction (`pairs_v2`).           | 0.9702     | historical |
| run_003 | pre-convention | Canonical deduplication of pairs (`pairs_v3`).                          | 0.9459     | historical |
| run_004 | pre-convention | Cap each identity to ≤ 3 pairs (`pairs_v4`) for diversity.              | 0.9577     | historical |
| run_005 | pre-convention | Center-crop 15 % inset before embedding to match FaceNet's FOV.         | 0.9893     | historical |
| run_006 | 2026-04-18 | Add `REMATCH_THRESHOLD = 0.60` for first-contact re-match; `MATCH_THRESHOLD = 0.70` retained for cross-track dedup (failure mode #1 — re-acquisition). | 0.9893 | inconclusive — awaiting live S1–S6 |
