"""
Smoke test — Milestone 3 inference path.

Runs the CLI end-to-end on a stubbed verifier (no FaceNet, no LFW) to confirm
the main path returns a well-formed JSON record with all contract fields.
Run: pytest tests/test_smoke_cli.py -v
"""

import io
import json
import sys

import numpy as np
import pytest


class _StubVerifier:
    """Drop-in replacement for FaceVerifier — skips embedding entirely."""
    threshold = 0.45

    def verify(self, left, right):
        from src.verifier import VerifyResult
        return VerifyResult(
            decision=1, score=0.81, confidence=0.94,
            threshold=self.threshold, latency_ms=12.3,
        )


def test_cli_single_pair_outputs_contract_fields(monkeypatch, tmp_path, capsys):
    # Create two dummy image files so path-existence checks pass.
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"x")
    img_b.write_bytes(b"x")

    # src.cli does `from src.verifier import FaceVerifier` at call time,
    # so patching src.verifier.FaceVerifier is enough.
    import src.cli
    import src.verifier
    monkeypatch.setattr(src.verifier, "FaceVerifier", lambda *a, **kw: _StubVerifier())

    rc = src.cli.main([
        "--left", str(img_a), "--right", str(img_b),
        "--calibration", str(tmp_path / "nope.json"),  # unused by stub
    ])
    assert rc == 0

    captured = capsys.readouterr().out.strip()
    record = json.loads(captured)
    for key in ("decision", "score", "confidence", "threshold", "latency_ms",
                "left", "right"):
        assert key in record, f"missing {key} in CLI output"
    assert record["decision"] in (0, 1)
    assert 0.0 <= record["confidence"] <= 1.0


def test_cli_rejects_missing_args(capsys):
    import src.cli
    with pytest.raises(SystemExit):
        src.cli.main([])  # no --left/--right and no --pairs
