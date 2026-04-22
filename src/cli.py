"""
Command-line interface for the face verifier.

Single pair:
    python -m src.cli --left A.jpg --right B.jpg

Batch (CSV with columns: left_path,right_path,[label]):
    python -m src.cli --pairs pairs.csv

Each verified pair is printed as one JSON line with:
    decision         — 0 or 1
    score            — cosine similarity
    confidence       — Platt-calibrated P(same)
    latency_ms       — wall time for the pair

Exit code is 0 on success, non-zero on I/O or calibration errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Iterable


def _single(args, verifier) -> int:
    try:
        result = verifier.verify(args.left, args.right)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = result.as_dict()
    out["left"] = args.left
    out["right"] = args.right
    print(json.dumps(out))
    return 0


def _batch(args, verifier) -> int:
    with open(args.pairs, newline="") as f:
        reader = csv.DictReader(f)
        required = {"left_path", "right_path"}
        if not required.issubset(reader.fieldnames or []):
            print(
                f"error: --pairs CSV must have columns {sorted(required)}; "
                f"got {reader.fieldnames}",
                file=sys.stderr,
            )
            return 2
        rows: Iterable[dict] = list(reader)

    for row in rows:
        try:
            result = verifier.verify(row["left_path"], row["right_path"])
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        out = result.as_dict()
        out["left"] = row["left_path"]
        out["right"] = row["right_path"]
        if "label" in row and row["label"] != "":
            out["label"] = int(row["label"])
        print(json.dumps(out), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="face-verify", description="Embedding-based face verification CLI."
    )
    parser.add_argument("--left", help="Path to first image")
    parser.add_argument("--right", help="Path to second image")
    parser.add_argument("--pairs", help="CSV with left_path,right_path[,label]")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (default: from calibration.json)")
    parser.add_argument("--calibration", default="configs/calibration.json",
                        help="Path to calibration JSON")
    args = parser.parse_args(argv)

    has_single = bool(args.left and args.right)
    has_batch = bool(args.pairs)
    if has_single == has_batch:
        parser.error("provide either --left AND --right, or --pairs (not both, not neither)")

    from src.verifier import FaceVerifier
    verifier = FaceVerifier(threshold=args.threshold, calibration_path=args.calibration)

    if has_single:
        return _single(args, verifier)
    return _batch(args, verifier)


if __name__ == "__main__":
    sys.exit(main())
