"""
Pipeline Validator
==================
Validates pair CSVs and run JSON artifacts for schema correctness,
identity leakage, image path integrity, and metric soundness.

Usage:
    python scripts/pipeline_validator.py
"""

import os
import json
from dataclasses import dataclass, field

import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
DEFAULT_LFW_PATH  = "data/lfw"
DEFAULT_PAIRS_CSV = "configs/pairs_v4.csv"
DEFAULT_RUN_JSON  = "outputs/runs/run_005.json"
REQUIRED_COLUMNS  = ["left_identity", "right_identity",
                      "left_index", "right_index", "label", "split"]
VALID_SPLITS      = {"train", "val", "test"}
PATH_SAMPLE_SIZE  = 100

# 0.65 rather than 0.60: identity-capping naturally produces slight imbalance
# when rare identities have only one image and cannot form same-identity pairs.
BALANCE_THRESHOLD = 0.65
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Holds the outcome of a single validation check.
    passed  — True if no errors were found.
    errors  — list of human-readable problem descriptions.
    """
    name:   str
    passed: bool = True
    errors: list = field(default_factory=list)

    def fail(self, message: str):
        self.passed = False
        self.errors.append(message)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines  = [f"[{status}] {self.name}"]
        for e in self.errors:
            lines.append(f"       ✗ {e}")
        return "\n".join(lines)


class PipelineValidator:
    def __init__(self, lfw_path: str = DEFAULT_LFW_PATH):
        self.lfw_path = lfw_path

    # ── Pairs CSV checks ────────────────────────────────────────────────────

    def validate_pairs_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Checks required columns, binary labels, and valid split names."""
        result = ValidationResult("Pairs Schema")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            result.fail(f"Missing columns: {missing}")
            return result

        invalid_labels = df[~df["label"].isin([0, 1])]["label"].unique()
        if len(invalid_labels):
            result.fail(f"Non-binary label values: {list(invalid_labels)}")

        bad_splits = set(df["split"].unique()) - VALID_SPLITS
        if bad_splits:
            result.fail(f"Unrecognised split names: {bad_splits}")

        return result

    def validate_no_leakage(self, df: pd.DataFrame) -> ValidationResult:
        """Ensures no identity in train also appears in val or test."""
        result = ValidationResult("Identity Leakage")

        train_ids = (
            set(df[df["split"] == "train"]["left_identity"]) |
            set(df[df["split"] == "train"]["right_identity"])
        )
        eval_ids = (
            set(df[df["split"].isin(["val", "test"])]["left_identity"]) |
            set(df[df["split"].isin(["val", "test"])]["right_identity"])
        )

        leakage = train_ids & eval_ids
        if leakage:
            result.fail(
                f"{len(leakage)} identities overlap between train and eval "
                f"(sample: {list(leakage)[:5]})"
            )

        return result

    def validate_image_paths(self, df: pd.DataFrame) -> ValidationResult:
        """Spot-checks that the first PATH_SAMPLE_SIZE image files exist on disk."""
        result  = ValidationResult("Image Path Integrity")
        missing = []

        for _, row in df.head(PATH_SAMPLE_SIZE).iterrows():
            for side in ["left", "right"]:
                name = row[f"{side}_identity"]
                idx  = str(int(row[f"{side}_index"])).zfill(4)
                path = os.path.join(self.lfw_path, name, f"{name}_{idx}.jpg")
                if not os.path.exists(path):
                    missing.append(path)

        for p in missing[:5]:
            result.fail(f"Missing image: {p}")
        if len(missing) > 5:
            result.fail(f"... and {len(missing) - 5} more missing images")

        return result

    def validate_degenerate_pairs(self, df: pd.DataFrame) -> ValidationResult:
        """Checks for same-identity pairs where both image indices are identical."""
        result = ValidationResult("Degenerate Pairs")

        degen = df[
            (df["left_identity"] == df["right_identity"]) &
            (df["left_index"]    == df["right_index"])    &
            (df["label"] == 1)
        ]
        if len(degen):
            result.fail(
                f"{len(degen)} same-identity pairs reference identical indices "
                f"(image compared to itself)"
            )

        return result

    def validate_label_balance(self, df: pd.DataFrame) -> ValidationResult:
        """
        Warns if any split's majority class exceeds BALANCE_THRESHOLD.
        Threshold is 0.65 (not 0.60) because identity-capping naturally
        produces slight imbalance when some identities have only one image
        and cannot contribute same-identity pairs.
        """
        result = ValidationResult("Label Balance")

        for split_name in df["split"].unique():
            split_df = df[df["split"] == split_name]
            counts   = split_df["label"].value_counts()
            total    = len(split_df)
            if total == 0:
                continue
            majority_ratio = counts.iloc[0] / total
            if majority_ratio > BALANCE_THRESHOLD:
                result.fail(
                    f"Split '{split_name}' is imbalanced: "
                    f"{dict(counts)} ({majority_ratio:.1%} majority class)"
                )

        return result

    def validate_metrics(self, run_json_path: str) -> ValidationResult:
        """
        Checks that the run JSON is valid and contains the expected keys.

        Accepts all key schemas produced by the run scripts:
          Schema A (runs 001–004): val_sweep / test_metrics / best_threshold_from_val
          Schema B (integration):  val_sweep / best_val_metrics / best_threshold_from_val
          Schema C (run 005):      sweep / best_metrics / best_threshold
        """
        result = ValidationResult("Run Metrics")

        if not os.path.exists(run_json_path):
            result.fail(f"Run JSON not found: {run_json_path}")
            return result

        try:
            with open(run_json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result.fail(f"Invalid JSON: {e}")
            return result

        # ── Sweep key — accept any name used across runs ────────────────────
        sweep = data.get("val_sweep") or data.get("sweep")
        if sweep is None:
            result.fail("Missing sweep key (expected 'val_sweep' or 'sweep')")
        else:
            required_sweep_keys = {"threshold", "tp", "fp", "fn", "tn", "f1"}
            for i, entry in enumerate(sweep):
                missing_m = required_sweep_keys - set(entry.keys())
                if missing_m:
                    result.fail(f"Sweep entry {i} missing keys: {missing_m}")
                    break
                if not (0.0 <= entry["threshold"] <= 1.0):
                    result.fail(f"Sweep entry {i} threshold out of [0,1]: {entry['threshold']}")

        # ── Threshold key — accept any name used across runs ─────────────
        has_threshold = any(k in data for k in (
            "best_threshold_from_val", "best_threshold", "threshold"
        ))
        if not has_threshold:
            result.fail("Missing threshold key (expected 'best_threshold_from_val' or 'best_threshold')")

        # ── Metrics key — accept any name used across runs ────────────────
        metrics = (data.get("test_metrics") or data.get("best_val_metrics")
                   or data.get("best_metrics"))
        if metrics is None:
            result.fail("Missing metrics key (expected 'test_metrics', 'best_val_metrics', or 'best_metrics')")
        elif "f1" not in metrics:
            result.fail("Metrics block missing 'f1' key")
        elif not (0.0 <= metrics["f1"] <= 1.0):
            result.fail(f"F1 out of [0,1]: {metrics['f1']}")

        return result

    # ── Full validation run ──────────────────────────────────────────────────

    def validate_all(self, pairs_csv: str = DEFAULT_PAIRS_CSV,
                     run_json: str = DEFAULT_RUN_JSON) -> bool:
        """Runs all checks and prints a summary. Returns True if all passed."""
        print(f"Validating pairs CSV : {pairs_csv}")
        print(f"Validating run JSON  : {run_json}")
        print("─" * 52)

        df = pd.read_csv(pairs_csv)

        results = [
            self.validate_pairs_schema(df),
            self.validate_no_leakage(df),
            self.validate_image_paths(df),
            self.validate_degenerate_pairs(df),
            self.validate_label_balance(df),
            self.validate_metrics(run_json),
        ]

        all_passed = True
        for r in results:
            print(r)
            if not r.passed:
                all_passed = False

        print("─" * 52)
        print("RESULT:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
        return all_passed


if __name__ == "__main__":
    validator = PipelineValidator()
    validator.validate_all()