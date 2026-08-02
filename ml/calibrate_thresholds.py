"""Empirically calibrate the per-AU geometric thresholds in core/scoring.py.

Neither cited automated-FGS paper (see docs/au_threshold_calibration.md) defines an explicit
per-AU geometric formula, so there's no literature value for ears/muzzle/whiskers/head thresholds
to align to (same situation as eyes -- see docs/eye_scoring_calibration.md). This computes the
exact production ratio/angle formulas (imported directly from core/scoring.py, not reimplemented)
across every CatFLW ground-truth image, so thresholds can be checked against a real distribution
instead of guessed.

Usage:
    python -m ml.calibrate_thresholds --data-dir "ml/data/CatFLW dataset"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grimaceguide.core.landmarks import landmarks_from_catflw_array
from grimaceguide.core.scoring import (
    _ear_geometry,
    _estimate_face_width,
    _head_geometry,
    _label_landmarks,
    _muzzle_geometry,
    _whisker_geometry,
)


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _print_table(title: str, stats: dict[str, float]) -> None:
    print(f"\n=== {title} (n={stats['n']}) ===")
    for key in ("mean", "median", "std", "min", "p1", "p5", "p10", "p90", "p95", "p99", "max"):
        print(f"  {key:>6}: {stats[key]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    labels_dir = Path(args.data_dir) / "labels"
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {labels_dir}")

    ear_angles: list[float] = []
    ear_ratios: list[float] = []
    muzzle_ratios: list[float] = []
    whisker_spreads: list[float] = []
    whisker_branch_counts = {"OBVIOUS": 0, "MODERATE_via_mixed": 0, "MODERATE_via_neither": 0, "ABSENT": 0}
    head_diffs: list[float] = []
    skipped = {"ears": 0, "muzzle": 0, "whiskers": 0, "head": 0}

    for label_file in label_files:
        data = json.loads(label_file.read_text())
        flat = [coord for point in data["labels"] for coord in point]
        landmarks = landmarks_from_catflw_array(flat, image_width=1, image_height=1)
        labeled = _label_landmarks(landmarks)
        face_width = _estimate_face_width(labeled)

        ear_geom = _ear_geometry(labeled)
        if ear_geom is None:
            skipped["ears"] += 1
        else:
            ear_angles.append(ear_geom[0])
            ear_ratios.append(ear_geom[1])

        muzzle_ratio = _muzzle_geometry(labeled)
        if muzzle_ratio is None:
            skipped["muzzle"] += 1
        else:
            muzzle_ratios.append(muzzle_ratio)

        whisker_geom = _whisker_geometry(labeled, face_width)
        if whisker_geom is None:
            skipped["whiskers"] += 1
        else:
            normalized_spread, is_straight, is_forward = whisker_geom
            if normalized_spread is not None:
                whisker_spreads.append(normalized_spread)
            if is_straight and is_forward:
                whisker_branch_counts["OBVIOUS"] += 1
            elif not is_straight and not is_forward:
                if normalized_spread and normalized_spread > 0.24:
                    whisker_branch_counts["ABSENT"] += 1
                else:
                    whisker_branch_counts["MODERATE_via_neither"] += 1
            else:
                whisker_branch_counts["MODERATE_via_mixed"] += 1

        head_diff = _head_geometry(labeled, face_width)
        if head_diff is None:
            skipped["head"] += 1
        else:
            head_diffs.append(head_diff)

    print(f"Loaded {len(label_files)} label files. Skipped (missing landmarks): {skipped}")

    _print_table("ears: avg_ear_angle (degrees)", _percentiles(ear_angles))
    _print_table("ears: tip_base_ratio", _percentiles(ear_ratios))
    _print_table("muzzle: muzzle_ratio", _percentiles(muzzle_ratios))
    _print_table("whiskers: normalized_spread", _percentiles(whisker_spreads))
    print(f"\n=== whiskers: branch distribution (n={len(label_files) - skipped['whiskers']}) ===")
    for branch, count in whisker_branch_counts.items():
        print(f"  {branch}: {count}")
    _print_table("head: normalized_diff", _percentiles(head_diffs))


if __name__ == "__main__":
    main()
