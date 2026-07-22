"""Feline Grimace Scale scoring — pure functions.

The individual score_* functions are stubs that need to be populated by
porting the logic from grimaceguide/fgsScoreCalc.py. Do that one AU at a
time and add a unit test for each in tests/test_scoring.py.
"""
from __future__ import annotations

import time
import numpy as np

from grimaceguide.core.exceptions import ScoringError
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
    LandmarkSet,
)

PAIN_THRESHOLD = 0.39


def calculate_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def score_ears(landmarks: LandmarkSet) -> ActionUnitScore:
    raise NotImplementedError("Port ear-scoring logic from fgsScoreCalc.py")


def score_eyes(landmarks: LandmarkSet) -> ActionUnitScore:
    raise NotImplementedError("Port eye-scoring logic from fgsScoreCalc.py")


def score_muzzle(landmarks: LandmarkSet) -> ActionUnitScore:
    raise NotImplementedError("Port muzzle-scoring logic from fgsScoreCalc.py")


def score_whiskers(image: np.ndarray, landmarks: LandmarkSet) -> ActionUnitScore:
    raise NotImplementedError("Port whisker-scoring logic from fgsScoreCalc.py")


def score_head(landmarks: LandmarkSet) -> ActionUnitScore:
    raise NotImplementedError("Port head-alignment logic from fgsScoreCalc.py")


def compute_grimace_score(image: np.ndarray, landmarks: LandmarkSet) -> GrimaceResult:
    """Compute the full FGS score from an image + landmarks."""
    if image is None or image.size == 0:
        raise ScoringError("Empty image supplied to scoring.")
    if not landmarks.points:
        raise ScoringError("No landmarks supplied to scoring.")

    start = time.perf_counter()
    breakdown = ActionUnitBreakdown(
        ears=score_ears(landmarks),
        eyes=score_eyes(landmarks),
        muzzle=score_muzzle(landmarks),
        whiskers=score_whiskers(image, landmarks),
        head=score_head(landmarks),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return GrimaceResult.from_breakdown(
        breakdown=breakdown,
        landmarks=landmarks,
        processing_ms=elapsed_ms,
    )
