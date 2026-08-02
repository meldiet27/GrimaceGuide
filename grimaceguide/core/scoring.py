"""Feline Grimace Scale scoring.

Per-AU geometry ported from the original grimaceguide.fgsScoreCalc prototype (since
removed), operating on typed Landmark/LandmarkSet objects instead of raw
{'x':.., 'y':..} dicts.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from grimaceguide.core.exceptions import ScoringError
from grimaceguide.core.landmarks import LANDMARK_LABELS
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
    Landmark,
    LandmarkSet,
)

PAIN_THRESHOLD = 0.39


def _label_landmarks(landmarks: LandmarkSet) -> dict[str, Landmark]:
    """Pair LandmarkSet points with their FGS labels for name-based lookup."""
    return dict(zip(LANDMARK_LABELS, landmarks.points))


def _distance(p1: Optional[Landmark], p2: Optional[Landmark]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return float(np.hypot(p1.x - p2.x, p1.y - p2.y))


def _vertical_angle(p1: Optional[Landmark], p2: Optional[Landmark]) -> Optional[float]:
    """Angle of the line p1-p2 relative to the vertical axis, in degrees."""
    if p1 is None or p2 is None:
        return None
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    if delta_y == 0:
        return 90.0
    angle_rad = np.arctan(delta_x / delta_y)
    return float(abs(np.degrees(angle_rad)))


def _estimate_face_width(labeled: dict[str, Landmark]) -> Optional[float]:
    return _distance(labeled.get("left_eye_1"), labeled.get("right_eye_1"))


def _score_ears(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = forward, 1 = slightly apart, 2 = flattened/rotated outwards."""
    le_base, le_tip, le_outer = labeled.get("left_ear_1"), labeled.get("left_ear_3"), labeled.get("left_ear_5")
    re_base, re_tip, re_outer = labeled.get("right_ear_1"), labeled.get("right_ear_3"), labeled.get("right_ear_5")

    # le_outer/re_outer are presence-gate only, not used in the math below (parity with legacy).
    if not (le_base and le_tip and le_outer and re_base and re_tip and re_outer):
        return ActionUnitScore.ABSENT

    left_ear_angle = _vertical_angle(le_base, le_tip)
    right_ear_angle = _vertical_angle(re_base, re_tip)
    avg_ear_angle = (
        (left_ear_angle + right_ear_angle) / 2
        if left_ear_angle is not None and right_ear_angle is not None else None
    )
    tip_dist = _distance(le_tip, re_tip)
    base_dist = _distance(le_base, re_base)
    tip_base_ratio = tip_dist / base_dist if base_dist and base_dist > 0 else None

    if avg_ear_angle is None or tip_base_ratio is None:
        return ActionUnitScore.ABSENT
    if avg_ear_angle > 45 or tip_base_ratio > 1.5:
        return ActionUnitScore.OBVIOUS
    if avg_ear_angle > 25 or tip_base_ratio > 1.2:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_eyes(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = open, 1 = partially open, 2 = squinted.

    eye_1..4 are a 4-point diamond: outer corner, inner corner, top, bottom (not
    two top points + two bottom points -- CatFLW-derived landmarks only supply
    one of each, so this reads the diamond directly: h_dist is the true eye
    width (outer-to-inner corner) and v_dist is the actual eyelid opening
    (top-to-bottom), rather than averaging corners into a fake top/bottom
    midpoint the way the original ported prototype logic did.
    """
    l_outer, l_inner, l_top, l_bottom = (labeled.get(f"left_eye_{i + 1}") for i in range(4))
    r_outer, r_inner, r_top, r_bottom = (labeled.get(f"right_eye_{i + 1}") for i in range(4))

    if not all([l_outer, l_inner, l_top, l_bottom, r_outer, r_inner, r_top, r_bottom]):
        return ActionUnitScore.ABSENT

    left_h_dist = _distance(l_outer, l_inner)
    left_v_dist = _distance(l_top, l_bottom)
    right_h_dist = _distance(r_outer, r_inner)
    right_v_dist = _distance(r_top, r_bottom)

    if left_v_dist is None or right_v_dist is None:
        return ActionUnitScore.ABSENT
    avg_v_dist = (left_v_dist + right_v_dist) / 2
    avg_h_dist = (left_h_dist + right_h_dist) / 2 if left_h_dist and right_h_dist else None

    if not avg_h_dist or avg_h_dist <= 0:
        return ActionUnitScore.ABSENT

    # Thresholds below are empirically calibrated, not from FGS literature --
    # neither of the two automated-FGS papers in README.md's references define
    # an explicit eye-openness formula/cutoff (both use end-to-end learned
    # models on raw landmarks instead of hand-crafted per-AU geometry), so
    # unlike PAIN_THRESHOLD=0.39 there's no published number to align to. See
    # docs/eye_scoring_calibration.md for the full derivation; summary: ratio
    # over all 2079 CatFLW photos has mean=0.75, median=0.77, min=0.271 (the
    # single most-squinted-looking cat in the dataset, visually confirmed).
    # OBVIOUS was raised from 0.25 to 0.30 (2026-08-02) so that real extreme
    # squinting actually reaches OBVIOUS instead of maxing out at MODERATE.
    eye_aspect_ratio = avg_v_dist / avg_h_dist
    if eye_aspect_ratio < 0.30:
        return ActionUnitScore.OBVIOUS
    if eye_aspect_ratio < 0.5:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_muzzle(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = relaxed round shape, 1 = mildly tense, 2 = tense elliptical shape."""
    nose_tip = labeled.get("nose_3")
    mouth_left = labeled.get("mouth_1")
    mouth_right = labeled.get("mouth_4")

    if not (nose_tip and mouth_left and mouth_right):
        return ActionUnitScore.ABSENT

    mouth_width = _distance(mouth_left, mouth_right)
    mouth_center = Landmark(
        x=(mouth_left.x + mouth_right.x) / 2,
        y=(mouth_left.y + mouth_right.y) / 2,
    )
    nose_to_mouth_dist = _distance(nose_tip, mouth_center)

    if not (mouth_width and nose_to_mouth_dist and nose_to_mouth_dist > 0):
        return ActionUnitScore.ABSENT

    muzzle_ratio = mouth_width / nose_to_mouth_dist
    if muzzle_ratio > 1.5:
        return ActionUnitScore.OBVIOUS
    if muzzle_ratio > 1.1:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_whiskers(labeled: dict[str, Landmark], face_width: Optional[float]) -> ActionUnitScore:
    """0 = loose/curved, 1 = slightly curved/straight, 2 = straight/forward-pointing."""
    lw1, lw3, lw5 = labeled.get("left_whisker_1"), labeled.get("left_whisker_3"), labeled.get("left_whisker_5")
    rw1, rw3, rw5 = labeled.get("right_whisker_1"), labeled.get("right_whisker_3"), labeled.get("right_whisker_5")
    # nose_base_l/nose_base_r are presence-gate only, not used in the math below (parity with legacy).
    nose_base_l, nose_base_r = labeled.get("nose_1"), labeled.get("nose_5")

    if not all([lw1, lw3, lw5, rw1, rw3, rw5, nose_base_l, nose_base_r]):
        return ActionUnitScore.ABSENT

    left_spread = abs(lw5.x - lw1.x)
    right_spread = abs(rw5.x - rw1.x)
    avg_spread = (left_spread + right_spread) / 2

    left_forward = lw5.x < lw1.x
    right_forward = rw5.x > rw1.x

    left_curve_proxy = lw3.y > (lw1.y + lw5.y) / 2
    right_curve_proxy = rw3.y > (rw1.y + rw5.y) / 2

    normalized_spread = avg_spread / face_width if face_width else None

    is_straight = not (left_curve_proxy or right_curve_proxy)
    is_forward = left_forward and right_forward

    if is_straight and is_forward:
        return ActionUnitScore.OBVIOUS
    if not is_straight and not is_forward:
        if normalized_spread and normalized_spread > 0.8:
            return ActionUnitScore.ABSENT
        return ActionUnitScore.MODERATE
    return ActionUnitScore.MODERATE


def _score_head(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = above shoulder line, 1 = aligned, 2 = below shoulder line/tilted down."""
    chin_point = labeled.get("chin_point")
    left_ear_base = labeled.get("left_ear_1")
    right_ear_base = labeled.get("right_ear_1")

    if not (chin_point and left_ear_base and right_ear_base):
        return ActionUnitScore.ABSENT

    avg_ear_base_y = (left_ear_base.y + right_ear_base.y) / 2
    chin_y = chin_point.y

    head_height_proxy = abs(chin_y - avg_ear_base_y)
    diff = chin_y - avg_ear_base_y
    normalized_diff = diff / head_height_proxy if head_height_proxy and head_height_proxy > 0 else 0

    if normalized_diff > 0.1:
        return ActionUnitScore.OBVIOUS
    if normalized_diff > -0.1:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def compute_grimace_score(
    image: Optional[np.ndarray],
    landmarks: LandmarkSet,
) -> GrimaceResult:
    """Compute the full FGS score from an image + landmarks."""
    if not landmarks.points:
        raise ScoringError("No landmarks supplied to scoring.")

    labeled = _label_landmarks(landmarks)

    start = time.perf_counter()
    face_width = _estimate_face_width(labeled)
    breakdown = ActionUnitBreakdown(
        ears=_score_ears(labeled),
        eyes=_score_eyes(labeled),
        muzzle=_score_muzzle(labeled),
        whiskers=_score_whiskers(labeled, face_width),
        head=_score_head(labeled),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return GrimaceResult.from_breakdown(
        breakdown=breakdown,
        landmarks=landmarks,
        processing_ms=elapsed_ms,
    )
